import json
import os
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup

from util.data_processing import create_mini_batch, REDataset
from util.eval import evaluate
from util.model import REBertModel
from util.sampling import *
from util.util import set_seed, set_logger, load_data, data_process, merge


def set_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--epochs", type=int, help='training epochs')
    parser.add_argument("--lr", type=float, default=5e-6, help='learning rate')
    parser.add_argument("--warm_up", type=float, default=0.1, help='warm_up rate')
    parser.add_argument("--sample", type=str, choices=['random', 'topk', 'fixed', 'adaptive'], help='sampling strategy')
    parser.add_argument("--step", type=int, help='i-th iteration')
    parser.add_argument("--total_step", type=int, help='number of iterations')
    parser.add_argument("--k", type=int, default=7, help='number of samples')

    # file_path
    parser.add_argument("--syn_data_path", type=str, help='synthetic data path')
    parser.add_argument("--real_data_path", type=str, help='real data path')
    parser.add_argument("--demos_path", type=str, help='demonstrations path')
    parser.add_argument("--model_path", type=str, help='pretrained token model path')
    parser.add_argument("--log_path", type=str, help='log save path')
    parser.add_argument("--dataset", type=str, choices=['semeval', 'tacred', 'tacrev', 'retacred'],
                        help='original dataset')
    parser.add_argument("--pretrained_model", type=str, default='bert-large-uncased',
                        help='huggingface pretrained model')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = set_args()

    # set seed
    set_seed(args.seed)
    log_path = f"{args.log_path}/{args.dataset}/step{args.step}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = set_logger(f"{log_path}/{args.dataset}_{args.sample}.log")
    logger.info(f"dataset: {args.dataset}")
    logger.info(f"pretrained_model: {args.pretrained_model}")
    logger.info(f"sampling strategy: {args.sample}")

    # syn_data_path = f"{args.syn_data_path}/{args.dataset}-{args.sample}.json"
    if args.step == 1:
        syn_data_path = f"{args.syn_data_path}/step{args.step - 1}/{args.dataset}.json"
    else:
        syn_data_path = f"{args.syn_data_path}/step{args.step - 1}/{args.dataset}-{args.sample}.json"
    real_data_path = f"{args.real_data_path}/{args.dataset}.csv"
    train_data, train_labels = load_data("syn", syn_data_path)
    test_data, test_labels = load_data("real", real_data_path)
    train_data = {k: v for k, v in train_data.items() if k in test_labels}
    labels = list(set(test_labels))
    train_data = data_process(train_data)
    relation2idx = {relation: idx for idx, relation in enumerate(labels)}
    idx2relation = dict((v, k) for k, v in relation2idx.items())
    if args.dataset == 'semeval':
        nota = relation2idx['Other']
    else:
        nota = relation2idx['no_relation']

    # print info
    logger.info(f'seed: {args.seed}, lr: {args.lr}, epochs: {args.epochs}, batch_size: {args.batch_size}')
    logger.info('the lengths of label is {} '.format(len(labels)))
    logger.info('the lengths of train data is {} '.format(len(train_data)))
    logger.info('the lengths of test data is {} '.format(len(test_data)))

    model_path = f"{args.model_path}/{args.pretrained_model}"
    trainset = REDataset(args, 'syn', 'train', train_data, relation2idx, model_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=True)

    # To evaluate the inference time
    test_batchsize = 10 * args.batch_size
    testset = REDataset(args, 'real', 'test', test_data, relation2idx, model_path)
    testloader = DataLoader(testset, batch_size=test_batchsize, collate_fn=create_mini_batch, shuffle=False)
    test_labels = np.array([relation2idx[t['label']] for t in test_data])

    # load model
    config = AutoConfig.from_pretrained(model_path, num_labels=len(set(labels)))
    config.pretrained_model = model_path
    model = REBertModel.from_pretrained(model_path, config=config)
    model = model.cuda()

    # optimizer and scheduler
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(trainset) * args.epochs // args.batch_size
    warmup_steps = num_training_steps * args.warm_up
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    for epoch in range(args.epochs):
        logger.info(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
        running_loss = 0.0
        for step, data in enumerate(trainloader):
            tokens_tensors, attention_mask, marked_e1, marked_e2, labels_ids = [t.cuda() for t in data]
            optimizer.zero_grad()
            loss = model(
                input_ids=tokens_tensors,
                attention_mask=attention_mask,
                e1_mask=marked_e1,
                e2_mask=marked_e2,
                target=labels_ids
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 100 == 0:
                logger.info(f'[step {step}] running_loss:{running_loss / (step + 1)}')

    logger.info(f'============== SAMPLE PSEUDO LABELS ==============')
    logits = torch.softmax(logits, dim=-1)
    if args.sample == 'random':
        u_psl_mask = random_sample(logits, args.k)
    elif args.sample == 'topk':
        u_psl_mask = topk_sample(logits, args.k)
    elif args.sample == 'fixed':
        u_psl_mask = fixed_threshold(logits)
    elif args.sample == 'adaptive':
        cw_avg_prob = np.load(f'threshold/{args.dataset}/threshold-{args.step - 1}.npy')
        cw_avg_prob = torch.tensor(cw_avg_prob)
        u_psl_mask, cw_avg_prob = adaptive_threshold_adjustment(logits, cw_avg_prob)
        logger.info(f'{cw_avg_prob}')
        np.save(f'threshold/{args.dataset}/threshold-{args.step}.npy', cw_avg_prob.cpu().numpy())
    else:
        raise ValueError("No such sample strategy!")
    sample_data = [test_data[i] for i in u_psl_mask]
    pseudo_labels = torch.argmax(logits[u_psl_mask], dim=-1)
    pseudo_labels = [idx2relation[p.item()] for p in pseudo_labels]
    demos = [{'sent': k['sents'], 'h': k['ent1'], 't': k['ent2'],
              'gold_label': k['label'], 'pseudo_label': v} for k, v in zip(sample_data, pseudo_labels)]
    demos_path = f'{args.demos_path}/{args.dataset}/step{args.step}'
    if not os.path.exists(demos_path):
        os.mkdir(demos_path)
    logger.info(f'{demos_path}')
    with open(f'{demos_path}/{args.dataset}-demos-{args.sample}.json', 'w', encoding='utf-8') as f:
        json.dump(demos, f)
    merge(args.dataset, args.step, args.sample, f'{args.demos_path}/{args.dataset}')

    if args.step == args.total_step:
        logger.info(f'============== EVALUATION ON TEST DATA ==============')
        preds = logits.cpu().argmax(axis=-1)
        preds = np.array(preds)
        results = evaluate(preds, test_labels, nota)
        logger.info(
            f'NoTA: precision: {results["NoTA"][0] * 100:.2f}, recall: {results["NoTA"][1] * 100:.2f}, f1 score: {results["NoTA"][2] * 100:.2f}')
        logger.info(
            f'micro: precision: {results["micro"][0] * 100:.2f}, recall: {results["micro"][1] * 100:.2f}, f1 score: {results["micro"][2] * 100:.2f}')
        logger.info(
            f'macro: precision: {results["macro"][0] * 100:.2f}, recall: {results["macro"][1] * 100:.2f}, f1 score: {results["macro"][2] * 100:.2f}')
        logger.info(
            f'macro*: precision: {results["macro*"][0] * 100:.2f}, recall: {results["macro*"][1] * 100:.2f}, f1 score: {results["macro*"][2] * 100:.2f}')
