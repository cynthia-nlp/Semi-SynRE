# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import json
import os
import random
import re
import warnings
from collections import Counter

import torch
from tqdm import tqdm
from typing import List, Optional

import fire
from llama import Llama, Dialog

warnings.filterwarnings('ignore')


def set_seed(seed):
    """Sets the relevant random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        prompt_path: str,
        demo_path: str,
        data_path: str,
        data_size: int,
        k: int,
        min_len: int,
        max_len: int,
        entity_len: int,
        entity_cnt: int,
        batch_size: int,
        seed: int = 42,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_seq_len: int = 1024,
        max_batch_size: int = 64,
        max_gen_len: Optional[int] = None
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        prompt_path (str): Prompt template path.
        demo_path (str): Demonstration path.
        data_path (str): Synthetic data save path.
        data_size (int): Synthetic data size.
        k (int): Number of demonstration.
        min_len (int): The minimum length of the sentence.
        max_len (int): The maximum length of the sentence.
        entity_len (int): The maximum length of the entity.
        entity_cnt (int): The maximum number of the entity.
        batch_size (int): batch size.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        seed (int, optional): random seed.
    """
    set_seed(seed)
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        rel = json.load(f)
    prompts = rel['prompt']
    for dataset in ['semeval', 'tacred', 'tacrev', 'retacred']:
        print('-' * 20 + f'\n\n{dataset}\n\n' + '-' * 20)
        rel_desc = rel[dataset]
        with open(demo_path, 'r', encoding='utf-8') as f:
            demo_seed = json.load(f)

        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                ori_data = json.load(f)
        else:
            ori_data = {label: [] for label in demo_seed.keys()}

        for label, sentences in tqdm(ori_data.items()):
            cnt = 0
            nums = len(sentences)
            head = dict(Counter(sentence['h'] for sentence in sentences))
            tail = dict(Counter(sentence['t'] for sentence in sentences))
            while nums < data_size:
                dialogs: List[Dialog] = []
                for i in range(batch_size):
                    demos = random.sample(demo_seed[label], k=min(len(demo_seed[label]), k))
                    random.shuffle(demos)
                    examples = '\n'.join(f"{d['sent']} (subj: {d['h']}, obj: {d['t']})" for d in demos)
                    dialogs.append([
                        {"role": "system",
                         "content": f"{random.choice(prompts[0])}"},
                        {"role": "user",
                         "content": f"Write some sentences that {rel_desc[label]} Here are some "
                                    f"examples:\n\n{examples}\n\n{random.choice(prompts[1])}"}
                    ])
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                enough = False
                for result in results:
                    if enough:
                        break
                    res = result['generation']['content'].split('\n')
                    res = [r.strip() for r in res if r]
                    res = [re.sub(r'^[^a-zA-Z]+', '', r) for r in res if re.match(r'^\d+', r)]
                    for r in res:
                        match = re.match(r'(.+?)\(subj: (.+?), obj: (.+?)\).*?', r)
                        if match:
                            s = match.group(1).strip()
                            h = match.group(2).strip()
                            t = match.group(3).strip()
                            if min_len <= len(s.split()) <= max_len:
                                if len(h.split()) <= entity_len and len(t.split()) <= entity_len:
                                    if s.lower().find(h.lower()) != -1 and s.lower().find(t.lower()) != -1:
                                        h_nums = 0 if h not in head else head[h]
                                        t_nums = 0 if t not in tail else tail[t]
                                        if h_nums < entity_cnt and t_nums < entity_cnt:
                                            head[h] = h_nums + 1
                                            tail[t] = t_nums + 1
                                            sentences.append({'sent': s, 'h': h, 't': t})
                                            nums += 1
                                            if nums >= data_size:
                                                enough = True
                                                break
                # nums = len(sentences)
                print(nums)
                cnt += 1
                if cnt % 5 == 0:
                    with open(data_path, 'w', encoding='utf-8') as f:
                        json.dump(ori_data, f)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(ori_data, f)


if __name__ == "__main__":
    fire.Fire(main)
