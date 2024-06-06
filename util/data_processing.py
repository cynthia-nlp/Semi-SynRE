import re

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def create_mini_batch(samples):
    # all of here are positive samples
    tokens_tensors = [s[0] for s in samples]
    attention_mask = [s[1] for s in samples]
    marked_e1_tensor = [s[2] for s in samples]
    marked_e2_tensor = [s[3] for s in samples]
    if samples[0][4] is not None:
        label_ids = torch.stack([s[4] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    marked_e1_tensor = pad_sequence(marked_e1_tensor, batch_first=True)
    marked_e2_tensor = pad_sequence(marked_e2_tensor, batch_first=True)

    return tokens_tensors, attention_mask, marked_e1_tensor, marked_e2_tensor, label_ids


def mark_entity(new_pos, sent_len):
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    marked_e1[new_pos[0]] = 1
    marked_e2[new_pos[1]] = 1
    return torch.tensor(marked_e1, dtype=torch.long), torch.tensor(marked_e2, dtype=torch.long)


class REDataset(Dataset):
    def __init__(self, args, genre, mode, data, relation2idx, model_path):
        assert genre in ['syn', 'real']
        self.genre = genre
        self.mode = mode
        self.data = data
        self.relation2idx = relation2idx
        self.len = len(data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 256
        self.head_mark_ids = 1001
        self.tail_mark_ids = 1030

    def __getitem__(self, idx):
        single_data = self.data[idx]
        if self.genre == 'syn':
            sents = single_data['sent']
            pos1 = sents.find(single_data['h'])
            pos2 = sents.find(single_data['t'])
            if pos1 < pos2:
                sentence = single_data['sent'][:pos1] + ' # ' + single_data['h'] + ' # ' \
                           + single_data['sent'][pos1 + len(single_data['h']):pos2] \
                           + ' @ ' + single_data['t'] + ' @ ' \
                           + single_data['sent'][pos2 + len(single_data['t']):]
            else:
                sentence = single_data['sent'][:pos2] + ' @ ' + single_data['t'] + ' @ ' \
                           + single_data['sent'][pos2 + len(single_data['t']):pos1] \
                           + ' # ' + single_data['h'] + ' # ' \
                           + single_data['sent'][pos1 + len(single_data['h']):]

        else:
            pos1 = single_data['masked_sents'].find('ENT1')
            pos2 = single_data['masked_sents'].find('ENT2')

            if pos1 < pos2:
                sentence = single_data['masked_sents'][:pos1] + ' # ' + single_data['ent1'] + ' # ' \
                           + single_data['masked_sents'][pos1 + 4:pos2] + ' @ ' + single_data['ent2'] + ' @ ' \
                           + single_data['masked_sents'][pos2 + 4:]
            else:
                sentence = single_data['masked_sents'][:pos2] + ' @ ' + single_data['ent2'] + ' @ ' \
                           + single_data['masked_sents'][pos2 + 4:pos1] + ' # ' + single_data['ent1'] + ' # ' \
                           + single_data['masked_sents'][pos1 + 4:]

        sentence = re.sub(r'\s+', ' ', sentence).strip()
        tokens_info = self.tokenizer(sentence)
        tokens_ids = tokens_info['input_ids']
        attention_mask = torch.tensor(tokens_info['attention_mask'])
        # for roberta
        if pos2 == 0:
            tokens_ids[1] = self.tail_mark_ids
        elif pos1 == 0:
            tokens_ids[1] = self.head_mark_ids

        new_head_pos = tokens_ids.index(self.head_mark_ids)
        new_tail_pos = tokens_ids.index(self.tail_mark_ids)
        new_pos = (new_head_pos, new_tail_pos)
        marked_e1, marked_e2 = mark_entity(new_pos, len(tokens_ids))

        tokens_ids = torch.tensor(tokens_ids)
        if self.mode == 'train':
            label_idx = int(self.relation2idx[single_data['label']])
            label_idx_tensor = torch.tensor(label_idx)
        else:
            label_idx_tensor = None

        return tokens_ids, attention_mask, marked_e1, marked_e2, label_idx_tensor

    def __len__(self):
        return self.len
