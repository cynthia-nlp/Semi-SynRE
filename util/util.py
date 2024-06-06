import logging
import json
import os
import random
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string


def set_seed(seed):
    """Sets the relevant random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    logger = logging.getLogger('Relation Extraction')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.DEBUG)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHeader)
    return logger


def load_data(genre, data_path):
    # load data
    assert genre in ['syn', 'real']
    if genre == 'syn':
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label = list(data.keys())
    else:
        df = pd.read_csv(data_path, sep='\t')
        label = list(set(df['label']))
        data = df[['ent1', 'ent2', 'sents', 'masked_sents', 'label']].to_dict('records')

    return data, label


def data_process(raw_data, sample_size=0):
    data = []
    for k, v in raw_data.items():
        data_filter = v
        if sample_size != 0:
            data_filter = random.sample(v, k=min(len(v), sample_size))
        data.extend({'h': clean_str(s['h']), 't': clean_str(s['t']),
                     'sent': clean_str(s['sent']), 'label': k} for s in data_filter)
    return data


def merge(dataset, step, sample, demo_path):
    with open(f'{demo_path}/step{step - 1}/{dataset}-demos-{sample}.json', 'r', encoding='utf-8') as f:
        demos = json.load(f)
    old_demos = [{'sent': s['sent'], 'h': s['h'], 't': s['t'], 'label': k} for k, v in demos.items() for s in v]

    with open(f'{demo_path}/step{step}/{dataset}-demos-{sample}.json', 'r', encoding='utf-8') as f:
        pseudo = json.load(f)
    for p in pseudo:
        repeat = False
        for idx, demo in enumerate(old_demos):
            if demo['sent'] == p['sent'] and demo['h'] == p['h'] and demo['t'] == p['t']:
                old_demos[idx]['label'] = p['pseudo_label']
                repeat = True
                break
        if not repeat:
            old_demos.append({'sent': p['sent'], 'h': p['h'], 't': p['t'], 'label': p['pseudo_label']})
    new_demos = {}
    for demo in old_demos:
        if demo['label'] in new_demos:
            new_demos[demo['label']].append({'sent': demo['sent'], 'h': demo['h'], 't': demo['t']})
        else:
            new_demos[demo['label']] = [{'sent': demo['sent'], 'h': demo['h'], 't': demo['t']}]
    with open(f'{demo_path}/step{step}/{dataset}-demos-{sample}.json', 'w', encoding='utf-8') as f:
        json.dump(new_demos, f)
