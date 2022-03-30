# -*- coding: utf-8 -*-
# source code: https://github.com/laihuiyuan/pre-trained-formality-transfer/blob/main/classifier/test.py
import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import T5Tokenizer
from transformers import BartTokenizer

sys.path.append("")
from utils.dataset import SCIterator
from classifier.textcnn import TextCNN

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = 'cuda' if cuda.is_available() else 'cpu'
special_tokens = [{'bos_token': '<bos>'},
                  {'eos_token': '<eos>'}, {'sep_token': '<sep>'},
                  {'pad_token': '<pad>'}, {'unk_token': '<unk>'}]
tokenizer = T5Tokenizer.from_pretrained('t5-base')



def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('-order', default=0, type=str, help='order')
    parser.add_argument('-style', default=0, type=int, help='from 0 to 1')
    parser.add_argument('-max_len', default=50, type=int, help='max tokens in a batch')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-dataset', default='em', type=str, help='the name of dataset')
    parser.add_argument('-model', default='textcnn', type=str, help='the name of model')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=32, type=int, help='max sents in a batch')
    parser.add_argument("-dropout", default=0.5, type=float, help="Keep prob in dropout")
    parser.add_argument("-file", type=str)
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    test_src, test_tgt = [], []

    if opt.file is not None:
        with open(opt.file, 'r') as f:
            for line in f.readlines():
                test_tgt.append(tokenizer.encode(line.strip())[:opt.max_len])

    print('[Info] {} instances from src test set'.format(len(test_src)))
    print('[Info] {} instances from tgt test set'.format(len(test_tgt)))
    test_loader = SCIterator(test_src, test_tgt, opt, tokenizer.pad_token_id, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes,
                    num_filters, None, dropout=opt.dropout)
    model.to(device).eval()
    model.load_state_dict(torch.load('./checkpoints/t5_textcnn_{}.chkpt'.format(
        opt.dataset)))

    total_num = 0.
    total_acc = 0.
    total_loss = 0.
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            print(F.softmax(logits, dim=-1))
            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

    print('Test: {}'.format('acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))


if __name__ == '__main__':
    main()
