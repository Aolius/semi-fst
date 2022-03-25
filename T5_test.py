# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import nlpaug.augmenter.word as naw
import nltk
import random


import torch
from torch import cuda

from utils.T5_dataset import T5Dataset
from torch.utils.data import DataLoader
from utils.dataset import  SCIterator
from utils.nltk_bleu import evaluate_bleu


device = 'cuda' if cuda.is_available() else 'cpu'

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def test(model, tokenizer, cls, cls_tokenizer, opt):

    styles = ['informal', 'formal']

    test_src_file = 'data/{}/{}/{}'.format(opt.dataset, 'test', styles[opt.style])

    test_tgt_file = 'data/{}/{}/{}.ref0'.format(opt.dataset, 'test', styles[1 - opt.style])

    test_label_files = f'data/{opt.dataset}/test/{styles[1 - opt.style]}'

    test_dataset = T5Dataset(test_src_file, test_tgt_file, tokenizer, opt.max_len)
    test_loader = DataLoader(test_dataset,
                             num_workers=2,
                             batch_size=opt.val_batch_size,
                             shuffle=False)

    print('[Info] {} instances from test set'.format(len(test_dataset)))

    print("Test starts...")

    model.eval()


    start = time.time()
    pred_list = []

    if not os.path.exists(f'./data/{opt.dataset}/outputs/{opt.model}/'):
        os.mkdir(f'./data/{opt.dataset}/outputs/{opt.model}/')
    with open('./data/{}/outputs/{}/{}_{}_{}.{}_best_test.txt'.format(opt.dataset, opt.model,
                                                                      opt.model, opt.dataset, opt.order, opt.style),
              'w') as fout:
        for idx, data in enumerate(test_loader):
            if idx % 10 == 0:
                print('[Info] processing {} batches | seconds {:.4f}'.format(
                    idx, time.time() - start))
                start = time.time()

            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(ids,
                                           attention_mask=mask,
                                           num_beams=5,
                                           max_length=30)

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                     generated_ids]

            pred_list.extend(preds)
        for text in pred_list:
            fout.write(text.strip() + '\n')

    model.train()

    pred_file = './data/{}/outputs/{}/{}_{}_{}.{}_best_test.txt'.format(opt.dataset, opt.model,
                                                                        opt.model, opt.dataset, opt.order, opt.style)

    bleu = evaluate_bleu(test_label_files, pred_file)
    print(bleu)

    test_tgt = []
    test_src = []
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            if opt.style == 0:
                test_tgt.append(cls_tokenizer.encode(line.strip())[:opt.max_len])
            else:
                test_src.append(cls_tokenizer.encode(line.strip())[:opt.max_len])
    cls_loader = SCIterator(test_src, test_tgt, opt, cls_tokenizer.pad_token_id)
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    total_num = 0.
    total_acc = 0.
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(cls_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = cls(x_batch)
            # print(F.softmax(logits, dim=-1))
            total_loss += cls_loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits, dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

    print('Test: {}'.format('acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))

    with open('./data/{}/outputs/{}/{}_{}_{}_bleu_test.txt'.format(opt.dataset, opt.model,
                                                                   opt.model, opt.dataset, opt.order, opt.style),
              'a') as fbl:
        fbl.write(
            'Test Bleu score for model {}: {:.4f};  Acc: {:.4f}\n'.format(opt.order, bleu, total_acc / total_num * 100))

    return bleu, total_acc / total_num * 100, total_loss / total_num

