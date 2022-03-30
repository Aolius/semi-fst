# -*- coding: utf-8 -*-

import os
import time
import argparse
import random, re, math



import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer

import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from pyskiplist import SkipList
import fitlog
import kenlm
import statistics


from classifier.textcnn_t5 import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import optimize
from utils.T5_dataset import T5Dataset, T5UnsupDataset, T5AugDataset
from utils.dataset import SCIterator
from utils.nltk_bleu import evaluate_bleu
from T5_test import test


device = 'cuda' if cuda.is_available() else 'cpu'

# parameters for textCNN
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch_cls(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch for CNN classifier '''

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq

def score_generated_sentences(generated_text_file_path, model):
    #get perplexity scores for evaluation
    log_probs = list()
    perplexity_scores = list()

    with open(generated_text_file_path) as generated_text_file:
        for sentence in generated_text_file:
            cleaned_sentence = clean_text(sentence)
            log_probs.append(model.score(cleaned_sentence))
            perplexity_scores.append(model.perplexity(cleaned_sentence))

    return statistics.mean(log_probs), statistics.mean(perplexity_scores)


def clean_text(string):
    string = string.replace(".", "")
    string = string.replace(".", "")
    string = string.replace("\n", " ")
    string = string.replace(" 's", " is")
    string = string.replace("'m", " am")
    string = string.replace("'ve", " have")
    string = string.replace("n't", " not")
    string = string.replace("'re", " are")
    string = string.replace("'d", " would")
    string = string.replace("'ll", " will")
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = re.sub(r'\d+', "number", string)
    string = ''.join(x for x in string if x.isalnum() or x == " ")
    string = re.sub(r'\s{2,}', " ", string)
    string = string.strip().lower()

    return string


def score_sentence(sentences, model):
    # log_probs = list()
    perplexity_scores = list()

    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        # log_probs.append(model.score(cleaned_sentence))
        perplexity_scores.append(model.perplexity(cleaned_sentence))

    return perplexity_scores

def sentence_bleu_score(sents, refs, ngrams=3):
    sents = [nltk.word_tokenize(sent) for sent in sents]
    refs = [nltk.word_tokenize(ref) for ref in refs]
    weight=[1.0 / ngrams] * ngrams
    scores = []
    for sent, ref in zip(sents, refs):
        scores.append(sentence_bleu([ref], sent, weights=weight))
    return scores
    




def main():
    parser = argparse.ArgumentParser('Fine-Tuned T5 for style transfer')
    parser.add_argument('-order', default=0, type=str, help='the order of traing')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-lr', default=1e-5, type=float, help='the learning rate')
    parser.add_argument('-ratio', default=1., type=float, help='proportion of data')
    parser.add_argument('-model', default='t5', type=str, help='the name of model')
    parser.add_argument('-model_name', default='t5-large', type=str, help='the name of model')
    parser.add_argument('-dataset', default='em', type=str, help='the name of dataset')
    parser.add_argument('-steps', default=100000, type=int, help='force stop at x steps')
    parser.add_argument('-batch_size', default=16, type=int, help='the size in a batch')
    parser.add_argument('-val_batch_size', default=16, type=int, help='the size in a batch')
    parser.add_argument('-max_len', default=50, type=int, help='maximum tokens a batch')
    parser.add_argument('-dropout', default=0.5, type=float, help='Keep prob in dropout')
    parser.add_argument('-patience', default=10, type=int, help='early stopping fine-tune')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-eval_step', default=1000, type=int, help='evaluate every x step')
    parser.add_argument('-unsup', action='store_true', help='use unsupervised loss')
    parser.add_argument('-unsup_batch_size', default=32, type=int, help='batch size for unlabeled data')
    parser.add_argument('-weight', default=1.0, type=float, help='balance weight of unsup loss')
    parser.add_argument('-pre_step', default=100, type=int, help='pretrain steps')
    parser.add_argument('-aug_type', default='real-time',type=str, help="whether to augment sentences while training")
    parser.add_argument('-aug_choice', default='spell', type=str)
    parser.add_argument('-aug_p', default=0.1, type=float, help='augmentation probability')
    parser.add_argument('-filter', default='bleu', type=str, help='metric used for data filtering')
    parser.add_argument('-phi', default=0.6, type=float, help='threshold for formality evaluation (cls filter)')
    parser.add_argument('-n_step', default=100, type=int, help='number of steps for computing initial score list')
    parser.add_argument('-sigma', default=0.3, type=float, help='dynamic threshold for lm/bleu filtering')
    parser.add_argument('-ngrams', default=4, type=int)
    parser.add_argument('-log_dir', default='./logs', type=str, help='directory of logs')


    opt = parser.parse_args()
    print('[Info]', opt)
    with open('./data/{}/outputs/{}/{}_{}_{}.{}_bleu.txt'.format(opt.dataset, opt.model,
                                                                 opt.model, opt.dataset, opt.order, opt.style),
              'a') as fbl:
        fbl.write(str(opt) + '\n')

    set_seed(opt.seed)
    # fitlog.debug() disenables fitlog, comment it if you want to use fitlog
    fitlog.debug()
    fitlog.set_log_dir(opt.log_dir)
    fitlog.add_hyper(opt)
    fitlog.add_hyper_in_file(__file__)

    tokenizer = T5Tokenizer.from_pretrained(opt.model_name)
    cls_tokenizer = tokenizer

    model = T5ForConditionalGeneration.from_pretrained(opt.model_name)


    model.to(device).train()


    # CNN classifier for evaluation
    cls = TextCNN(300, len(cls_tokenizer), filter_sizes,
                  num_filters, None, dropout=opt.dropout)
    cls.to(device).eval()
    cls.load_state_dict(torch.load('./checkpoints/t5_textcnn_{}.chkpt'.format(
        opt.dataset)))


    styles = ['informal', 'formal']
    if opt.style == 0:
        unsup_file = f"data/unlabeled/{opt.dataset.upper()}_200k_inf.txt"
    else:
        raise ValueError("Invalid style.")

    train_src_file = 'data/{}/{}/{}'.format(opt.dataset, 'train', styles[opt.style])
    train_tgt_file = 'data/{}/{}/{}'.format(opt.dataset, 'train', styles[1 - opt.style])
    train_dataset = T5Dataset(train_src_file, train_tgt_file, tokenizer, opt.max_len)
    train_loader = DataLoader(train_dataset,
                              num_workers=2,
                              batch_size=opt.batch_size,
                              shuffle=True)

    val_src_file = 'data/{}/{}/{}'.format(opt.dataset, 'tune', styles[opt.style])
    val_tgt_file = 'data/{}/{}/{}.ref0'.format(opt.dataset, 'tune', styles[1 - opt.style])
    val_label_files = f'data/{opt.dataset}/tune/{styles[1 - opt.style]}'
    val_dataset = T5Dataset(val_src_file, val_tgt_file, tokenizer, opt.max_len)
    val_loader = DataLoader(val_dataset,
                            num_workers=2,
                            batch_size=opt.val_batch_size,
                            shuffle=False)

    # test_src_file = 'data/{}/{}/{}'.format(opt.dataset, 'test', styles[opt.style])
    # test_tgt_file = 'data/{}/{}/{}.ref0'.format(opt.dataset, 'test', styles[1 - opt.style])
    # test_label_files = f'data/{opt.dataset}/test/{styles[1 - opt.style]}'


    print('[Info] {} instances from train set'.format(len(train_dataset)))
    print('[Info] {} instances from validation set'.format(len(val_dataset)))

    # pretrained lm model for lm filter
    language_model_path = f'./checkpoints/{opt.dataset}_{styles[1-opt.style]}.arpa'
    lm_model = kenlm.Model(language_model_path)

    if opt.unsup:
        #'real-time' means augmenting the texts on-the-fly
        if opt.aug_type == 'real-time':
            aug = opt.aug_choice
            unlabeled_dataset = T5AugDataset(
                src_file=unsup_file,
                augmentor=aug,
                max_len=opt.max_len,
                tokenizer=tokenizer,
                aug_p=opt.aug_p,
                dataset=opt.dataset
            )
        else:
            # Otherwise, augment all the texts beforehand
            unlabeled_dataset = T5UnsupDataset(
                src_file=f"data/unlabeled/{opt.dataset.upper()}_200k_inf.txt",
                aug_file=f"data/unlabeled/{opt.dataset.upper()}_200k_inf_{opt.aug_choice}.txt",
                max_len=opt.max_len,
                tokenizer=tokenizer)

        unsup_loader = DataLoader(unlabeled_dataset,
                              num_workers=10,
                              batch_size=opt.unsup_batch_size,
                              shuffle=True)
        print('[Info] {} instances from unlabeled set'.format(len(unlabeled_dataset)))


    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, 10000)

    tab = 0
    eval_loss = 1e8
    total_loss_ce = []
    total_loss_unsup = []
    best_bleu = 0.0
    best_acc = 0.0

    perp_list_all = SkipList()
    bleu_list_all = SkipList()

    #count filtered samples
    num_all = 0
    num_chosen = 0

    start = time.time()
    train_iter = iter(train_loader)
    if opt.unsup:
        unsup_iter = iter(unsup_loader)

    for step in range(1, opt.steps):
        try:
            data = next(train_iter)
        except:
            train_iter = iter(train_loader)
            data = next(train_iter)

        if opt.unsup and step > opt.pre_step:
            try:
                unsup_batch = next(unsup_iter)
            except:
                unsup_iter = iter(unsup_loader)
                unsup_batch = next(unsup_iter)

        # supervised loss
        lm_labels = data['target_ids'].to(device, dtype=torch.long)
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
        loss_ce = outputs[0]
        total_loss_ce.append(loss_ce.item())

        # unsupervised loss
        if opt.unsup and step > opt.pre_step:
            unsup_ids = unsup_batch['source_ids'].to(device, dtype=torch.long)
            unsup_mask = unsup_batch['source_mask'].to(device, dtype=torch.long)
            aug_ids = unsup_batch['augment_ids'].to(device, dtype=torch.long)
            aug_mask = unsup_batch['augment_mask'].to(device, dtype=torch.long)
            model.eval()
            pseudo_labels = model.generate(unsup_ids,
                                           attention_mask=unsup_mask,
                                           num_beams=5,
                                           max_length=30)
            model.train()

            if opt.filter == 'lm':
                pseudo_targets = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                                  g in
                                  pseudo_labels]
                perplexities = score_sentence(pseudo_targets, lm_model)
                if num_all < len(unlabeled_dataset):
                    for score in perplexities:
                        perp_list_all.insert(score, None)
                    if step > opt.n_step + opt.pre_step:
                        idx = math.floor((1 - opt.sigma) * len(perp_list_all))
                        perp_threshold = perp_list_all[idx][0]
                    else:
                        perp_threshold = 10000
                else:
                    perp_threshold = perp_threshold


                y_mask = (torch.Tensor(perplexities) < perp_threshold).to(device, dtype=torch.float)
                num_all += opt.unsup_batch_size
                num_chosen += sum(y_mask)
                fitlog.add_metric(num_chosen, name="num_chosen", step=step)
                fitlog.add_metric(num_all, name="num_all", step=step)
                fitlog.add_metric(num_chosen / num_all, name="filter_ratio", step=step)
                fitlog.add_metric(perp_threshold, name="perp_threshold", step=step)

                if num_all >= len(unlabeled_dataset):
                    num_all = 0
                    num_chosen = 0
                    perp_list_all = SkipList()

                pseudo_labels[pseudo_labels[:, :] == tokenizer.pad_token_id] = -100
                unsup_output = model(aug_ids, attention_mask=aug_mask,
                                     labels=pseudo_labels)

                unsup_logits = unsup_output[1]
                pseudo_labels[pseudo_labels[:, :] == -100] = tokenizer.pad_token_id
                unsup_loss_ce = loss_fn(unsup_logits.view(-1, unsup_logits.size(-1)), pseudo_labels.view(-1))
                y_mask = y_mask.unsqueeze(1).repeat(1, pseudo_labels.size(-1)).view(-1)

                unsup_loss_ce = unsup_loss_ce * y_mask
                unsup_loss_ce = unsup_loss_ce.sum()
                if unsup_loss_ce > 0:
                    unsup_loss_ce /= y_mask.sum()
                total_loss_unsup.append(unsup_loss_ce.item())

            if opt.filter == 'bleu':
                pseudo_targets = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                                  g in
                                  pseudo_labels]
                sources = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                                  g in
                                  unsup_ids]
                sent_bleus = sentence_bleu_score(pseudo_targets, sources, opt.ngrams)

                if num_all < len(unlabeled_dataset):
                    for score in sent_bleus:
                        bleu_list_all.insert(score, None)
                    if step > opt.n_step + opt.pre_step:

                        idx = math.floor(opt.sigma * len(bleu_list_all))
                        bleu_threshold = bleu_list_all[idx][0]

                    else:
                        bleu_threshold = -1
                else:
                    #meaningless, just for notes
                    bleu_threshold = bleu_threshold

                y_mask = (torch.Tensor(sent_bleus) > bleu_threshold).to(device, dtype=torch.float)
                num_all += opt.unsup_batch_size
                num_chosen += sum(y_mask)
                fitlog.add_metric(num_chosen, name="num_chosen", step=step)
                fitlog.add_metric(num_all, name="num_all", step=step)
                fitlog.add_metric(num_chosen / num_all, name="filter_ratio", step=step)
                fitlog.add_metric(bleu_threshold, name="bleu_threshold",step=step)

                if num_all >= len(unlabeled_dataset):
                    num_all = 0
                    num_chosen = 0
                    bleu_list_all = SkipList()

                pseudo_labels[pseudo_labels[:, :] == tokenizer.pad_token_id] = -100
                unsup_output = model(aug_ids, attention_mask=aug_mask,
                                     labels=pseudo_labels)
                # unsup_loss_ce = unsup_output[0]
                unsup_logits = unsup_output[1]
                pseudo_labels[pseudo_labels[:, :] == -100] = tokenizer.pad_token_id
                unsup_loss_ce = loss_fn(unsup_logits.view(-1, unsup_logits.size(-1)), pseudo_labels.view(-1))
                y_mask = y_mask.unsqueeze(1).repeat(1, pseudo_labels.size(-1)).view(-1)

                unsup_loss_ce = unsup_loss_ce * y_mask
                unsup_loss_ce = unsup_loss_ce.sum()
                if unsup_loss_ce > 0:
                    unsup_loss_ce /= y_mask.sum()
                total_loss_unsup.append(unsup_loss_ce.item())

            if opt.filter == "none":
                pseudo_labels[pseudo_labels[:, :] == tokenizer.pad_token_id] = -100
                unsup_output = model(aug_ids, attention_mask=aug_mask,
                                     labels=pseudo_labels)
                unsup_loss_ce = unsup_output[0]
                total_loss_unsup.append(unsup_loss_ce.item())

            if opt.filter == "cls":
                pseudo_targets = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                                  g in
                                  pseudo_labels]
                pseudo_tgt = [cls_tokenizer.encode(line.strip())[:opt.max_len] for line in pseudo_targets]
                pseudo_tgt = prepare_batch_cls(pseudo_tgt, pad_token_id=cls_tokenizer.pad_token_id).to(device)
                logits_tgt = F.softmax(cls(pseudo_tgt), dim=-1)
                pseudo_source = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g
                                 in
                                 unsup_ids]
                pseudo_src = [cls_tokenizer.encode(line.strip())[:opt.max_len] for line in pseudo_source]
                pseudo_src = prepare_batch_cls(pseudo_src, pad_token_id=cls_tokenizer.pad_token_id).to(device)
                logits_src = F.softmax(cls(pseudo_src), dim=-1)
                y_mask = (logits_tgt[:, 1-opt.style] - logits_src[:, 1-opt.style] > opt.phi).float()

                num_all += opt.unsup_batch_size
                num_chosen += sum(y_mask)
                fitlog.add_metric(num_chosen, name="num_chosen", step=step)
                fitlog.add_metric(num_all, name="num_all", step=step)
                fitlog.add_metric(num_chosen / num_all, name="filter_ratio", step=step)

                if num_all >= len(unlabeled_dataset):
                    num_all = 0
                    num_chosen = 0

                pseudo_labels[pseudo_labels[:, :] == tokenizer.pad_token_id] = -100
                unsup_output = model(aug_ids, attention_mask=aug_mask,
                                     labels=pseudo_labels)
                unsup_logits = unsup_output[1]
                pseudo_labels[pseudo_labels[:, :] == -100] = tokenizer.pad_token_id
                unsup_loss_ce = loss_fn(unsup_logits.view(-1, unsup_logits.size(-1)), pseudo_labels.view(-1))
                y_mask = y_mask.unsqueeze(1).repeat(1, pseudo_labels.size(-1)).view(-1)

                unsup_loss_ce = unsup_loss_ce * y_mask
                unsup_loss_ce = unsup_loss_ce.sum()
                if unsup_loss_ce > 0:
                    unsup_loss_ce /= y_mask.sum()

                total_loss_unsup.append(unsup_loss_ce.item())

        if opt.unsup and step > opt.pre_step:
            optimize(optimizer, loss_ce + opt.weight * unsup_loss_ce)
        else:
            optimize(optimizer, loss_ce)

        if step % opt.log_step == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            if opt.unsup and step > opt.pre_step:
                print('[Info] steps {:05d} | loss_sup {:.4f} | '
                      'loss_unsup {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    step, np.mean(total_loss_ce), np.mean(total_loss_unsup)
                    , lr, time.time() - start))
                fitlog.add_loss(np.mean(total_loss_ce), name="Sup-loss", step=step)
                fitlog.add_loss(np.mean(total_loss_unsup), name="Unsup-loss", step=step)
            else:
                print('[Info] steps {:05d} | loss_ce {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    step, np.mean(total_loss_ce), lr, time.time() - start))
                fitlog.add_loss(np.mean(total_loss_ce), name="Sup-loss", step=step)

            total_loss_ce = []
            total_loss_unsup = []
            start = time.time()

        if ((len(train_loader) > opt.eval_step
             and step % opt.eval_step == 0)
                or (len(train_loader) < opt.eval_step
                    and step % len(train_loader) == 0)):

            print("validation starts...")
            # if eval_loss >= valid_loss:
            model.eval()

            start = time.time()
            pred_list = []

            if not os.path.exists(f'./data/{opt.dataset}/outputs/{opt.model}/'):
                os.mkdir(f'./data/{opt.dataset}/outputs/{opt.model}/')
            with open('./data/{}/outputs/{}/{}_{}_{}.{}_step{}.txt'.format(opt.dataset, opt.model,
                    opt.model, opt.dataset, opt.order, opt.style, step), 'w') as fout:
                for idx, data in enumerate(val_loader):
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

            pred_file = './data/{}/outputs/{}/{}_{}_{}.{}_step{}.txt'.format(opt.dataset, opt.model,
                    opt.model, opt.dataset, opt.order, opt.style, step)

            bleu = evaluate_bleu(val_label_files, pred_file)
            print(bleu)
            fitlog.add_metric(bleu, name="BLEU", step=step)
            _, perplexity = score_generated_sentences(pred_file, lm_model)
            fitlog.add_metric(perplexity, name="perplexity", step=step)

            if bleu > best_bleu:
                tab = 0
                best_bleu = bleu
                torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}.chkpt'.format(
                    opt.model, opt.dataset, opt.order, opt.style))
                print('[Info] The checkpoint file has been updated.')
                fitlog.add_best_metric({"dev":{"BLEU":best_bleu}})
                test_bleu, test_acc, test_loss = test(model, tokenizer, cls, cls_tokenizer, opt)
                test_hm = 2.0 / (1.0 / test_bleu + 100.0 / test_acc)
                fitlog.add_loss({"test":{"Loss":test_loss}}, step=step)

                test_file = './data/{}/outputs/{}/{}_{}_{}.{}_best_test.txt'.format(opt.dataset, opt.model,
                                                                        opt.model, opt.dataset, opt.order, opt.style)
                _, test_perplexity = score_generated_sentences(test_file, lm_model)
                fitlog.add_best_metric({"test": {"BLEU": test_bleu, "Acc": test_acc, "HM": test_hm, "perplexity": test_perplexity}})
            else:
                tab += 1
            if tab == opt.patience:
                #early stopping
                exit()

            # Evaluate style accuracy
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
            total_loss = 0.
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

            with open('./data/{}/outputs/{}/{}_{}_{}.{}_bleu.txt'.format(opt.dataset, opt.model,
                    opt.model, opt.dataset, opt.order, opt.style), 'a') as fbl:

                fbl.write('Bleu score at step {}: {:.4f};  Acc: {:.4f}\n'.format(step, bleu, total_acc / total_num * 100))
            acc = total_acc / total_num * 100
            fitlog.add_metric(acc, name="Acc", step=step)
            if bleu == best_bleu:
                best_acc = acc
                fitlog.add_best_metric({"dev": {"Acc": best_acc}})

    fitlog.finish()

if __name__ == "__main__":
    main()
