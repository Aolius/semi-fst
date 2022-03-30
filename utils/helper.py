# -*- coding: utf-8 -*-
# source code: https://github.com/laihuiyuan/pre-trained-formality-transfer/blob/main/utils/helper.py
import torch

from torch import cuda
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from utils.dataset import collate_fn
# from transformers.modeling_bart import  make_padding_mask, shift_tokens_right
import os, io, subprocess, re

device = 'cuda' if cuda.is_available() else 'cpu'


def optimize(opt, loss, retain_graph=False):
    opt.zero_grad()

    loss.backward(retain_graph=retain_graph)
    opt.step()




def sample_3d(probs, temperature=1):
    '''probs.shape = (batch, seq_len, dim)'''
    sample_idx = torch.zeros(probs.size(0), probs.size(1)).to(device)
    sample_probs = torch.zeros(probs.size(0), probs.size(1)).to(device)
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    for i, s in enumerate(temp):
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1)
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_probs[i] = temp_probs.squeeze(1)

    return sample_probs, sample_idx.long()


def evaluate(model, valid_loader, loss_fn,
             classifier, tokenizer, step, style):
    '''Evaluation function for BART'''
    model.eval()
    total_num = 0.
    total_acc = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)

            mask = (src == tokenizer.pad_token_id).float()
            mask = 1 - mask.long() if mask is not None else None
            # logits = model(src, attention_mask=mask,
            #                decoder_input_ids=tgt)[0]
            # decoder_input_ids = shift_tokens_right(tgt, tokenizer.pad_token_id)
            # labels = tgt.clone()
            # labels[labels[:, :] == tokenizer.pad_token_id] = -100
            logits = model(src, attention_mask=mask,
                           # decoder_input_ids=decoder_input_ids,
                           labels=tgt,
                           return_dict=True).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            probs, idxs = torch.max(logits, dim=-1)

            tgt = []
            for i in idxs:
                e = torch.arange(len(i))[i.eq(tokenizer.eos_token_id)]
                e = e[0] if 0<len(e) and e[0]<30 else 30
                tgt.append(i[:e].cpu().tolist())
            tgt = collate_fn(tgt).to(device)
            _, y_hat = torch.max(classifier(tgt),dim=-1)

            if style == 0:
                y_hat = [1 if p==1 else 0 for p in y_hat]
            else:
                y_hat = [1 if p==0 else 0 for p in y_hat]
            total_acc += sum(y_hat)
            total_num += len(tgt)
            total_loss += loss.mean()
    model.train()
    print('[Info] valid {:05d} | loss {:.4f} | acc_sc {:.4f}'.format(
        step, total_loss / len(valid_loader), total_acc / total_num))

    return total_loss / len(valid_loader), total_acc / total_num


def evaluate_sc(model, valid_loader, loss_fn, epoch):
    '''Evaluation function for style classifier'''
    model.eval()
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
                epoch, 'acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))

    return total_acc / total_num, total_loss / total_num

def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
        with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
            arg = [bleu_script] + labels_file
            bleu_out = subprocess.check_output(
                arg,
                stdin=predictions_file,
                stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            return float(bleu_score)

    except subprocess.CalledProcessError as error:

        return None