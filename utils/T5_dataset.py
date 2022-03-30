# -*- coding: utf-8 -*-

import random, re
import numpy as np

import torch
import torch.utils.data
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from .custom_aug import CapitalizeAug, AbbrAug, RepeatCharAug
from tqdm import tqdm

def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)

def augment_choice(arg, aug_p, dataset='em'):
    if arg == "drop":
        return naw.random.RandomWordAug(action='delete', aug_p=aug_p)
    elif arg == 'swap':
        return naw.random.RandomWordAug(action='swap', aug_p=aug_p)
    elif arg == 'unk':
        return naw.random.RandomWordAug(action='substitute', aug_p=aug_p, target_words=['_'])
    elif arg == 'synonym':
        return naw.SynonymAug(aug_src='wordnet', aug_p=aug_p)
    elif arg == 'spell':
        # If using official Nlpaug, randomness will happen even with fixed random seeds.
        return naw.SpellingAug(aug_p=aug_p)
    elif arg == 'keyboard':
        return nac.KeyboardAug(aug_word_p=aug_p)
    elif arg == 'tfidf':
        return naw.TfIdfAug(model_path=f'./data/unlabeled/{dataset}tfidf/', tokenizer=_tokenizer, aug_p=aug_p)
    elif arg == 'capital':
        return CapitalizeAug(aug_p=aug_p)
    elif arg == 'abbr':
        return AbbrAug(aug_p=aug_p)
    elif arg == 'repeatchar':
        return RepeatCharAug(aug_word_p=aug_p)
    elif arg == 'none':
        return None
    else:
        raise ValueError("Unsupported augmentor!")




def data_augment_all(dataset, augmentor):
    src_file = 'data/unlabeled/{}'.format(dataset)
    aug_file = 'data/unlabeled/{}-{}-aug'.format(dataset, augmentor)
    if augmentor == 'synonym':
        aug = naw.SynonymAug(aug_src='wordnet')
    elif augmentor == 'insert':
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="insert")
    elif augmentor == 'substitute':
        aug = naw.ContextualWordEmbsAug(
        model_path = 'bert-base-uncased', action = "substitute")
    with open(src_file, 'r') as f1, open(aug_file, 'w') as f2:
        f1 = f1.readlines()
        for i, s in enumerate(tqdm(f1)):
            augmented = data_augment(s, aug)
            f2.write(augmented + "\n")


def data_augment(sent, augmentor):

    return augmentor.augment(sent)

def read_data(dataset, style, max_len, prefix,
              tokenizer, domain=0, ratio=1.0):

    if domain!=0:
        domain = tokenizer.encode(domain, add_special_tokens=False)[0]

    if style == 0:
        src_file = 'data/{}/{}/informal'.format(dataset, prefix)
        if prefix == 'tune':
            tgt_file = 'data/{}/{}/formal.ref0'.format(dataset, prefix)
        else:
            tgt_file = 'data/{}/{}/formal'.format(dataset, prefix)
    else:
        src_file = 'data/{}/{}/formal'.format(dataset, prefix)
        if prefix == 'tune':
            tgt_file = 'data/{}/{}/informal.ref0'.format(dataset, prefix)
        else:
            tgt_file = 'data/{}/{}/informal'.format(dataset, prefix)

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()
        index = [i for i in range(len(f1))]
        random.shuffle(index)
        index = index[:int(len(index) * ratio)]
        for i, (s, t) in enumerate(zip(f1, f2)):
            if i in index:
                s = s.strip().replace("\n","")
                t = t.strip().replace("\n","")
                s = tokenizer.encode(s)
                t = tokenizer.encode(t)
                s = s[:min(len(s) - 1, max_len)] + s[-1:]
                t = t[:min(len(t) - 1, max_len)] + t[-1:]
                s[0] = domain
                src_seq.append(s)
                # tgt_seq.append([tokenizer.bos_token_id]+t)
                tgt_seq.append(t)

    return src_seq, tgt_seq


def read_unlabel_data(dataset, max_len,
              tokenizer, ratio=1.0, augmentor='synonym'):

    src_file = 'data/unlabeled/{}'.format(dataset)
    aug_file = 'data/unlabeled/{}-aug'.format(dataset)


    src_seq = []
    aug_seq = []
    with open(src_file, 'r') as f1, open(aug_file, 'r') as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()

        index = [i for i in range(len(f1))]
        random.shuffle(index)
        index = index[:int(len(index) * ratio)]
        for i, (s, a) in enumerate(tqdm(zip(f1, f2))):
            if i in index:

                s = tokenizer.encode(s)
                a = tokenizer.encode(a)

                s = s[:min(len(s) - 1, max_len)] + s[-1:]
                a = a[:min(len(a) - 1, max_len)] + a[-1:]
                src_seq.append(s)
                aug_seq.append(a)


    return src_seq, aug_seq


def collate_fn(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst

def unpaired_collate_fn(insts):
    src_inst = collate_fn(insts)

    return src_inst


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(insts_0, insts_1, opt, pad_token_id=1, shuffle=True):
    '''Data iterator for style classifier'''

    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts, pad_token_id)
        labels = torch.LongTensor(labels)
        return (seq, labels)

    num = len(insts_0) + len(insts_1)
    loader = torch.utils.data.DataLoader(
        CNNDataset(
            insts=insts_0 + insts_1,
            label=[0 if i < len(insts_0)
                   else 1 for i in range(num)]),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def load_embedding(tokenizer, embed_dim, embed_path=None):
    '''Parse an embedding text file into an array.'''

    embedding = np.random.normal(scale=embed_dim ** -0.5,
                                 size=(len(tokenizer), embed_dim))
    if embed_path == None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding


class T5Dataset(torch.utils.data.Dataset):

    def __init__(self, src_file, tgt_file, tokenizer, max_len):

        with open(src_file, 'r') as f1, open(tgt_file,'r') as f2:
            self.src = f1.readlines()
            self.tgt = f2.readlines()
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        ctext = self.src[index].strip()
        ctext = ' '.join(ctext.split())

        text = self.tgt[index].strip()
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class T5UnsupDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, aug_file, tokenizer, max_len):

        with open(src_file, 'r') as f1, open(aug_file,'r') as f2:
            self.src = f1.readlines()
            self.aug = f2.readlines()
        self.max_len = max_len
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        ctext = self.src[index].strip()
        ctext = ' '.join(ctext.split())

        text = self.aug[index].strip()
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')
        augment = self.tokenizer.batch_encode_plus([text], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        augment_ids = augment['input_ids'].squeeze()
        augment_mask = augment['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'augment_ids': augment_ids.to(dtype=torch.long),
            'augment_mask': augment_mask.to(dtype=torch.long)
        }



class T5AugDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, augmentor, tokenizer, max_len, aug_p=0.1, dataset='em'):

        with open(src_file, 'r') as f1:
            self.src = f1.readlines()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.augmentor = augment_choice(augmentor, aug_p, dataset=dataset)
        if self.augmentor is None:
            self.aug = False
        else:
            self.aug = True
        if isinstance(self.augmentor, list):
            self.mix = True
        else:
            self.mix = False


    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        ctext = self.src[index].strip()
        ctext = ' '.join(ctext.split())

        if self.aug:
            if self.mix:
                augmentor = random.choice(self.augmentor)
            else:
                augmentor = self.augmentor

            text = augmentor.augment(ctext)
        else:
            text = ctext

        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')
        augment = self.tokenizer.batch_encode_plus([text], max_length= self.max_len, padding='max_length',truncation=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        augment_ids = augment['input_ids'].squeeze()
        augment_mask = augment['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'augment_ids': augment_ids.to(dtype=torch.long),
            'augment_mask': augment_mask.to(dtype=torch.long)
        }

class BartDataset(torch.utils.data.Dataset):
    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def BARTIterator(train_src, train_tgt,
                 valid_src, valid_tgt, opt):
    '''Data iterator for fine-tuning BART'''

    train_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=train_src,
            tgt_inst=train_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=valid_src,
            tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


class UnsupDataset(torch.utils.data.Dataset):
    def __init__(self, src_file, aug_file, max_len, tokenizer):
        with open(src_file, 'r') as f1, open(aug_file,'r') as f2:
            self.f1 = f1.readlines()
            self.f2 = f2.readlines()
        self.max_len = max_len
        self.tokenizer = tokenizer




    def __len__(self):
        return len(self.f1)

    def __getitem__(self, idx):
        s = self.f1[idx]
        aug = self.f2[idx]
        s = self.tokenizer.encode(s)
        aug = self.tokenizer.encode(aug)
        s = s[:min(len(s) - 1, self.max_len)] + s[-1:]
        aug = aug[:min(len(aug) - 1, self.max_len)] + aug[-1:]
        return s, aug









def UnsupIterator(dataset, tokenizer, opt):
    '''Data iterator for fine-tuning BART'''

    train_loader = torch.utils.data.DataLoader(
        UnsupDataset(
            src_file=f"data/unlabeled/{dataset}",
            aug_file=f"data/unlabeled/{dataset}-aug",
            max_len=opt.max_len,
            tokenizer=tokenizer),
        num_workers=4,
        batch_size=opt.unsup_batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)


    return train_loader

