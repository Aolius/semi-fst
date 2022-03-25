import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nlpaug.model.word_stats as nmw
import re

def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)

# Load sample data
with open('./data/unlabeled/FR_200k_inf.txt', 'r') as fi:
    data = fi.readlines()
    data = [sent.rstrip() for sent in data]
    train_x_tokens = [_tokenizer(x) for x in data]

# Train TF-IDF model
tfidf_model = nmw.TfIdf()
tfidf_model.train(train_x_tokens)
tfidf_model.save('./data/unlabeled/frtfidf/')


augmentor = naw.TfIdfAug(model_path='./data/unlabeled/frtfidf', tokenizer=_tokenizer)

with open('./data/unlabeled/FR_200k_inf.txt', 'r') as fi,\
    open('./data/unlabeled/FR_200k_inf_tfidf.txt', 'w') as fo:
    data = fi.readlines()
    data = [sent.rstrip() for sent in data]
    for _, sent in enumerate(tqdm(data)):
        aug = augmentor.augment(sent)
        fo.write(aug + "\n")


