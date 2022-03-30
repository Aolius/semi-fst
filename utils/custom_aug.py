# -*- coding: utf-8 -*-

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Method
from .abbreviations import en_abbreviations
import os
import numpy as np
import random
import nlpaug.model.word_dict as nmwd

from nlpaug.util import Action, Doc, LibraryUtil
# from aug_fix.word_augmentor import WordAugmenter
# import aug_fix.spelling as spelling


word2abbr = {}
for key, val in en_abbreviations.items():
    if word2abbr.get(val, 0) == 0:
        word2abbr[val] = key



class CapitalizeAug(WordAugmenter):
    '''
    Word Capitalization augmentation
    '''
    def __init__(self, name='capitalize', aug_min=1, aug_max=10,
                 aug_p=0.3, stopwords=None, tokenizer=None, reverse_tokenizer=None,
                 device='cpu', verbose=0, stopwords_regex=None):
        super(CapitalizeAug, self).__init__(
            action='substitute', name=name, aug_min=aug_min, aug_max=aug_max,
            aug_p=aug_p, stopwords=stopwords, tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
            device=device, verbose=0, stopwords_regex=stopwords_regex)

    def substitute(self, data):
        tokens = self.tokenizer(data)
        results = tokens.copy()

        aug_idexes = self._get_random_aug_idxes(tokens)
        if aug_idexes is None:
            return data
        aug_idexes.sort(reverse=True)

        for aug_idx in aug_idexes:
            results[aug_idx] = results[aug_idx].upper()

        return self.reverse_tokenizer(results)


class AbbrAug(WordAugmenter):
    '''
    Word replacing with abbreviations
    '''
    def __init__(self, name='abbr', aug_min=1, aug_max=10,
                 aug_p=0.3, stopwords=None, tokenizer=None, reverse_tokenizer=None,
                 device='cpu', verbose=0, stopwords_regex=None):
        super(AbbrAug, self).__init__(
            action='substitute', name=name, aug_min=aug_min, aug_max=aug_max,
            aug_p=aug_p, stopwords=stopwords, tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
            device=device, verbose=0, stopwords_regex=stopwords_regex)

    def substitute(self, data):

        tokens = self.tokenizer(data)
        results = []
        for t in tokens:
            if word2abbr.get(t, 0) != 0:
                results.append(word2abbr[t])
            else:
                results.append(t)

        return self.reverse_tokenizer(results)


# class FirstAug(WordAugmenter):
#     '''
#     change the first letter of the sentence to upper case.
#     '''
#     def __init__(self, name='first', aug_min=1, aug_max=10,
#                  aug_p=0.2, stopwords=None, tokenizer=None, reverse_tokenizer=None,
#                  device='cpu', verbose=0, stopwords_regex=None):
#         super(FirstAug, self).__init__(
#             action='substitute', name=name, aug_min=aug_min, aug_max=aug_max,
#             aug_p=aug_p, stopwords=stopwords, tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
#             device=device, verbose=0, stopwords_regex=stopwords_regex)
#
#     def substitute(self, data):
#
#         tokens = self.tokenizer(data)
#         if tokens[0].istitle():
#             _ = random.random()
#             if _ < 0.7:
#                 tokens[0] = tokens[0][0].lower() + tokens[0][1:]
#
#
#         return self.reverse_tokenizer(tokens)


# class RepeatCharAug(CharAugmenter):
#     def __init__(self, name='repeat_char', repeat_times=3, min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
#                  aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
#                  stopwords=None, verbose=0, stopwords_regex=None):
#         super().__init__(
#             name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min,
#                 aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
#                 aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
#                 reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
#                 verbose=verbose, stopwords_regex=stopwords_regex)
#
#
#         self.repeat_times = repeat_times
#
#     def substitute(self, data):
#         results = []
#         # Tokenize a text (e.g. The quick brown fox jumps over the lazy dog) to tokens (e.g. ['The', 'quick', ...])
#         tokens = self.tokenizer(data)
#         # Get target tokens
#         aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
#
#         for token_i, token in enumerate(tokens):
#             # Do not augment if it is not the target
#             if token_i not in aug_word_idxes:
#                 results.append(token)
#                 continue
#
#
#             chars = self.token2char(token)
#
#             chars[-1] = chars[-1] * self.repeat_times
#
#             result = "".join(chars)
#
#             results.append(result)
#
#         return self.reverse_tokenizer(results)


"""
    Augmenter that apply spelling error simulation to textual input.
"""
#
#
# SPELLING_ERROR_MODEL = {}
#
#
# def init_spelling_error_model(dict_path, include_reverse, force_reload=False):
#     # Load model once at runtime
#     global SPELLING_ERROR_MODEL
#     if SPELLING_ERROR_MODEL and not force_reload:
#         return SPELLING_ERROR_MODEL
#
#     spelling_error_model = spelling.Spelling(dict_path, include_reverse)
#
#     SPELLING_ERROR_MODEL = spelling_error_model
#
#     return SPELLING_ERROR_MODEL
#
#
# class SpellingAug(WordAugmenter):
#     # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf
#     """
#     Augmenter that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.
#
#     :param str dict_path: Path of misspelling dictionary
#     :param float aug_p: Percentage of word will be augmented.
#     :param int aug_min: Minimum number of word will be augmented.
#     :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
#         calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
#         aug_p. Otherwise, using aug_max.
#     :param list stopwords: List of words which will be skipped from augment operation.
#     :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
#     :param func tokenizer: Customize tokenization process
#     :param func reverse_tokenizer: Customize reverse of tokenization process
#     :param str name: Name of this augmenter
#
#     >>> import nlpaug.augmenter.word as naw
#     >>> aug = naw.SpellingAug(dict_path='./spelling_en.txt')
#     """
#
#     def __init__(self, dict_path=None, name='Spelling_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
#                  tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
#                  verbose=0,random_seed=42):
#         super().__init__(
#             action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
#             tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, device='cpu', verbose=verbose,
#             stopwords_regex=stopwords_regex, include_detail=False,random_seed=random_seed)
#
#         self.dict_path = dict_path if dict_path else os.path.join(LibraryUtil.get_res_dir(), 'word', 'spelling',
#                                                                   'spelling_en.txt')
#         self.include_reverse = include_reverse
#         self.model = self.get_model(force_reload=False)
#         self.random_seed = random_seed
#
#     def skip_aug(self, token_idxes, tokens):
#         results = []
#         for token_idx in token_idxes:
#             # Some words do not exit. It will be excluded in lucky draw.
#             token = tokens[token_idx]
#             if token in self.model.dict and len(self.model.dict[token]) > 0:
#                 results.append(token_idx)
#
#         return results
#
#     def sample_x(self, x, num=None):
#         # random.seed(42)
#         # np.random.seed(42)
#         if isinstance(x, list):
#             return random.sample(x, num)
#         elif isinstance(x, int):
#             return np.random.randint(1, x-1)
#
#     def substitute(self, data):
#         # np.random.seed(self.random_seed)
#         # random.seed(self.random_seed)
#         if not data or not data.strip():
#             return data
#
#         change_seq = 0
#         doc = Doc(data, self.tokenizer(data))
#
#         aug_idxes = self._get_aug_idxes(doc.get_original_tokens())
#
#
#         if aug_idxes is None or len(aug_idxes) == 0:
#             if self.include_detail:
#                 return data, []
#             return data
#
#         for aug_idx, original_token in enumerate(doc.get_original_tokens()):
#             # Skip if no augment for word
#             if aug_idx not in aug_idxes:
#                 continue
#
#             candidate_words = self.model.predict(original_token)
#             # print(candidate_words)
#
#             substitute_token = ''
#             if candidate_words:
#                 substitute_token = self.sample_x(candidate_words, 1)[0]
#                 # print(substitute_token)
#             else:
#                 # Unexpected scenario. Adding original token
#                 substitute_token = original_token
#
#             if aug_idx == 0:
#                 substitute_token = self.align_capitalization(original_token, substitute_token)
#
#             change_seq += 1
#             doc.add_change_log(aug_idx, new_token=substitute_token, action=Action.SUBSTITUTE,
#                                change_seq=self.parent_change_seq + change_seq)
#
#         if self.include_detail:
#             return self.reverse_tokenizer(doc.get_augmented_tokens()), doc.get_change_logs()
#         else:
#             return self.reverse_tokenizer(doc.get_augmented_tokens())
#
#     def get_model(self, force_reload):
#         return init_spelling_error_model(self.dict_path, self.include_reverse, force_reload)
#
#
#
#
#
