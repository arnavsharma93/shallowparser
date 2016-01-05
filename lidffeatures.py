import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import re
import pickle
import itertools
import os
from itertools import chain
from nltk.util import ngrams

BNC_MAX_VALUE = 6187927

class AddLexTransformer(BaseEstimator):
    '''
    Adds the lex as a feature
    '''
    def transform(self, X, **transform_params):
        return [['LEX:%s' % obv['WORD'].lower() for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class AddPositionTransformer(BaseEstimator):
    '''
    Adds the word position as a feature
    '''
    def transform(self, X, **transform_params):
        return [['POSITION:%.2f' % obv['POSITION'] for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class PoSTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['EPOS:%s' % (obv['EPOS'] if not self.ignore else 'False')  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class PoSConfidenceTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['EPOSSCORE:%s' % (obv['EPOSSCORE'][2:4] if not self.ignore else 'False')  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class AffixesTransformer(BaseEstimator):
    '''
    Adds suffixes of length from [1, suffix_len] as features
    Adds prefixes of length from [1, prefix_len] as features
    forms:
    '''
    def __init__(self, suffix_len=5, prefix_len=5):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    def transform(self, X, Y=None, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                lex = obv['WORD']

                _obv = []

                for j in xrange(-self.suffix_len, 0):
                    suffix = lex[j:] if abs(j) <= len(lex) else ''
                    #if len(suffix) > 3:
                    if True:
                        f = 'word[%d:]=%s' % (j, suffix)
                        _obv.append(f)

                for j in xrange(1, self.prefix_len+1):
                    prefix = lex[:j] if abs(j) <= len(lex) else ''
                    #if len(prefix) > 3:
                    if True:
                        f = 'word[:%d]=%s' % (j, prefix)
                        _obv.append(f)

                _x.append(_obv)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self
class BNCCountsTransformer(BaseEstimator):
    '''
    BNC dictionary counts
    '''
    def __init__(self, ignore=False):
        pickle_file = "data/bncdict.p"
        if not os.path.exists(pickle_file):
            self.create_pickle()
        self.BNC_dict = pickle.load(open(pickle_file, "rb"))
        self.ignore = ignore


    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                word = obv['WORD'].lower()
                val = (self.BNC_dict[word]*1.0)/BNC_MAX_VALUE if (word in self.BNC_dict and not self.ignore)  else 0
                _x.append('BNC_COUNT:%s' % val)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

    def create_pickle(self, filename='data/bnc.txt'):
        BNC_dict = {}
        with open(filename) as f:
            for lines in f:
                try:
                    freq, word, pos, num_files_occurs_in = map(lambda x: x.strip(), lines.split())
                    if word in BNC_dict:
                        BNC_dict[word] += int(freq)
                    elif word not in BNC_dict:
                        BNC_dict[word] = int(freq)

                except:
                    pass
        pickle.dump(BNC_dict, open("data/bncdict.p", "wb"))

class LexNormCountsTransformer(BaseEstimator):
    '''
    LexNorm dictionary counts
    '''
    def __init__(self, ignore=False):
        pickle_file = "data/lexnormdict.p"
        if not os.path.exists(pickle_file):
            self.create_pickle()
        self.lexnorm_dict = pickle.load(open(pickle_file, "rb"))
        self.ignore = ignore


    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                word = obv['WORD'].lower()
                val = "True" if (word in self.lexnorm_dict and not self.ignore)  else "False"
                _x.append('LEXNORM_COUNT:%s' % val)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

    def create_pickle(self, filename='data/lexnorm.txt'):
        lexnorm_dict = {}
        with open(filename) as f:
            for lines in f:
                try:
                    orig, pos, corrected = map(lambda x: x.strip(), lines.split())
                    lexnorm_dict[orig] = corrected
                except:
                    # If there's any fuck up in the line, ignore it.
                    # Most likely it's just some metadata/ encoding issue.
                    pass
        pickle.dump(lexnorm_dict, open("data/lexnormdict.p", "wb"))

class LexNormBNCSpecialTransformer(BaseEstimator):
    """

    *Binary LEXNORM_COUNT OR BNC_COUNT*

    The idea is that LEXNORM_COUNT only exists for misspelled words.
    Hence for correct words, it's 0. We want it to have some value
    even for the correct words (as it does for misspelled words).
    This will be more discriminatory.
    """
    def __init__(self, ignore=False):
        bnc = 'data/bncdict.p'
        lex = 'data/lexnormdict.p'
        if not os.path.exists(lex):
            LexNormCountsTransformer().create_pickle()
        if not os.path.exists(bnc):
            BNCCountsTransformer().create_pickle()
        self.lexnorm = pickle.load(open(lex, "rb"))
        self.BNC_dict = pickle.load(open(bnc, "rb"))
        self.ignore = ignore


    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                word = obv['WORD'].lower()
                val = "True" if (word in self.lexnorm and word in self.BNC_dict and not self.ignore)  else "False"
                _x.append('IN_BNC_OR_LEXNORM:%s'%val)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

class HindiDictionaryCountsTransformer(BaseEstimator):
    """ Takes the sentence_dict and adds another feature to it -
    *HINDI_COUNT*
    """
    def __init__(self, ignore=False):
        pickle_file = "data/hindi_transliteration_pairs_dict.p"
        if not os.path.exists(pickle_file):
            self.create_pickle()
        self.hindi_dict = pickle.load(open(pickle_file, "rb"))
        self.ignore = ignore


    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                word = obv['WORD'].lower()
                val = "True" if (word in self.hindi_dict and not self.ignore)  else "False"
                _x.append('HINDI_COUNT:%s' % val)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

    def create_pickle(self, filename='data/hindi_transliteration_pairs_1.txt'):
        hindi_dict = {}
        with open(filename) as f:
            for lines in f:
                try:
                    english_word, hindi_word = map(lambda x: x.strip(), lines.split())
                    hindi_dict[english_word] = hindi_word
                except:
                    # If there's any fuck up in the line, ignore it.
                    # Most likely it's just some metadata/ encoding issue.
                    pass
        pickle.dump(hindi_dict, open("data/hindi_transliteration_pairs_dict.p", "wb"))

class WordLengthTransformer(BaseEstimator):

    def __init__(self, ignore=False):
        self.ignore = ignore


    def transform(self, X, **transform_params):
        return [['LENGTH:%d' % len(obv['WORD']) for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self


class CapitalizationTransformer(BaseEstimator):
    """Adds a capitalisation based feature for each word.
    If any of the letter in a word is capital, it's 1 else 0."""
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['CAPITALIZATION:%s' % (obv['WORD'] == obv['WORD'].lower()) for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class CharacterNgramTransformer(BaseEstimator):
    """Adds n-grams as features to the sentence_dict. The dimensionality
    is exploded by 27^n when you add an n gram - For example 27(26 + space character)
    for unigram, and 27^2 for bigram."""

    def __init__(self, indices=[1,2,3], ignore=False):
        self.indices = indices
        self.ignore = ignore


    def transform(self, X, **transform_params):
        _X = []

        for x in X:
            _x = []
            for obv in x:
                grams = []
                word = list(obv['WORD'].lower())
                for index in self.indices:
                    for item in ngrams(word, index):
                        grams.append(''.join(item))
                _x.append(grams)

            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

