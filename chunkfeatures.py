import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import re
import pickle
import itertools
import os
from itertools import chain
from nltk.util import ngrams

class PoSTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['POS:%s' % (obv['POS'] if not self.ignore else 'False')  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class LexPoSTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def transform(self, X, **transform_params):
        return [['LEX_POS:%s_%s' % (obv['WORD'], obv['POS'])  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class POSContextTransformer(BaseEstimator):
    '''
    Adds the words in the context window [start, end] of current word
    as features
    '''
    def __init__(self, start=-2, end=2):
        self.start = start
        self.end = end

    def transform(self, X, Y=None, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                _obv = []
                for j in xrange(self.start, self.end+1):
                    if not j:
                        continue

                    window_word = 'False'
                    if j != 0 and (i+j) >= 0 and (i+j) < len(x):
                        window_word = x[i+j]['POS']
                    f = '%d:context_pos=%s' % (j, window_word.lower())
                    _obv.append(f)
                _x.append(_obv)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self
