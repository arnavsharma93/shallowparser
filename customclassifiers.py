from sklearn.base import BaseEstimator
import pycrfsuite
import numpy as np
from collections import defaultdict, Counter
from operator import itemgetter
from itertools import count
from os.path import isfile
from converter_indic import wxConvert
from irtrans import transliterator
import codecs
import os
import pickle
from customutils import GetColumn, SeperateColumn

class GenericCRF(BaseEstimator):
    _serial = count(1)

    def __init__(self, lang, verbose=False):
        filename = './.crf_%s_%d' % (lang, self._serial.next())

        while os.path.isfile(filename):
            filename = './.crf_%s_%d' % (lang, self._serial.next())

        self.model_file = filename
        self.verbose = verbose

    def fit(self, X, Y):
        trainer = pycrfsuite.Trainer(verbose=self.verbose)
        for x, y in zip(X, Y):
            trainer.append(x, y)
        trainer.train(self.model_file)

    def predict(self, X):
        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_file)
        Y_pred = []
        for x in X:
            Y_pred.append(tagger.tag(x))
        return np.array(Y_pred)

    def __del__(self):
        if os.path.isfile(self.model_file):
            os.unlink(self.model_file)

class MostFrequentTag(BaseEstimator):
    def fit(self, X, Y):
        self._store = defaultdict(lambda: defaultdict(int))
        if X.shape != Y.shape:
            raise ValueError('Data and Target do not have the same shape')
        for sent, tags in zip(X, Y):
            for obv, tag in zip(sent, tags):
                self._store[obv[0]][tag] += 1
        for lex in self._store:
            self._store[lex] = max(self._store[lex].iteritems(), key=itemgetter(1))[0]

        c = Counter()
        for lex in self._store:
            c[self._store[lex]] += 1

        self._oov_label = c.most_common(1)[0][0]
        return self

    def predict(self, X):
        y_pred = []
        for sent in X:
            ctags = []
            for obv in sent:
                if obv[0] in self._store:
                    ctags.append(self._store[obv[0]])
                else:
                    ctags.append(self._oov_label)
            y_pred.append(ctags)
        return y_pred


