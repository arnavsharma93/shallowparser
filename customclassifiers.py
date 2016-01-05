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


class HPOSEstimator(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        _X = np.copy(X)
        _X, y =  SeperateColumn(_X, '_HPOS')
        return y

class NormEstimator(BaseEstimator):
    """
    Dummy model to convert GOLD NORM to output from HINDI Normalizer
    Takes parameters:
    a. script which decides the output of the script of HINDI Normalizer
    b. filename which has normalized predictions in wx format
    structure of filename - roman predictedwx actualwx
    #FIX ME: rerun in case of new data, how to pass params to NormEstimator from run_step
    """
    def __init__(self, script='wx', raw='data/final_normalization_predictions_wx', rebuild=False):
        self.script = script
        self.raw = raw
        self.rebuild = rebuild
        self.pickled = './data/normdict.p'
        # converts wx2utf
        self.con = wxConvert(order='utf2wx', lang='hin')
        self.revcon = wxConvert(order='wx2utf', lang='hin')
        self.trn = transliterator(source='eng', target='hin')
        #rebuild dict
        if not os.path.exists(self.pickled) or self.rebuild:
            self.create_norm_pickle()
        self.normdict = pickle.load(open(self.pickled, 'rb'))

    def predict(self, X):
        ypred = []
        for _x in X:
            _ypred = []
            for obv in _x:
                if obv['WORD'] in self.normdict and obv['LANG'] == 'hi':
                    pred = self.normdict[obv['WORD']]
                else:
                    pred = self.trn.transform(obv['WORD'])
                    #if word not present transliterate it to utf
                    if not pred:
                        pred = obv['WORD'].lower()
                        pred = self.revcon.convert(pred)

                #convert it nonetheless
                if self.script == 'wx':
                    pred = self.con.convert(pred)
                _ypred.append(pred)
            ypred.append(_ypred)
        return ypred

    def create_norm_pickle(self):
        '''
        Create a pickle file for the normalized dictionary.
        Dictionary file is in wx
        '''
        normdict = {}
        with open(self.raw) as f:
            for lines in f:
                try:
                    roman, pred, actual = map(lambda x: x.strip(), lines.split())
                    if self.script == 'utf':
                        pred = self.con.recon(pred)
                    normdict[roman] = pred

                except:
                    pass
        pickle.dump(normdict, open(self.pickled, "wb"))

    def fit(self, X, y):
        return self
