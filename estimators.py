from sklearn.base import BaseEstimator
import pycrfsuite
import numpy as np
from collections import defaultdict, Counter
from operator import itemgetter
from itertools import count
from os.path import isfile
from wxconv import WXC as wxConvert
from indictrans import Transliterator as transliterator
#from converter_indic import wxConvert
#from irtrans import transliterator
import codecs
import os
import pickle
from customutils import GetColumn, SeperateColumn

class HPOSEstimator(BaseEstimator):
    '''
    Dummy estimator - returns the predicted ilmt pos tags after running
    normalization
    '''
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
    """
    def __init__(self, script='wx', raw='data/all_normalization_predictions_wx', rebuild=False):
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
                if obv['LANG'] == 'hi':
                    if obv['WORD'] in self.normdict:
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
                else:
                    pred = obv['WORD']

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

    def fit(self, X=None, y=None):
        return self
