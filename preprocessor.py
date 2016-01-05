import numpy as np
from sklearn import preprocessing
import re
import pickle

def PruneSentencesPreprocessor(X, y):
    _X = []
    ytrans = []
    for x, _y in zip(X, y):
        hin_found = False
        en_found = False
        for obv, label in zip(x, _y):
            if obv['LANG'] == 'hi':
                hin_found = True
            if obv['LANG'] == 'en':
                en_found = True
            if hin_found and en_found:
                _X.append(x)
                ytrans.append(_y)
                break
    return _X, ytrans

