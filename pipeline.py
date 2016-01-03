from codemixeddata import *
import posfeatures as pos
import lidffeatures as lidf
from customutils import *
from sklearn_crfsuite import scorers, metrics, CRF
import numpy as np
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import pickle

def WXEstimator(X, y):
    if not os.path.exists('./data/normdict.p'):
        create_norm_pickle()
    normdict = pickle.load(open('./data/normdict.p', 'rb'))
    ypred = []
    for _x, _y in zip(X, y):
        _ypred = []
        for obv, label in zip(_x, _y):
            if obv['WORD'] in normdict and obv['LANG'] == 'hi':
                _ypred.append(normdict[obv['WORD']][1])
            else:
                _ypred.append(label)
        ypred.append(_ypred)
    return ypred

def run_step(X, step_model, step_name):
    X, y = TakeOutColumn(X, step_name)
    if step_model:
        ypred = cross_val_predict(step_model, X, y, cv=10, n_jobs=-1)
    elif step_name == 'WX':
        ypred = WXEstimator(X, y)

    X = AddColumn(X, ypred, step_name)
    return X
