from sklearn.datasets.base import Bunch
from copy import deepcopy
from collections import namedtuple
import numpy as np
import pandas as pd

def GetDataFeel(filename="./data/dataset.txt"):
    sentence_dict = {
        "WORD": [],
        "LANG": [],
        "POS": [],
        "NORM": [],
        "CHUNK": []
    }
    with open(filename) as f:
        sentence_number = 0
        for line in f:
            if not line.strip():
                word, lang, norm, pos, chunk = None, None, None, None, None
            else:
                word, lang, norm, pos, chunk = map(lambda x: x.strip(), line.split('\t'))
            sentence_dict["WORD"].append(word)
            sentence_dict["LANG"].append(lang)
            sentence_dict["POS"].append(pos)
            sentence_dict["NORM"].append(norm)
            sentence_dict["CHUNK"].append(chunk)
    return pd.DataFrame(sentence_dict)

def LoadDataSet(filename="./data/dataset.txt", xlabels=["WORD", "LANG", "NORM", "POS"], ylabel="CHUNK"):
    X = []
    y = []

    with open(filename) as f:
        _X, _y = [], []
        for line in f:
            if not line.strip() and _X:
                #add normailzed position to every word
                if "POSITION" in xlabels:
                    sent_length = len(_X)
                    for i, x_word in enumerate(_X):
                        x_word['POSITION'] = (i*1.0)/sent_length

                X.append(_X)
                y.append(_y)
                _X, _y = [], []
            else:
                try:
                    WORD, LANG, NORM, POS, CHUNK = map(lambda x: x.strip(), line.split('\t'))
                except:
                    print line

                x_word = {}
                for label in xlabels:
                    try:
                        x_word[label] = eval(label)
                    except NameError:
                        pass

                _X.append(x_word)
                _y.append(eval(ylabel))
        if _X:
            X.append(_X)
            y.append(_y)
    return X, y

def LoadDataSetWFeatures(filename="./data/datasetWfeatures.txt", xlabels=["WORD", "LANG", "NORM", "POS"], ylabel="CHUNK"):
    X = []
    y = []

    with open(filename) as f:
        _X, _y = [], []
        for line in f:
            if not line.strip() and _X:
                #add normailzed position to every word
                if "POSITION" in xlabels:
                    sent_length = len(_X)
                    for i, x_word in enumerate(_X):
                        x_word['POSITION'] = (i*1.0)/sent_length

                X.append(_X)
                y.append(_y)
                _X, _y = [], []
            else:
                try:
                    WORD, LANG, NORM, POS, CHUNK, EPOS, EPOSSCORE, HPOS, _HPOS = map(lambda x: x.strip(), line.split('\t'))
                except:
                    print line

                x_word = {}
                for label in xlabels:
                    try:
                        x_word[label] = eval(label)
                    except NameError:
                        pass

                _X.append(x_word)
                _y.append(eval(ylabel))
        if _X:
            X.append(_X)
            y.append(_y)
    return np.array(X), np.array(y)

def LoadDataSetWFeatures2(filename="./data/datasetWfeatures.txt"):
    X = []

    with open(filename) as f:
        _X, _y = [], []
        xlabels = ['WORD', 'NORM', 'POS', 'CHUNK', 'EPOS', 'EPOSSCORE', 'HPOS', '_HPOS', 'LANG', 'POSITION']
        for line in f:
            if not line.strip() and _X:
                #add normailzed position to every word
                if "POSITION" in xlabels:
                    sent_length = len(_X)
                    for i, x_word in enumerate(_X):
                        x_word['POSITION'] = (i*1.0)/sent_length

                X.append(_X)
                _X, _y = [], []
            else:
                WORD, LANG, NORM, POS, CHUNK, EPOS, EPOSSCORE, HPOS, _HPOS = map(lambda x: x.strip(), line.split('\t'))

                x_word = {}
                for label in xlabels:
                    try:
                        x_word[label] = eval(label)
                    except NameError:
                        pass

                _X.append(x_word)
        if _X:
            X.append(_X)
    return np.array(X)

def GetColumn(X, LABEL="LANG"):
    y = []
    for x in X:
        _y = []
        for obv in x:
            _y.append(obv[LABEL])
        y.append(_y)
    return y

def SeperateColumn(X, LABEL="LANG"):
    y = []
    for x in X:
        _y = []
        for obv in x:
            _y.append(obv[LABEL])
            del obv[LABEL]
        y.append(_y)
    return np.array(X), np.array(y)

def AddColumn(X, y, LABEL="LANG"):
    for x, _y in zip(X, y):
        for obv, label in zip(x, _y):
            obv[LABEL] = label
    return deepcopy(X)

def OverwriteColumn(X, y, LABEL="LANG"):
    for x, _y in zip(X, y):
        for obv, label in zip(x, _y):
            del obv[LABEL]
            obv[LABEL] = label
    return X

