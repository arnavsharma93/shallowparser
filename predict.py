'''
Runner file for the pipeline
'''
from codemixeddata import *
from customclassifiers import *
from preprocessor import *
from customutils import *
from estimators import *
from pipeline import CMSTPipeline

import posfeatures as pos
import lidffeatures as lidf
import chunkfeatures as chunk

from sklearn_crfsuite import scorers, metrics, CRF

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import cross_val_score, train_test_split, cross_val_predict, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, accuracy_score
from collections import namedtuple
from sklearn.externals import joblib

from subprocess import check_output

lidf_model = joblib.load('./models/lidf.p')
norm_model = NormEstimator(rebuild=True, script='wx')
pos_model = joblib.load('./models/pos.p')
chunk_model = joblib.load('./models/chunk.p')

def add_language_idf(X_test):
    y_pred = lidf_model.predict(X_test)
    return y_pred

def add_norm_model(X_test):
    y_pred = norm_model.predict(X_test)
    return y_pred

def add_hi_pos_tagger(X):
    ypos = []
    trn = transliterator(source='eng', target='hin')
    for x in X:
        for obv in x:
            if obv['LANG'] == 'en':
                obv['NORM'] = trn.transform(obv['LANG'])

    for x in X:
        sent = []
        for obv in x:
            sent.append(obv['NORM'])
        sent = ' '.join(sent)
        payload = {'input': sent}
        r = requests.post('http://api.ilmt.iiit.ac.in/hin/pan/1/4', data=payload)
        pos_output = r.json()['postagger-4']
        _y = []
        for line in pos_output.split('\n'):
            with_tag = False
            try:
                int(line[0])
                with_tag = True
            except:
                pass
            if with_tag:
                _y.append(line.split()[2])
        ypos.append(_y)
    for x in X:
        for obv in x:
            if obv['LANG'] == 'en':
                obv['NORM'] = obv['WORD']
    return ypos

def add_pos_tag(X):
    y_pred = pos_model.predict(X)
    return y_pred

def add_chunk_tag(X):
    y_pred = chunk_model.predict(X)
    return y_pred

def tokenize_epos_eposscore(sentence):
    inpf = open('input_file', 'w')
    inpf.write(sentence + '\n')
    inpf.close()
    tpos_command = './ark_tweet/runTagger.sh --output-format conll %s' % inpf.name
    out = check_output(tpos_command.split())
    X_test = []
    x = []
    for line in out.split('\n'):
        line = line.strip()
        if line:
            WORD, EPOS, EPOSSCORE = map(lambda d: d.strip(), line.split('\t'))

            obv = {'WORD': WORD, 'EPOS': EPOS, 'EPOSSCORE': EPOSSCORE}
            x.append(obv)
    X_test.append(np.array(x))
    X_test = np.array(X_test)
    return X_test

def print_output(X):
    for x in X:
        for obv in x:
            line = [obv['WORD'], obv['LANG'], obv['POS'], obv['CHUNK']]
            print '\t'.join(line)
        print '\n'


sentence = raw_input()
X_test = tokenize_epos_eposscore(sentence)

AddColumn(X_test, add_language_idf(X_test), 'LANG')
AddColumn(X_test, add_norm_model(X_test), 'NORM')
AddColumn(X_test, add_hi_pos_tagger(X_test), 'HPOS')
AddColumn(X_test, add_pos_tag(X_test), 'POS')
AddColumn(X_test, add_chunk_tag(X_test), 'CHUNK')

print_output(X_test)
