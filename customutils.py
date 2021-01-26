import numpy as np
import requests
import os
import pickle
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator
import itertools
from codemixeddata import *
from customclassifiers import *
from wxconv import WXC as wxConvert
from indictrans import Transliterator as transliterator
#from irtrans import transliterator
#from converter_indic import wxConvert
import codecs
from estimators import *

class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def transform(self, X):
        def combine(f1, f2):
            if not f1:
                return list(f2)

            F = []
            for X1, X2 in zip(f1, f2):
                out = []
                for x1, x2 in zip(X1, X2):
                    temp = []
                    temp.extend(x1) if type(x1) == list else temp.append(x1)
                    temp.extend(x2) if type(x2) == list else temp.append(x2)
                    out.append(temp)
                F.append(out)
            return F

        features = []
        for name, trans in self.transformer_list:
            features = combine(features, trans.transform(X))

        return np.array(features)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out

def chunk_label_boundary_accuracy(y_pred, y_true):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    y_true_boundary = [label.split('-')[0] for label in y_true]
    y_pred_boundary = [label.split('-')[0] for label in y_pred]
    y_true_label = [label.split('-')[1] for label in y_true]
    y_pred_label = [label.split('-')[1] for label in y_pred]
    ret_dict = {
        'BOUNDARY': accuracy_score(y_true_boundary, y_pred_boundary),
        'LABEL': accuracy_score(y_true_label, y_pred_label),
        'COMBINED': accuracy_score(y_true, y_pred)
    }
    return ret_dict

def postagger_accuracy_score(y_true, y_pred, normailze=True):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    return accuracy_score(y_true, y_pred, normailze)

def write_colums_to_file(y1, y2, name):
    fp = open('./data/.%s_err' % name, 'w')
    for _y1, _y2 in zip(y1, y2):
        for l1, l2 in zip(_y1, _y2):
            fp.write('%s\t%s\n' % (l1, l2))
        fp.write('\n')

def fix_indent_dataset_file():
    X, y = LoadDataSet()
    of = open('./data/dataset.txt', 'w')
    for x, _y in zip(X, y):
        for obv, label in zip(x, _y):
            line = []
            line.append(obv['WORD'])
            line.append(obv['LANG'])
            line.append(obv['NORM'])
            line.append(obv['POS'])
            line.append(label)
            of.write('\t'.join(line) + '\n')
        of.write('\n')
    of.close()

def run_hindi_pos_tagger(X, gold):
    ypos = []
    if not gold:
        norm_model = NormEstimator(rebuild=True, script='utf')
        _y = norm_model.predict(X)
        X = OverwriteColumn(X, _y, 'NORM')
    else:
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
    return ypos

def convert_pos_tag(word, tag):
    mapper = {'CC': 'CONJ',
     'CCC': 'CONJ',
     'DEM': 'DET',
     'INFT': 'ADV',
     'INFTC': 'ADV',
     'INJ': 'INTJ',
     'JJ': 'ADJ',
     'JJC': 'ADJ',
     'NEG': 'PART',
     'NN': 'NOUN',
     'NN ': 'ADV',
     'NNC': 'NOUN',
     'NNP': 'PROPN',
     'NNPC': 'PROPN',
     'NST': 'ADV',
     'PRP': 'PRON',
     'PRP ': 'ADV',
     'PRPC': 'PRON',
     'PSP': 'ADP',
     'PSPC': 'ADP',
     'QC': 'NUM',
     'QCC': 'NUM',
     'QF': 'DET',
     'QFC': 'DET',
     'QO': 'ADJ',
     'RB': 'ADV',
     'RBC': 'ADV',
     'RP': 'PART',
     'RPC': 'PART',
     'SYM': 'PUNCT',
     'UNK': 'X',
     'VAUX': 'AUX',
     'VAUXC': 'AUX',
     'VM': 'VERB',
     'VMC': 'VERB',
     'WQ': 'DET',
     'WQ ': 'ADV'}
    return mapper[tag]

def hindi_pos_tags(X, gold=True, rerun=False):
    outfilename = './data/.hin_pos_tags_%s.out' % ('gold' if gold else 'pred')
    if rerun:
        fp = open('./data/.hin_pos_tags_%s.out', 'w')
        ypos = run_hindi_pos_tagger(X, gold)
        for _y1 in ypos:
            for l1 in _y1:
                if not l1.strip():
                    l1 = 'UNK'
                fp.write('%s\n' % l1)
            fp.write('\n')
        fp.close()
        return ypos
    else:
        outfile = open(outfilename, 'r')
        y = []
        _y = []
        for line in outfile:
            if not line.strip() and _y:
                y.append(_y)
                _y = []
            else:
                _y.append(line.strip())
        if _y:
            y.append(_y)
        return y

def twitter_pos_tags(X, rerun=False):
    outfilename = './ark_tweet/.ark_output'
    if rerun:
        sents = []
        for x in X:
            sent = []
            for obv in x:
                sent.append(obv['WORD'])
            sents.append(sent)

        inf = open("./ark_tweet/.ark_input", "w")
        for sent in sents:
            inf.write(' '.join(sent) + "\n")
        inf.close()

        os.system('./ark_tweet/runTagger.sh --output-format conll ./ark_tweet/.ark_input | cut -f2,3 > ' + outfilename)

    outfile = open(outfilename, 'r')
    y_epos = []
    y_score = []
    y = []
    _y = []
    for line in outfile:
        if not line.strip() and _y:
            y_epos.append(y)
            y_score.append(_y)
            _y = []
            y = []
        else:
            epos, score = map(lambda word: word.strip(), line.split())
            _y.append(score)
            y.append(epos)
    if _y:
        y_score.append(_y)
        y_epos.append(y)
    return y_epos, y_score

def write_dataset_to_file(X, filename='./data/datasetWfeatures.txt'):
    of = codecs.open(filename, 'w')
    for x in X:
        for obv in x:
            line = []
            line.append(obv['WORD'])
            line.append(obv['LANG'])
            line.append(obv['NORM'])
            line.append(obv['POS'])
            line.append(obv['CHUNK'])
            line.append(obv['EPOS'])
            line.append(obv['EPOSSCORE'])
            line.append(obv['HPOS'])
            line.append(obv['_HPOS'])
            of.write('\t'.join(line) + '\n')
        of.write('\n')
    of.close()

def create_dataset_with_features_file(rerun_pos_tagger=False):
    X, y = LoadDataSet(xlabels=['NORM', 'LANG', 'WORD', 'POS', 'CHUNK'], ylabel='CHUNK')
    X_EPOS, X_EPOSSCORE = twitter_pos_tags(X, rerun_pos_tagger)
    X_HPOS = hindi_pos_tags(X, gold=True, rerun=rerun_pos_tagger)
    X__HPOS = hindi_pos_tags(X, gold=False, rerun=rerun_pos_tagger)
    X = AddColumn(X, X_EPOS, 'EPOS')
    X = AddColumn(X, X_EPOSSCORE, 'EPOSSCORE')
    X = AddColumn(X, X_HPOS, 'HPOS')
    X = AddColumn(X, X__HPOS, '_HPOS')
    write_dataset_to_file(X)

