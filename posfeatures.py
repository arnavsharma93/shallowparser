import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import re
import pickle


class AddLexTransformer(BaseEstimator):
    '''
    Adds the lex as a feature
    '''
    def transform(self, X, **transform_params):
        return [['LEX:%s' % obv['WORD'].lower() for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class ComputedPOSTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, strategy='combined', ignore=False):
        self.ignore = ignore
        self.strategy = strategy

    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for obv in x:
                if self.strategy == 'combined':
                    pos = obv['HPOS'] if obv['LANG'] == 'hi' else obv['EPOS']
                elif self.strategy == 'only_hi':
                    pos = obv['HPOS']
                elif self.strategy == 'only_en':
                    pos = obv['EPOS']
                else:
                    raise Exception('Invalid strategy')
                _x.append('ComputedPOS:%s' % pos)

            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self


class PoSConfidenceTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['EPOSSCORE:%s' % (obv['EPOSSCORE'][2:4] if (not self.ignore and obv['LANG'] != 'hi') else 'False')  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class NormLexTransformer(BaseEstimator):
    '''
    Adds the POS tag computed from other sources as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return [['NORMLEX:%s' % (obv['NORM'] if (not self.ignore and obv['LANG'] == 'hi') else 'False')  for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class AddLangTransformer(BaseEstimator):
    '''
    Adds the gold language id as a feature
    '''
    def transform(self, X, **transform_params):
        return [['LANG:%s' % obv['LANG'] for obv in x] for x in X]

    def fit(self, X, Y=None):
        return self

class ContextTransformer(BaseEstimator):
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
                        window_word = x[i+j]['WORD']
                    f = '%d:word.lower=%s' % (j, window_word.lower())
                    _obv.append(f)
                _x.append(_obv)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

class AffixesTransformer(BaseEstimator):
    '''
    Adds suffixes of length from [1, suffix_len] as features
    Adds prefixes of length from [1, prefix_len] as features
    forms:
    0 - Hin Norm
    1 - Hin Norm + Eng + Rest
    2 - Hin + Eng + Rest
    '''
    def __init__(self, suffix_len=5, prefix_len=5, strategy='only_hi'):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.strategy = strategy

    def transform(self, X, Y=None, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                lex = ''
                if self.strategy in ('all_norm', 'all_raw'):
                    lex = obv['WORD']
                if self.strategy in ('only_hi', 'all_norm') and obv['LANG'] == 'hi':
                    lex = obv['NORM']

                _obv = []

                for j in xrange(-self.suffix_len, 0):
                    suffix = lex[j:] if abs(j) <= len(lex) else ''
                    f = 'word[%d:]=%s' % (j, suffix)
                    _obv.append(f)

                for j in xrange(1, self.prefix_len+1):
                    prefix = lex[:j] if abs(j) <= len(lex) else ''
                    f = 'word[:%d]=%s' % (j, prefix)
                    _obv.append(f)

                _x.append(_obv)
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self


class NumeralTransformer(BaseEstimator):
    '''
    Binary feature: Numeral or not
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        _X = []
        for x in X:
            _x = []
            for i, obv in enumerate(x):
                lex = obv['WORD'].lower()
                val = True if (re.search(ur"^[+-]?\d+[-+\d]+$", lex) and not self.ignore) else False
                _x.append("numeral:%s" % ("True" if val else "False"))
            _X.append(_x)
        return _X

    def fit(self, X, Y=None):
        return self

class NormalizeTransformer(BaseEstimator):
    '''
    Normalization of Hindi and English words
    '''
    def __init__(self, ignore=False):
        self.mapper = pickle.load(open("data/hitranslit.p", "rb"))
        d2 = pickle.load(open("data/entranslit.p", "rb"))
        self.mapper.update(d2)
        self.ignore = ignore


    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                lex = obv['WORD'].lower()
                val = self.mapper[lex] if (lex in self.mapper and not self.ignore)  else "False"
                features.append("translit:%s" % val)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class HWCTransformer(BaseEstimator):
    '''
    Heirarchial Word Cluster Transformer for English
    stored in a pickled dict in the form word => cluster freq
    '''
    def __init__(self, ignore=False):
        self.mapper = pickle.load(open("data/eng_clusters.p", "rb"))
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                lex = obv['WORD'].lower()
                val = self.mapper[lex][0] if (lex in self.mapper and not self.ignore)  else -1
                features.append("HWC:%s" % val)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class FirstCharUpperTransformer(BaseEstimator):
    '''
    Binary feature: First character is uppercase for languages ("en", "ne")
    '''

    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                word = obv['WORD']
                val = True if (re.search(ur"^[A-Z]+", word) and \
                                obv['LANG'] in ("en", "ne") and \
                                not self.ignore) \
                            else False
                features.append("fupper:%s" % val)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class AnyCharUpperTransformer(BaseEstimator):
    '''
    Binary feature: Any character is uppercase
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                word = obv['WORD']
                val = True if (re.search(ur'[A-Z]+', word) and \
                                obv['LANG'] in ("en", "ne") and \
                                not self.ignore) \
                            else False
                features.append("upper:%s" % val)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class SymbolTransformer(BaseEstimator):
    '''
    Binary feature: Word is a symbol
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                lex = obv['WORD']
                val = True if (re.search(ur'^[^a-zA-Z0-9\'\",.?!;]+$', lex) and \
                                not self.ignore) \
                            else False
                features.append("symbol:%s" % ("True" if val else "False"))
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

