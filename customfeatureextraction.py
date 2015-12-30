import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import re
import pickle

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

class AddLexTransformer(BaseEstimator):
    '''
    Adds the lex as a feature
    '''
    def transform(self, X, **transform_params):
        return np.array([[lex.lower() for lex in x[:,0]] for x in X])

    def fit(self, X, Y=None):
        return self

class PoSTransformer(BaseEstimator):
    '''
    Adds the lex as a feature
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return np.array([[obv[2] if (obv[1] != "hi" and not self.ignore) else "False" for obv in x] for x in X])

    def fit(self, X, Y=None):
        return self

class GoldLangTransformer(BaseEstimator):
    '''
    Adds the gold language id as a feature
    '''
    def transform(self, X, **transform_params):
        return np.array([x[:,1] for x in X])

    def fit(self, X, Y=None):
        return self

class ContextTransformer(BaseEstimator):
    '''
    Adds the words in the context window [start, end] of current word
    as features
    '''
    def __init__(self, start=-1, end=2):
        self.start = start
        self.end = end

    def transform(self, X, Y=None, **transform_params):
        transx = []
        for x in X:
            x = x[:,0]
            features = []
            for i, lex in enumerate(x):
                feature_lex = []
                for j in xrange(self.start, self.end+1):
                    if not j:
                        continue

                    window_word = ''
                    if j != 0 and (i+j) >= 0 and (i+j) < len(x):
                        window_word = x[i+j]
                    f = '%d:word.lower=%s' % (j, window_word.lower())
                    feature_lex.append(f)
                features.append(feature_lex)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class AffixesTransformer(BaseEstimator):
    '''
    Adds suffixes of length from [1, suffix_len] as features
    Adds prefixes of length from [1, prefix_len] as features
    '''
    def __init__(self, suffix_len=5, prefix_len=5, hin_prefix_len=5, hin_suffix_len=5):
        self.mapper = pickle.load(open("hin/hitranslit.p", "rb"))
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.hin_prefix_len = hin_prefix_len
        self.hin_suffix_len = hin_suffix_len

    def transform(self, X, Y=None, **transform_params):
        transx = []
        for x in X:
            x = x[:,0]
            features = []
            for i, lex in enumerate(x):
                #lex = self.mapper[lex] if lex in self.mapper else lex.lower()
                lex = lex.lower()
                feature_lex = []

                for j in xrange(-self.suffix_len, 0):
                    suffix = lex[j:] if abs(j) <= len(lex) else ''
                    f = 'word[%d:]=%s' % (j, suffix)
                    feature_lex.append(f)

                for j in xrange(1, self.prefix_len+1):
                    prefix = lex[:j] if abs(j) <= len(lex) else ''
                    f = 'word[:%d]=%s' % (j, prefix)
                    feature_lex.append(f)

                if lex in self.mapper:
                    lex = self.mapper[lex]
                    for j in xrange(-self.hin_suffix_len, 0):
                        suffix = lex[j:] if abs(j) <= len(lex) else ''
                        f = 'word[%d:]=%s' % (j, suffix)
                        feature_lex.append(f)

                    for j in xrange(1, self.hin_prefix_len+1):
                        prefix = lex[:j] if abs(j) <= len(lex) else ''
                        f = 'word[:%d]=%s' % (j, prefix)
                        feature_lex.append(f)


                features.append(feature_lex)
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self


class NumeralTransformer(BaseEstimator):
    '''
    Binary feature: Numeral or not
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            x = x[:,0]
            features = []
            for i, lex in enumerate(x):
                val = True if (re.search(ur"^[+-]?\d+[-+\d]+$", lex) and not self.ignore) else False
                features.append("numeral:%s" % ("True" if val else "False"))
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class NormalizeTransformer(BaseEstimator):
    '''
    Normalization of Hindi and English words
    '''
    def __init__(self, ignore=False):
        self.mapper = pickle.load(open("hin/hitranslit.p", "rb"))
        d2 = pickle.load(open("hin/entranslit.p", "rb"))
        self.mapper.update(d2)
        self.ignore = ignore


    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                lex = obv[0].lower()
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
        self.mapper = pickle.load(open("eng_clusters.p", "rb"))
        self.ignore = ignore

    def transform(self, X, **transform_params):
        return np.array([[self.mapper[lex][0] if (lex in self.mapper and not self.ignore) else "False" for lex in x[:,0]] for x in X])

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
                val = True if (re.search(ur"^[A-Z]+", obv[0]) and \
                                obv[1] in ("en", "ne") and \
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
                val = True if (re.search(ur'[A-Z]+', obv[0]) and \
                                obv[1] in ("en", "ne") and \
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
            x = x[:,0]
            features = []
            for i, lex in enumerate(x):
                val = True if (re.search(ur'^[^a-zA-Z0-9\'\",.?!;]+$', lex) and \
                                not self.ignore) \
                            else False
                features.append("symbol:%s" % ("True" if val else "False"))
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self

class UnivTransformer(BaseEstimator):
    '''
    Binary feature: Lang id is univ
    '''
    def __init__(self, ignore=False):
        self.ignore = ignore

    def transform(self, X, **transform_params):
        transx = []
        for x in X:
            features = []
            for i, obv in enumerate(x):
                val = (obv[1] == "univ" and not self.ignore)
                features.append("univ:%s" % ("True" if val else "False"))
            transx.append(features)
        return np.array(transx)

    def fit(self, X, Y=None):
        return self


