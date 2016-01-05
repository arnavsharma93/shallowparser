from codemixeddata import *
from customclassifiers import *
from preprocessor import *
from customutils import *

import posfeatures as pos
import lidffeatures as lidf
import chunkfeatures as chunk

from sklearn_crfsuite import scorers, metrics, CRF

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import cross_val_score, train_test_split, cross_val_predict, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, accuracy_score

def chunk_label_boundary_accuracy(y_pred, y_true):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    y_true_boundary = [label.split('-')[0] for label in y_true]
    y_pred_boundary = [label.split('-')[0] for label in y_pred]
    y_true_label = [label.split('-')[1] for label in y_true]
    y_pred_label = [label.split('-')[1] for label in y_pred]
    ret_dict = {
        'boundary': accuracy_score(y_true_boundary, y_pred_boundary),
        'label': accuracy_score(y_true_label, y_pred_label),
        'combined': accuracy_score(y_true, y_pred)
    }
    return ret_dict

X, y = LoadDataSetWFeatures(xlabels=['WORD', 'POSITION', 'EPOS', 'EPOSSCORE', 'LANG', 'NORM', 'HPOS', '_HPOS', 'POS'], ylabel='CHUNK')

kf = KFold(len(X), n_folds=10)

lidf_features = FeatureStacker([
    ('bnc_count', lidf.BNCCountsTransformer()),
    ('lex_norm', lidf.LexNormCountsTransformer()),
    ('in_bnc_or_lexnorm', lidf.LexNormBNCSpecialTransformer()),
    ('hindi_dict', lidf.HindiDictionaryCountsTransformer()),
    ('capitalization', lidf.CapitalizationTransformer()),
    ('cngram', lidf.CharacterNgramTransformer()),
    ('affixes', lidf.AffixesTransformer())
])

lidf_model = Pipeline([
    ('features', lidf_features),
    ('classifier', CRF())
])

pos_features = FeatureStacker([
    ('lex', pos.AddLexTransformer()),
    ('lang', pos.AddLangTransformer()),
    ('computed_pos', pos.ComputedPOSTransformer()),
    ('computed_pos_confidence', pos.PoSConfidenceTransformer()),
    ('affixes', pos.AffixesTransformer()),
    ('en_clusters', pos.HWCTransformer())
  ])

pos_model = Pipeline([
    ('features', pos_features),
    ('classifier', CRF())
])
pos_model.set_params(features__affixes__strategy='all_raw', features__computed_pos__strategy='only_en')

chunk_features = FeatureStacker([
    ('lex', pos.AddLexTransformer()),
    ('lang', pos.AddLangTransformer()),
    ('computed_pos', pos.ComputedPOSTransformer()),
    ('en_rest_pos_confidence', pos.PoSConfidenceTransformer()),
    ('context', pos.ContextTransformer()),
    ('affixes', pos.AffixesTransformer()),
    ('en_clusters', pos.HWCTransformer()),
    ('pos', chunk.PoSTransformer()),
    ('pos_context', chunk.POSContextTransformer()),
    ('lex__predicted_pos', chunk.LexPoSTransformer()),
])

chunk_model = Pipeline([
        ('features', chunk_features),
        ('classifier', CRF())
])
chunk_model.set_params(features__affixes__strategy='all_raw', features__computed_pos__strategy='only_en')

norm_model = NormEstimator(rebuild=False)
accuracy = []
fold_num = 1
label = []
boundary = []
combined = []

for train_index, test_index in kf:
        print "Running Fold %d" % fold_num
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Language Identification
        lidf_model.fit(X_train, GetColumn(X_train, 'LANG'))
        X_test_lang = GetColumn(X_test, 'LANG')
        X_test = OverwriteColumn(X_test, lidf_model.predict(X_test), 'LANG')
        print "\tLanguage identification done"

        #Normalization
        #X_test_norm = GetColumn(X_test, 'NORM')
        #X_test = OverwriteColumn(X_test, norm_model.predict(X_test), 'NORM')
        #print "\tNormalization done"

        #HPOS - depends on Normalization
        #X_test_hpos = GetColumn(X_test, 'HPOS')
        #hpos_pred = GetColumn(X_test, '_HPOS')
        #X_test = OverwriteColumn(X_test, hpos_pred, 'HPOS')
        #print "\tHPOS added"

        #pos model
        pos_model.fit(X_train, GetColumn(X_train, 'POS'))
        X_test_pos = GetColumn(X_test, 'POS')
        X_test = OverwriteColumn(X_test, pos_model.predict(X_test), 'POS')
        print "\tPOS tagging done"

        #chunk pred
        chunk_model.fit(X_train, y_train)
        y_pred = chunk_model.predict(X_test)

        #accuracy
        current = chunk_label_boundary_accuracy(y_test, y_pred)
        print '\t', current
        label.append(current['label'])
        boundary.append(current['boundary'])
        combined.append(current['combined'])

        #restore columns in X_test
        X_test = OverwriteColumn(X_test, X_test_lang, 'LANG')
        #X_test = OverwriteColumn(X_test, X_test_norm, 'NORM')
        #X_test = OverwriteColumn(X_test, X_test_hpos, 'HPOS')
        X_test = OverwriteColumn(X_test, X_test_pos, 'POS')

        fold_num += 1

label = np.array(label)
boundary = np.array(boundary)
combined = np.array(combined)
print "\nLabel accuracy %.2f" % (100 * label.mean())
print "\nBoundary %.2f" % (100 * boundary.mean())
print "\nCombined accuracy %.2f" % (100 * combined.mean())


class CMSTPipeline(BaseEstimator):
    '''
    Pipeline for parsing CMST
    It takes an argument steps - list of tuples where each tuple contains the name
    of the step and its instantiated model
    Last step has to predict
    '''
    def __init__(self, steps):
        self.final_step = steps.pop()
        self.steps = steps

    def fit(self, X, y):
        for i in xrange(len(self.steps)):
            self.steps[i][1].fit(X, GetColumn(X, self.steps[i][0]))

        self.final_step[1].fit(X, y)

    def predict(self, X):
        X_backup = []
        for name, model in self.steps:
            X_backup.append(GetColumn(X, name))
            X = OverwriteColumn(X, model.predict(X), name)

        y_pred = self.final_step[1].predict(X)
        for i, name, model in enumerate(self.steps):
            X = OverwriteColumn(X, X_backup[i], name)

        return y_pred

