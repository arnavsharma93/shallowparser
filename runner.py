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
    #('en_clusters', pos.HWCTransformer())
  ])

pos_model = Pipeline([
    ('features', pos_features),
    ('classifier', CRF())
])
pos_model.set_params(features__affixes__strategy='all_norm', features__computed_pos__strategy='combined')

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
chunk_model.set_params(features__affixes__strategy='all_norm', features__computed_pos__strategy='combined')

norm_model = Pipeline([
    ('classifier', NormEstimator(rebuild=True))
])

hpos_model = Pipeline([
    ('classifier', HPOSEstimator())
])

data = LoadDataSetWFeatures2()

step = namedtuple('step', 'name model')

model = CMSTPipeline([
    #step('LANG', lidf_model),
    #step('NORM', norm_model),
    #step('HPOS', hpos_model),
    #step('POS', pos_model),
    step('CHUNK', chunk_model),
])
X, y = SeperateColumn(data, model.final_step.name)
accuracy = []
kf = KFold(len(X), n_folds=10)
fold_num = 1

for train_index, test_index in kf:
    print "fold number %d" % fold_num
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    current = postagger_accuracy_score(y_pred, y_test)
    print '\t%.2f' % (current * 100)

    fold_num += 1
    accuracy.append(current)

print '%.2f' % (np.array(accuracy).mean() * 100)


