import numpy as np
import os
import pickle
from sklearn.datasets.base import Bunch
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator
import itertools
from codemixeddata import *
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

def postagger_accuracy_score(y_true, y_pred, normailze=True):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    return accuracy_score(y_true, y_pred, normailze)

def create_norm_pickle(filename='data/final_normalization_predictions_wx'):
    from converter_indic import wxConvert
    import codecs
    con = wxConvert(order='wx2utf', lang='hin')
    normdict = {}
    with open(filename) as f:
        for lines in f:
            try:
                roman, predwx, actualwx = map(lambda x: x.strip(), lines.split())
                predutf = con.convert(predwx).decode('utf-8')
                normdict[roman] = (predwx, predutf)

            except:
                pass
    pickle.dump(normdict, open("data/normdict.p", "wb"))

def write_for_error_analysis(filepath, X_test, y_test, y_pred):
    filep = open(filepath, 'w')
    sents = []
    for X, Y in zip(X_test, y_test):
        sent = []
        for x, y in zip(X, Y):
            temp = list(x)
            temp.append(y)
            sent.append(temp)
        sents.append(sent)

    for sent, Y in zip(sents, y_pred):
        for obv, y in zip(sent, Y):
            obv.append(y)

    for sent in sents:
        for obv in sent:
            filep.write('\t'.join(obv) + '')

        filep.write('\n')
    filep.close()

def write_test_output(filepath, X_test, y_pred):
    filep = open(filepath, 'w')
    sents = []
    for X, Y in zip(X_test, y_pred):
        if len(X) != len(Y):
            print X
        sent = []
        for x, y in zip(X, Y):
            temp = list(x)
            temp.append(y)
            sent.append(temp)
        sents.append(sent)

    for sent in sents:
        for obv in sent:
            filep.write('\t'.join(obv) + '\n')

        filep.write('\n')
    filep.close()

def add_twitter_pos_tags():
    X, y = LoadDataSet()
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

    os.system('./ark_tweet/runTagger.sh --input-format conll ./data/dataset.txt | cut -f2,3 > ./ark_tweet/.ark_output')
    os.system('paste ./data/dataset.txt ./ark_tweet/.ark_output > ./data/datasetWfeatures.txt')

def add_transliterated_words():
    if not os.path.exists('./data/datasetWfeatures.txt'):
        raise Exception('first run add_twitter_pos_tags()')
    from irtrans import transliterator
    from converter_indic import wxConvert
    import codecs
    con = wxConvert(order='utf2wx', lang='hin')
    trn = transliterator(source='eng', target='hin')
    X, y = LoadDataSet()
    sents = []
    for x in X:
        sent = []
        for obv in x:
            sent.append(obv['WORD'])
        sents.append(" ".join(sent))
    out_sents = []
    for sent in sents:
        out_sent = trn.transform(sent)
        #out_sent = con.convert(out_sent).decode('utf-8')
        if len(sent.split(' ')) != len(out_sent.split(' ')):
            out_sent = ' ' + out_sent
        out_sents.append(out_sent)

    outfilename = './data/.translit'
    outf = codecs.open(outfilename, 'w', 'utf-8')
    for line in out_sents:
        for word in line.split(' '):
            outf.write(word.decode('utf-8') + '\n')
        outf.write('\n')

    os.system('paste ./data/datasetWfeatures.txt '+  outfilename +' > ./data/temp.txt')
    os.system('mv ./data/temp.txt ./data/datasetWfeatures.txt')

def create_dataset_with_features_file():
    add_twitter_pos_tags()
    add_transliterated_words()

def create_pickle_from_file(filename, lang):
    filename = "./hin/%s-trans" % lang
    mapper = dict()
    for line in open(filename):
        line = line.strip()
        if line:
            line = line.split()
            try:
                mapper[line[0].lower()] = line[1]
            except:
                raise Exception("Could not split into two %s " % ("\t".join(line)))
    pickle.dump(mapper, open("hin/%stranslit.p" % lang, "wb"))

def write_twitterpos_to_file():
    """
    To be generalized
    """
    outfile = open("./data/test/HI_EN_ARK.txt", "w")
    X = LoadTestData('hi')
    for i, (x1, x2) in enumerate(zip(data, X)):
        for (word_lang, ark_tag) in zip(x2, x1):
            outfile.write("%s\n" % " ".join([word_lang[0], word_lang[1], ark_tag[1]]))
        outfile.write("\n")
    outfile.close()
