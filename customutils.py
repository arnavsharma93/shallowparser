import numpy as np
import pickle
from sklearn.datasets.base import Bunch
from sklearn.metrics import make_scorer, accuracy_score
import itertools

def postagger_accuracy_score(y_true, y_pred, normailze=True):
    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    return accuracy_score(y_true, y_pred, normailze)

def LoadRaveeshSplit():
    filenames = ["./hin/train", "./hin/test"]
    retval = []
    for filename in filenames:
        data = []
        target = []

        cdata = []
        ctarget = []
        pos_tags = set()
        for line in open(filename):
            line = line.strip()
            if line:
                try:
                    line = line.split()
                except:
                    continue
                cdata.append(np.array([line[0], line[-2]]))
                ctarget.append(line[-1])
                pos_tags.add(line[-1])
            else:
                if cdata or ctarget:
                    data.append(np.array(cdata))
                    target.append(ctarget)
                    cdata = list()
                    ctarget = list()



        obj = Bunch(data=np.array(data), target=np.array(target), tags=np.array(pos_tags))
        retval.append(obj)
    return retval

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
