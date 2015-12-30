from sklearn.datasets.base import Bunch
from collections import namedtuple
import numpy as np

def LoadDataSet(lang):
    if lang == 'hi':
        filename = './data/HI_EN_TRAIN_ARK.txt'
        line_len = 4
    elif lang == 'ta':
        filename = './data/TA_EN_TRAIN.txt'
        line_len = 3
    elif lang == 'bn':
        filename = './data/BN_EN_TRAIN.txt'
        line_len = 3
    else:
        raise Exception("Language not identified")

    data = []
    target = []

    cdata = []
    ctarget = []
    pos_tags = set()
    for line in open(filename):
        line = line.strip()
        if line:
            line = line.split()
            if len(line) != line_len:
                continue
            #try:
                #word, lang, cluster, tag = line.split()
            #except:
                #continue
            cdata.append(np.array(line[:-1]))
            ctarget.append(line[-1])
            pos_tags.add(line[-1])
        else:
            if cdata or ctarget:
                data.append(np.array(cdata))
                target.append(ctarget)
                cdata = list()
                ctarget = list()
    if cdata or ctarget:
        data.append(np.array(cdata))
        target.append(ctarget)
        cdata = list()
        ctarget = list()


    return Bunch(
            data=np.array(data),
            target=np.array(target),
            tags=pos_tags
            )

def LoadTestData(lang, unconstrained=False):
    if lang == 'hi':
        if unconstrained:
            filename = './data/test/HI_EN_ARK.txt'
        else:
            filename = './data/test/HI_EN.txt'

    elif lang == 'ta':
        filename = './data/test/TA_EN.txt'
    elif lang == 'bn':
        filename = './data/test/BN_EN.txt'
    else:
        raise Exception("Language not identified")

    data = []

    cdata = []
    for line in open(filename):
        line = line.strip()
        if line:
            line = line.split()
            #try:
                #word, lang, cluster, tag = line.split()
            #except:
                #continue
            cdata.append(np.array(line))
        else:
            if cdata or ctarget:
                data.append(np.array(cdata))
                cdata = list()
            else:
                print "Another empty line found"
    if cdata:
        data.append(np.array(cdata))

    data=np.array(data)
    return data


