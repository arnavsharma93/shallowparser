from sklearn.datasets.base import Bunch
from collections import namedtuple
import numpy as np
import pandas as pd

def LoadDataSet(filename="./data/dataset.txt"):
    """Reads the file, and forms a sentence_dict which looks like this -
    {
    SENTENCE_NUMBER:
      [
        {
          WORD:,
          LANG:,
          POS:,
          POSITION:, # Represents word number position in a sentence.
        },
      ],
    }
    """
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

def LoadTestData(lang, unconstrained=False):
    """
    To be modified
    """
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


