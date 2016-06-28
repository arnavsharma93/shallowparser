'''
Runner file for the pipeline
'''
from estimators import *
from customutils import *
import numpy as np
from sklearn.externals import joblib
import string
import re
import csv
import pickle

from subprocess import check_output

lidf_model = joblib.load('./models/lidf.p')
norm_model = NormEstimator(rebuild=True, script='wx')
pos_model = joblib.load('/home/indira/Downloads/pos.p')
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
        #print "hello i am here"
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

def add_dummy_hi_pos(X):
    Y = []
    for x in X:
        y = []
        for obv in x:
            y.append('UNK')
        Y.append(y)
    return Y


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
            line = [obv['WORD'], obv['LANG'], obv['HPOS'], obv['CHUNK']]
            print '\t'.join(line)
        print '\n'

def tokenizer(sentence):
    X_test = tokenize_epos_eposscore(sentence)
    return X_test

def language_identifier(sentence):
    X_test = tokenize_epos_eposscore(sentence)

    AddColumn(X_test, add_language_idf(X_test), 'LANG')
    return X_test

def pos_tagger(sentence):
    X_test = tokenize_epos_eposscore(sentence)

    AddColumn(X_test, add_language_idf(X_test), 'LANG')
    AddColumn(X_test, add_norm_model(X_test), 'NORM')
    return X_test

def shallow_parser(sentence):
    X_test = tokenize_epos_eposscore(sentence)

    AddColumn(X_test, add_language_idf(X_test), 'LANG')
    AddColumn(X_test, add_norm_model(X_test), 'NORM')
    AddColumn(X_test, add_hi_pos_tagger(X_test), 'HPOS')
    #AddColumn(X_test, add_dummy_hi_pos(X_test), 'HPOS')
    AddColumn(X_test, add_pos_tag(X_test), 'POS')
    #AddColumn(X_test, add_chunk_tag(X_test), 'CHUNK')
    return X_test


if __name__ == '__main__':

    df = pd.read_csv('/home/indira/IASNLP/textAnalysisPrecog/data/nerData.csv')
    sentences = df['text'][0:10]
    #sentences = ["NIGHT smS * *  * *  *     Teri Palkon Mein Rehna Hy Raat Bhar K Liye a\r Dost\r Main To Ek Khwab Hoon Subha Ko Chala Jaon Ga.'Good Night'"]
    sents = []
    nes_list = []
    nesn_list = []

    exclude = set(string.punctuation)
    for sent in sentences:
        sent = ''.join([ch if ch not in exclude else ' ' for ch in sent])
        sent = re.sub(r'[^\x00-\x7F]+',' ', sent)
        sents.append(sent) 


    X_tests = []
    for sentence in sents:
        X_tests.append(shallow_parser(sentence))
    #print_output(X_test)
    #print X_test
    for X_test in X_tests:
        nes = []
        nesn = []
        for x in X_test:
            for obv in x:
                line = [obv['WORD'], obv['LANG'], obv['POS']]
                if obv['POS'] == 'PROPN' or obv['POS'] == 'NOUN':
                    nesn.append(obv['WORD'])
                if obv['POS'] == 'PROPN':
                    nes.append(obv['WORD'])
                print '\t'.join(line)
            print '\n'
        nes_list.append(nes)
        nesn_list.append(nesn)

    out = []
    for i, word in enumerate(sentences):
        out.append([word, ann1[i], ann2[i], ann3[i], nes_list[i], nesn_list[i]])

    
    #write the csv  
    
    with open('/home/indira/textAnalysisPrecog/data/nerDataTaggedPOS.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "ann1", "ann2", "ann3", "nes", "nesn"])
        writer.writerows(out)
    
