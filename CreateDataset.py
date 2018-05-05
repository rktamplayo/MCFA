from __future__ import print_function

import numpy as np
import random
import re
import itertools
import cPickle

random.seed(1234)
np.random.seed(1234)

cv = 10
languages = ['mn', 'ru', 'ar', 'no', 'ko', 'uk', 'it', 'fi', 'pl', 'fr'] + ['en']
type = 'mr'

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

dataset = ['neg.txt.proc', 'pos.txt.proc']

data = []
vocabs = {}
maxes = {}

cur = 0
np.random.seed(1234)
for i, file in enumerate(dataset):
    print("file:", file)

    c = np.array([0] * len(dataset))
    c[i] = 1
    cur_len = len(data)
        
    for j, language in enumerate(languages):
        print("lang:", language)
            
        if language not in vocabs:
            vocabs[language] = {}
            maxes[language] = 0
            
        fin = open('data/' + type + '/' + language + '/' + file, 'r')
        count = 0
            
        for line in fin:
            words = line.lower().split()
            tokens = []
                
            for word in words:
                if word not in vocabs[language]:
                    vocabs[language][word] = len(vocabs[language])
                tokens.append(vocabs[language][word])
                
            maxes[language] = max(maxes[language], len(tokens))
                
            if j == 0:
                instance = {}
                instance['class'] = c
                instance['split'] = np.random.randint(0, cv)
                instance[language] = tokens
                data.append(instance)
            else:
                data[count+cur_len][language] = tokens
                
            count += 1
          
        fin.close()

np.random.seed(1234)
for language in vocabs:
    print(language)
        
    vocab = vocabs[language]
    word_vecs = np.random.uniform(-0.25, 0.25, (len(vocab)+1, 300))
        
    f = open('pretrained/wiki.' + language + '.vec', 'r')
    f.readline()
    for line in f:
        spl = line.split()
        if spl[0] in vocab:
            word_vecs[vocab[spl[0]]] = np.array([float(x) for x in spl[-300:]])
            
    f.close()
    
    print(len(vocab), len(word_vecs))
    cPickle.dump(word_vecs, open('vectors/' + type + '_' + language + ".p", "wb"))

cPickle.dump([data, vocabs, maxes], open('pickles/' + type + ".p", "wb"))
