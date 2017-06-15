import numpy as np
import cPickle as pk

concept_path = '/home/wwt/THULAC-Python/data/train_concept.txt'
vocab_path = '/home/wwt/THULAC-Python/data/word2id_concept.pkl'
vocab = pk.load(open(vocab_path))
lines = open(concept_path).readlines()
N = len(lines)
nHit = 0
for line in lines:
    spt = line.split(' ')
    for word in spt:
        try:
            _ = vocab[word]
            nHit += 1
            break
        except:
            pass
print '%d hit / %d' %(nHit, N)
    
