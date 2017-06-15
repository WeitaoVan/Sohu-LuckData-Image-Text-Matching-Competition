import cPickle as pk
import numpy

root = '/home/wwt/THULAC-Python/data/'
lines = open(root + 'vocabulary_1w2.txt').readlines()
save_path = root + 'word2id_1w2.pkl'
word2id = {}
for idx, line in enumerate(lines):
    word = line.split()[0]
    word2id[word] = idx
pk.dump(word2id, open(save_path, 'w'))