import cPickle as pk
import numpy

words = open('/home/wwt/THULAC-Python/data/vocabulary_nv_4w.txt').readlines()
save_path = '/home/wwt/THULAC-Python/data/id2word_nv_4w.pkl'
id2word = {}
for idx, word in enumerate(words):
    id2word[idx] = word.split()[0]
pk.dump(id2word, open(save_path, 'w'))