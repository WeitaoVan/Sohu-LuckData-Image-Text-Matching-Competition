# create vocabulary.pkl from vocabulary.txt
import cPickle as pk
vocab_path = '/home/wwt/THULAC-Python/data/vocabulary_1w2'
vocab = {}
lines = open(vocab_path+'.txt').readlines()
for idx,line in enumerate(lines):
    line = line.strip().split(' ')
    vocab[line[0]] = int(line[1])
pk.dump(vocab, open(vocab_path+'.pkl', 'w'))