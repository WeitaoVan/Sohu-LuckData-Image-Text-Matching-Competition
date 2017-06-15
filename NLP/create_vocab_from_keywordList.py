# 
import numpy as np
import cPickle as pk
from SH_split import save_vocab_txt

keyword_path = '/home/wwt/THULAC-Python/data/keyword.txt'
save_path = '/home/wwt/THULAC-Python/data/vocab_keyword.txt'
K = 1

vocab = {}
for idx, line in enumerate(open(keyword_path).readlines()):
    spt = line.strip().split(' ')
    for word in spt[:min(K, len(spt))]:
        try:
            vocab[word] += 1
        except:
            vocab[word] = 1
    if np.mod(idx, 10000) == 0:
        print '# %d' %idx    
save_vocab_txt(vocab, save_path)