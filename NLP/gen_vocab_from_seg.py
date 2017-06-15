# generate vocabulary based on segmented texts
import cPickle as pk
import time
import numpy as np
import SH_split
seg_root = '/media/wwt/860G/data/souhu_data/fusai/train_seg/'
match_list = '/media/wwt/860G/data/souhu_data/fusai/trainMatching_shuffle.txt'
save_path = '/home/wwt/THULAC-Python/data/vocabulary'
SPLIT = '      '
lines = open(match_list).readlines()
N = len(lines)
vocab = {}
t0 = time.time()
for idx, line in enumerate(lines):
    filename = line.strip().split(SPLIT)[1]
    words = open(seg_root+filename).readlines()[0].split(' ')
    SH_split.update_vocab(vocab, words)
    if np.mod(idx, 10000) == 0:
        print '%d/%d, time = %d minutes' %(idx, N, (time.time() - t0)/60)
SH_split.save_vocab_txt(vocab, save_path+'.txt')
pk.dump(vocab, open(save_path+'.pkl', 'w'))