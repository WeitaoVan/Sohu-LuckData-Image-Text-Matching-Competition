#-*- coding: utf-8 -*-
import sys
import thulac
import numpy as np
import time
reload(sys)
sys.setdefaultencoding('utf-8')
vocab_path = '/home/wwt/THULAC-Python/data/vocabulary.txt'
save_path = '/home/wwt/THULAC-Python/data/vocabulary_nv.txt'
thu1 = thulac.thulac(seg_only=False, model_path="./models", rm_space=True)
lines = open(vocab_path).readlines()
fout = open(save_path, 'w')
t0 = time.time()
for idx, line in enumerate(lines[:100000]):
    word = line.split(' ')[0]
    tup = thu1.cut(word, text=False)
    if len(tup) == 0:
        continue
    if 'n' in tup[0][1] or 'v' in tup[0][1]:
        fout.write('%s' %line)
    if np.mod(idx, 1000) == 0:
        print '# %d, t = %ds' %(idx, time.time() - t0)
fout.close()
