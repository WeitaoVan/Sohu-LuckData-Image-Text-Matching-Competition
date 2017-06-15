# shuffle list
import numpy as np
import random
file_path = '/media/wwt/860G/data/souhu_data/fusai/trainMatching_filter.txt';
file_save = '/media/wwt/860G/data/souhu_data/fusai/trainMatching_shuffle_filter.txt';
lines = open(file_path).readlines()
N = len(lines)
idx = np.random.permutation(N)
fout = open(file_save, 'w')
for i in range(N):
    fout.write('%s' %lines[idx[i]])
    if np.mod(i, 10000) == 0:
        print '# %d' %i
fout.close()
