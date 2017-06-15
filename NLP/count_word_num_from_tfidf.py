# 
import h5py 
import cPickle as pk
import numpy as np

tfidf_path = '/media/wwt/860G/data/souhu_data/fusai/train/train_tfidf.h5'


h5file = h5py.File(tfidf_path, 'r')
h5set = h5file['feature']
N = h5set.shape[0]
DIM = h5set.shape[1]

num = []
for i in range(N):
    tfidf = np.array(h5set[i, :], dtype=np.float32)
    num.append(tfidf.nonzero()[0].shape[0])
    if np.mod(i, 10000) == 0:
        print '# %d' %i
h5file.close()

