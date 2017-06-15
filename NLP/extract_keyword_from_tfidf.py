# 
import h5py 
import cPickle as pk
import numpy as np
import time
t0 = time.time()

tfidf_path = '/media/wwt/860G/data/souhu_data/fusai/test/test_tfidf.h5'
id2word_path = '/home/wwt/THULAC-Python/data/id2word_nv_4w.pkl'
save_path = '/media/wwt/860G/data/souhu_data/fusai/test/test_keyword.txt'
K = 256
is_fill = False

h5file = h5py.File(tfidf_path, 'r')
h5set = h5file['feature']
N = h5set.shape[0]
DIM = h5set.shape[1]
id2word = pk.load(open(id2word_path))
assert DIM == len(id2word.keys())

fout = open(save_path, 'w')
for i in range(N):
    tfidf = np.array(h5set[i, :], dtype=np.float32)
    idxs = np.argsort(-tfidf)
    nWord = tfidf.nonzero()[0].shape[0]
    k = 0
    n = 0
    while(k < K and n < nWord):
        word = id2word[idxs[n]]
        if '<' not in word:
            fout.write('%s ' %word)
            k += 1
        n += 1
    # filler
    if is_fill:
        for _ in range(K - k):
            fout.write('. ' %word)
    fout.write('\n')
    if np.mod(i, 10000) == 0:
        t = time.time() - t0
        print '# %d. t= %dminues' %(i, t/60)
fout.close()

