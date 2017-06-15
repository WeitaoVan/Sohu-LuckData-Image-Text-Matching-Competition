#
import h5py
import numpy as np

text_path = '/media/wwt/860G/data/souhu_data/fusai/train_txt/train_cont.txt'
docs = open(text_path).readlines()
h5file = h5py.File('/media/wwt/860G/data/souhu_data/fusai/train/train_tfidf_word2vec_cluster.h5')
dataset = 'feature'
label = np.array(h5file[dataset], dtype=np.float32)
K = 513
maxDoc = 10
for k in range(K):
    flags = label[:, k] == 1
    count = 0
    print '\nlabel = %d' %k
    for idx, flag in enumerate(flags):
        if flag:
            print docs[idx].decode('utf-8')[:80]
            count += 1
        if count >= maxDoc:
            break