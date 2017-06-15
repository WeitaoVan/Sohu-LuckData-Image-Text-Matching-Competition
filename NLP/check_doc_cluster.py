#
import h5py
import numpy as np
from sklearn.externals import joblib

text_path = '/media/wwt/860G/data/souhu_data/fusai/train_txt/train_cont.txt'
docs = open(text_path).readlines()
maxDoc = 10
#h5file = h5py.File('/media/wwt/860G/data/souhu_data/fusai/train/train_topOneHot_T1.h5')
#label = np.array(h5file['label'], dtype=np.float32)
#K = 513
#for k in range(K):
    #flags = label[:, k] == 1
    #count = 0
    #print '\nlabel = %d' %k
    #for idx, flag in enumerate(flags):
        #if flag:
            #print docs[idx].decode('utf-8')[:80]
            #count += 1
        #if count >= maxDoc:
            #break
model = joblib.load('/media/wwt/860G/data/souhu_data/fusai/train/kmeans_docVec_1000.model')
labels = model.labels_
for k in range(max(labels) + 1):
    flags = (labels == k)
    print 'label %d' %k
    nPrint = 0
    for idx, flag in enumerate(flags):
        if flag:
            print docs[idx].decode('utf-8')[:80]
            nPrint += 1
        if nPrint >= maxDoc:
            break
        