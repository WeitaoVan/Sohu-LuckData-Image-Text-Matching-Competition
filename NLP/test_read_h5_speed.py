#
import h5py
import time
h5_path = '/media/wwt/860G/data/souhu_data/fusai/train/train_GMM_distr_nv4w_T20.h5'
#h5_path = '/media/wwt/新加卷/data/souhu_data/fusai/train/' + 'train_LSTM_seq_20.h5'
h5file = h5py.File(h5_path, 'r')
print h5file.keys()
h5set = h5file['distribution']
t0 = time.time()
for i in range(h5set.shape[0]):
    feat = h5set[i, :]
    s = feat.sum()
    if not  s == 1.0:
        print 'i = %d, s = %f' %(i, s)
t = time.time() - t0
print 'time cost = %fs' %t