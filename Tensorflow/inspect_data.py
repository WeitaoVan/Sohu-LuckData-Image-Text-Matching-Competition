import h5py
import numpy as np
root = '/media/wwt/860G/data/formalCompetition4/'
filename = root + 'val_img_feat.h5'
h5file = h5py.File(filename, 'r')
data = np.array(h5file['feature'])
sort_data = np.array(sorted(data, key=lambda v:v[0], reverse=True))
print sort_data.shape
for i in range(5):
    print data[i, :10]
    print np.var(data[:, i])
    print ' '