# merge 3-crop feat by averaging the 3 crops
import h5py
import numpy as np
root = '/media/wwt/860G/data/souhu_data/fusai/'
data_path = root + 'test/test_img_feat_3crop_notAvg.h5'
save_path = root + 'test/test_img_feat_3crop_mean.h5'
h5file = h5py.File(data_path, 'r')
h5out = h5py.File(save_path, 'w')
features = []
for dataset in h5file.keys():
    del features
    features = np.array(h5file[dataset], dtype=np.float32)
    s = features.shape
    features = np.mean(features, axis=1, dtype=np.float32)
    h5out.create_dataset(dataset, data=features)
    print 'dataset %s done' %dataset
h5out.close()
h5file.close()
    