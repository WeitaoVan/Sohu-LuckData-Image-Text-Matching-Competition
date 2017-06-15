from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
import numpy as np
import h5py
import cv2
from sklearn.externals import joblib
from skimage.io import imread,imshow
#x=h5py.File('./train_img_feat_3crop_norm1.h5')
#img_f = np.array(x['feature'])
cls = joblib.load('./MiniBatchKmeans_20000_16c.model')
image_files = [line.strip().split('\t')[0] for line in open('./formalCompetition4/trainMatching_filter.txt').readlines()]
image_dir ='./formalCompetition4/News_pic_info_train/'
for i,v in enumerate(cls1.labels_):
    if v ==0:
        try:
            im=imread(image_dir+image_files[i])
            im=im[:,:,(2,1,0)]
            cv2.imshow(winname='label %d'%v, mat=im)
            cv2.waitKey(-1)
        except:
            pass
                     
#cls = KMeans(10,n_jobs=-4)
#cls.fit(img_f)
#joblib.dump(cls,'Kmeans.model')
#cls1 =MiniBatchKMeans(n_clusters=16, 
                     #batch_size=20000, n_init=10,max_iter=100)
#cls1.fit(img_f)
#joblib.dump(cls1,'MiniBatchKmeans_20000_16c.model')