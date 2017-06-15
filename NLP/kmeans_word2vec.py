# kMeans for word2vec vectors.
import cPickle as pk
from sklearn.cluster import MiniBatchKMeans
import sklearn.externals.joblib as joblib
import time
import numpy as np
import h5py

def create_data_from_dict(dic, words): 

    N = len(words)
    DIM = len(dic[words[0]])
    X = np.zeros((N, DIM), dtype=np.float32)
    for idx, word in enumerate(words):
        X[idx, :] = dic[word]
    return X

if __name__ == '__main__':
    vocab_path = '/home/wwt/THULAC-Python/data/vocabulary_10w.txt'
    root = '/media/wwt/860G/data/souhu_data/fusai/train/'
    word2vec_path = root + 'ourword2vec.pkl'
    nCluster = 1000
    save_path = root + 'kmeans_docVec_%d.model' %nCluster
    t0 = time.time()
    model = MiniBatchKMeans(n_clusters=nCluster, init='k-means++', max_iter=200, 
                           batch_size= 100000, 
                           verbose=1, 
                           compute_labels=True, 
                           random_state=None, 
                           tol=0.0, 
                           max_no_improvement=20, 
                           init_size=None, 
                           n_init=3, 
                           reassignment_ratio=0.01)
    #word2vec = pk.load(open(word2vec_path, 'r'))
    #words = [line.split(' ')[0] for line in open(vocab_path).readlines()]
    #X = create_data_from_dict(word2vec, words)
    h5file = h5py.File(root + 'train_tfidf_word2vec_cluster.h5')
    X = h5file['feature'][...]
    print 'Start training. Data shape ', X.shape
    model.fit(X)
    t = time.time() - t0
    print 'training done. t = %d minutes' %(t/60)
    joblib.dump(model, save_path)
    h5file.close()
