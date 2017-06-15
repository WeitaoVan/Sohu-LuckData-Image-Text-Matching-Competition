# compute class distribution of word2vec vectors based on kMeans model
# input:
#   - GMM model. Specifically, mean ,variance and weights of 
#     K gaussian distributions in GMM.
#   - word2vec vectors.
# output:
#   - [dict] word2vec_distr. key: word, value: distribution
#     vector of shape (K,)
import numpy as np
import sklearn.externals.joblib as joblib
import cPickle as pk


    
if __name__ == '__main__':   
    root = '/media/wwt/860G/data/souhu_data/fusai/train/'
    model_path = root + 'kmeans_word2vec_512.model'
    vocab_list = '/home/wwt/THULAC-Python/data/vocabulary_10w.txt'
    save_path = root + 'word2vec_distr_10w.pkl'

    # load
    model = joblib.load(model_path)
    labels = model.labels_
    K = max(labels) + 1
    # compute
    word2vec_distr = {}
    for idx, line in enumerate(open(vocab_list).readlines()):
        word = line.split(' ')[0]
        try:
            word2vec_distr[word] = np.zeros((K,), dtype=np.float32)
            word2vec_distr[word][labels[idx]] = 1
        except:
            word2vec_distr[word] = 1.0 / K * np.ones((K,), dtype=np.float32)
            print 'unknown word %s' %word
        if np.mod(idx, 5000) == 0:
            print '# %d' %idx
    pk.dump(word2vec_distr, open(save_path, 'w'))
    print 'saved to %s' %save_path