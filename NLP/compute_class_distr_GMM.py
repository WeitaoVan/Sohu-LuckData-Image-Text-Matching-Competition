# compute class distribution of word2vec vectors based on GMM model
# input:
#   - GMM model. Specifically, mean ,variance and weights of 
#     K gaussian distributions in GMM.
#   - word2vec vectors.
# output:
#   - [dict] word2vec_distr. key: word, value: distribution
#     vector of shape (K,)
import numpy as np
from scipy.stats import multivariate_normal as mnormal
import cPickle as pk

def create_mnormal_distr(mu, cov):
    '''
    mu, var are parameters of GMM
    '''
    K = mu.shape[0]
    mnormal_distrs = []
    for k in range(K):
        mnormal_distrs.append(mnormal(mean=mu[k, :], cov=cov[k, ...]))
    return mnormal_distrs  

def post_prob(x, mnormal_distri, w):
    '''
    use Bayesian rule to compute posterior prob of class|x
     -x : data vector
     -mnormal_dsitr [list]: multivariate normal distributions
     -w : mixing coefficents of GMM
    
    '''
    K = len(mnormal_distri)
    likeli_prob = np.zeros((K,))
    for k in range(K):
        likeli_prob[k] = mnormal_distri[k].pdf(x)
    try:
        post_p = np.zeros((K,))
        post_p[np.argmax(likeli_prob)] = 1.0
        #post_p = np.multiply(w, likeli_prob) / np.dot(w, likeli_prob)
    except:
        print 'exception'
    return post_p
    
if __name__ == '__main__':   
    root = '/media/wwt/860G/data/souhu_data/fusai/train/'
    GMM_model_path = root + 'gmm_1000.model'
    vocab_list = '/home/wwt/THULAC-Python/data/vocabulary_nv_4w.txt'
    word2vec_model_path = root + 'word2vec_11w.pkl'
    save_path = root + 'word2vec_distr_nv_4w.pkl'
    # load and create GMM
    gmm = pk.load(open(GMM_model_path))
    w = gmm[0]
    mu = gmm[1]
    cov = gmm[2]
    w.astype(np.float128)
    mu.astype(np.float128)
    cov.astype(np.float128)
    assert np.sum(w) == 1
    mnormal_distrs = create_mnormal_distr(mu, cov)
    K = len(mnormal_distrs)
    # load word2vec
    word2vec = pk.load(open(word2vec_model_path, 'r'))
    # compute
    word2vec_distr = {}
    for idx, line in enumerate(open(vocab_list).readlines()):
        word = line.split(' ')[0]
        try:
            vec = word2vec[word]
            word2vec_distr[word] = post_prob(vec, mnormal_distrs, w)
        except:
            word2vec_distr[word] = 1.0 / K * np.ones((K,), dtype=np.float32)
            print 'unknown word %s' %word
        if np.mod(idx, 5000) == 0:
            print '# %d' %idx
    pk.dump(word2vec_distr, open(save_path, 'w'))
    print 'saved to %s' %save_path