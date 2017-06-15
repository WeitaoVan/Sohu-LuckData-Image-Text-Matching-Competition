# compute class distribution for texts based on
# word2vec cluster distributions
# use the top-truncN tf-idf words to represent the 
# document
import numpy as np
import h5py
import cPickle as pk
import time


def compute_doc_distr(words, K, MAX_LEN, word2vec_distr):
    # method: mean of all the vector distributions in doc
    distr = np.zeros((K+1,), dtype=np.float32)
    nGet = 0
    for word in words:
        try:
            distr[:K] += word2vec_distr[word]
            nGet += 1
        except:
            print 'unknown word %s' %word
        if nGet >= MAX_LEN:
            break
    ## mean distribution        
    if distr.sum() == 0:
        distr[-1] = 1
        return distr
    else:
        return distr / nGet
    ## multi-hot
    #return (distr > 0).astype(int)
    ## majority one-hot
    #idx = np.argmax(distr)
    #distr = np.zeros((K,), dtype=np.float32)
    #distr[idx] = 1

if __name__ == '__main__':
    mode = 'train'
    # truncate because we use truncated docs training LSTM
    truncN = 10
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    text_path = root + '%s_txt/%s_keyword.txt' %(mode,mode)
    word2vec_distr_path = root + '%s/word2vec_distr_10w.pkl' %mode
    save_path = root + '%s/%s_distr_T%d.h5' %(mode, mode, truncN)
    
    t0 = time.time()
    print 'start. mode:%s, truncN:%d. text:%s' %(mode, truncN, text_path)
    # load word distribution
    word2vec_distr = pk.load(open(word2vec_distr_path, 'r'))
    # class numbers
    K = word2vec_distr.items()[0][1].shape[0]
    docs = open(text_path).readlines()
    N = len(docs)
    h5file = h5py.File(save_path, 'w')
    h5set = h5file.create_dataset('label', shape=(N, K+1), dtype=np.float32)    
    for idx, doc in enumerate(docs):
        words = doc.strip().split(' ')
        h5set[idx, :] = compute_doc_distr(words, K, truncN, word2vec_distr)
        if np.mod(idx, 10000) == 0:
            t = time.time() - t0
            print '# %d. t = %f minutes' %(idx, t/60)
    h5file.close()
    print 'saved to %s' %save_path