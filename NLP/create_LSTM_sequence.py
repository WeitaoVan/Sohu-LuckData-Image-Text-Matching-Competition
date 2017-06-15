
import numpy as np
import h5py
import cPickle as pk
import time


def create_word2vec(words, word2vec, T, DIM):
    # method: mean of all the vector distributions in doc
    feat = np.zeros((T, DIM), dtype=np.float32)
    t = 0
    for count, word in enumerate(words):
        try:
            feat[t, :] = word2vec[word]
            t += 1
        except:
            print 'unknown word %s' %word
        if t >= T:
            break        
    return feat

if __name__ == '__main__':
    mode = 'train'
    # truncate because we use truncated docs training LSTM
    T = 20
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    text_path = root + '%s_txt/%s_keyword.txt' %(mode, mode)
    word2vec_path = root + '%s/word2vec_11w.pkl' %mode
    save_path = '/media/wwt/新加卷/data/souhu_data/fusai/train/' + '%s_LSTM_seq_%d.h5' %(mode, T)
    
    
    print 'start. mode:%s, T:%d. text:%s' %(mode, T, text_path)
    # load 
    word2vec = pk.load(open(word2vec_path, 'r'))
    t0 = time.time()
    # class numbers
    DIM = word2vec.items()[0][1].shape[0]
    docs = open(text_path).readlines()
    N = len(docs)
    h5file = h5py.File(save_path, 'w')
    h5set = h5file.create_dataset('feature', shape=(N, T, DIM), dtype=np.float32)    
    for idx, doc in enumerate(docs):
        words = doc.strip().split(' ')
        h5set[idx, :, :] = create_word2vec(words, word2vec, T, DIM)
        if np.mod(idx, 10000) == 0:
            t = time.time() - t0
            print '# %d. t = %f minutes' %(idx, t/60)
    h5file.close()
    print 'saved to %s' %save_path