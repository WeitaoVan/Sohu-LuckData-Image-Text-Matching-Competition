# based on the word2vec cluster, randomly choose a subset of 
# the clusters.
import cPickle as pk
import numpy as np


def create_one_hot_labels(words, word2vec_distr):
    N = len(words)
    K = word2vec_distr.values()[0].shape[0]
    labelMat = np.zeros((N, K), dtype=int)
    for n in range(N):
        try:
            labelMat[n, :] = word2vec_distr[words[n]]
        except:
            print 'unknown word in word2vec_distri: %s' %words[n]
    return labelMat, K

def words_in_clusters(labelMat, nChoose):
    K = labelMat.shape[1]
    randIdx = np.random.permutation(K)[:nChoose]
    flag = labelMat[:, randIdx].sum(axis=1)
    wordIdx = flag.nonzero()[0]
    return wordIdx

def save_choose_word_txt(words, wordIdx, save_path):
    fout = open(save_path)
    fout.write('%d words\n' %len(wordIdx))
    for idx in wordIdx:
        fout.write('%s\n' %words[idx])
    fout.close()

if __name__ == '__main__':
    root = '/media/wwt/860G/data/souhu_data/fusai/train/'
    word2vec_distr_path = root + 'word2vec_distr_10w.pkl'
    vocab_path = '/home/wwt/THULAC-Python/data/vocabulary_nv_4w.txt'
    save_path = root + 'wordRandIdx/rand2'
    nChoose = 128
    
    # create label matrix (each row is one-hot)
    word2vec_distr = pk.load(open(word2vec_distr_path, 'r'))
    words = [line.split(' ')[0] for line in open(vocab_path).readlines()]
    labelMat, K = create_one_hot_labels(words, word2vec_distr)
    assert nChoose <= K
    # random choose
    wordIdx = words_in_clusters(labelMat, nChoose)
    # save
    np.save(save_path+'.npy', wordIdx)
    save_choose_word_txt(words, wordIdx, save_path+'.txt')
        