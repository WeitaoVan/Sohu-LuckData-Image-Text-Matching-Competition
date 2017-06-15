#
import cPickle as pk
import numpy as np

def similarity_word(word, word2vec):
    s = {}
    queryVec = word2vec[word]
    for w, vec in word2vec.items():
        s[w] = np.dot(vec, queryVec) / (np.linalg.norm(vec) * np.linalg.norm(queryVec))
    return s

def find_similar_words(word, word2vec, K=20):
    s = similarity_word(word, word2vec)
    sort_s = sorted(s.iteritems(), key=lambda v:v[1], reverse=True)
    for i in range(K):
        print '%s: %f' %(sort_s[i][0], sort_s[i][1])

if __name__ == '__main__':
    
    word2vec_path = '/media/wwt/860G/data/souhu_data/fusai/train/ourword2vec.pkl'
    word2vec = pk.load(open(word2vec_path))
    find_similar_words('美国', word2vec, K=20)
    a = 0