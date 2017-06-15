# For Souhu competetion
# tf-idf weighted word2vec
# run 'compute_df.py' first
# Coder: Weitao Wan
import sys
import thulac
import os
import numpy as np
import time
import cPickle as pk
import h5py
import gensim
reload(sys)
sys.setdefaultencoding('utf-8')


def tfidf2wordvec(tfidf, id2word, model):
    N = len(tfidf)
    feat = np.zeros((N, 300), dtype=np.float32)
    for i, doc in enumerate(tfidf):
        vec = np.zeros((N, 1), dype=np.float32)
        for tup in doc:
            wordId = tup[0]
            value = tup[1]
            vec += value * model.word_vec(id2word[wordId], use_norm=True)
        feat[i, :] = vec
    return feat

def update_vocab(vocab, words, exception='.'):
    # update vocabulary
    count = 0
    for word in words:
        if word == exception:
            break
        if len(word) == 0:
            continue
        try:
            vocab[word] += 1
        except:
            vocab[word] = 1
        count += 1
    return count

def save_vocab_txt(vocab, file_path):
    # sort and write vocabulary.txt
    vocab = sorted(vocab.iteritems(), key=lambda v:v[1], reverse=True)          
    fout = open(file_path, mode='w')
    for item in vocab:
        fout.write('%s %d\n' %(item[0], item[1])) # word(space)count
    fout.close()    
    return

def word2vec(model, word):
    #return  model.word_vec(word.decode('utf-8'), use_norm=True)
    return model[word]

def compute_word2vec(docs, DF, nDoc, model, vecDim=300, MAX_LEN=10):
    N = len(docs)
    nonExist_vocab = {}
    feat = np.zeros((N, vecDim), dtype=np.float32)
    for idx, doc in enumerate(docs):
        nonExist_list = []
        TF = {}
        words = doc.strip().split(' ')
        nWord = len(words)
        update_vocab(TF, words)
        vec = np.zeros(vecDim, dtype=np.float32)
        nGet = 0
        for word in words:
            try:
                tfidf = 1.0 * TF[word] / nWord * np.log2(1.0 * nDoc / DF[word])
                vec += tfidf * word2vec(model, word)
                nGet += 1
            except:
                #nonExist_list.append(word)
                pass
            if nGet >= MAX_LEN:
                break
        feat[idx, :] = vec / (np.linalg.norm(vec) + 1e-10)
        #update_vocab(nonExist_vocab, nonExist_list)
        if np.mod(idx, 10000) == 0:
            print '# %d' %idx
            print 'nonExist: %d' %len(nonExist_vocab.keys())
    return feat, nonExist_vocab

def concat_word2vec(docs, DF, nDoc, model, vecDim=300, concat_len=10):
    N = len(docs)
    nonExist_vocab = {}
    feat = np.zeros((N, vecDim*concat_len), dtype=np.float32)
    for idx, doc in enumerate(docs):
        nonExist_list = []
        TFIDF = {}
        spt = doc.split(' ')
        nWord = update_vocab(TFIDF, spt)
        for word, tf in TFIDF.items():
            try:
                TFIDF[word] = 1.0 * tf / nWord * np.log2(1.0 * nDoc / DF[word])
            except:
                #nonExist_list.append(word)
                pass
        # sort
        tup = sorted(TFIDF.iteritems(), key=lambda v:v[1], reverse=True)
        # concat
        vec = np.zeros(vecDim*concat_len, dtype=np.float32)
        idx_get = 0
        idx_search = 0
        while(idx_get < concat_len and idx_search < len(tup)):
            try:
                vec[idx_get*vecDim : (idx_get+1)*vecDim] = word2vec(model, tup[idx_search][0])
                idx_get += 1
            except:
                pass
            idx_search += 1
        # assign
        feat[idx, :] = vec
        #update_vocab(nonExist_vocab, nonExist_list)
        if np.mod(idx, 10000) == 0:
            print '# %d' %idx
            #print 'nonExist: %d' %len(nonExist_vocab.keys())
    return feat, nonExist_vocab    

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path
        
def compute_tfidf_sorted_word2vec(data_txt_path, df_path, nDoc, word2vec_model_path, save_path):
    # compute tfidf, then use it to sort the words and create word2vec vectors
    thu1 = thulac.thulac(seg_only=True, model_path="./models")
    t0 = time.time()
    if word2vec_model_path[-3:] == 'bin':
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True, unicode_errors='ignore')
        model.init_sims(replace=True)
    elif word2vec_model_path[-3:] == 'pkl':
        model = pk.load(open(word2vec_model_path))
    else:
        raise TypeError('Unknown type for word2vec_model_path: %s' %word2vec_model_path)
        
    
    docs = open(data_txt_path).readlines()
    # compute DF
    DF = pk.load(open(df_path))
    feat, nonExist_vocab = compute_word2vec(docs, DF, nDoc, model, vecDim=300)
    #feat, nonExist_vocab = concat_word2vec(docs, DF, nDoc, model, vecDim=300)
    h5file = h5py.File(save_path, 'w') #''train_tfidf_weighted_word2vec_fasttext.h5'
    h5file.create_dataset('feature', dtype=np.float32, data=feat)
    h5file.close()
    #save_vocab_txt(nonExist_vocab, save_root+'nonExist_vocab.txt')
    print 'TF-IDF weighted word2vec saved to %s' %save_path
    print 'nonExist_vocab length: %d' %len(nonExist_vocab.keys())

def create_vec_from_words(words, word2vec, dim, length, MAX_MEAN=20):
    assert length > 0
    vec = np.zeros((dim * length,), dtype=np.float32)
    nGet = 0
    if length == 1:
        # mean
        for word in words:
            try:
                vec += word2vec[word]
                nGet += 1
            except:
                pass
            if nGet >= MAX_MEAN:
                break
        return vec / (nGet + 1e-10)
    else:
        # concate
        for word in words:
            try:
                vec[nGet * dim : (nGet + 1) * dim] = word2vec[word]
                nGet += 1
            except:
                pass
            if nGet >= length:
                break
        return vec / (np.sqrt(nGet) + 1e-10) # normalize

def doc2word2vec(data_txt_path, word2vec_model, save_path, dim=300, length=10):
    # do not use tf-idf values as coefficients.
    # usually because the data_txt_path is a tfidf-sorted text.
    # length = 1: mean of vectors
    # length > 1: concate vectors
    word2vec = pk.load(open(word2vec_model, 'r'))
    docs = open(data_txt_path).readlines()
    N = len(docs)
    feat = np.zeros((N, dim * length), dtype=np.float32)
    t0 = time.time()
    for idx, doc in enumerate(docs):
        words = doc.strip().split(' ')
        feat[idx, :] = create_vec_from_words(words, word2vec, dim, length)
        if np.mod(idx, 10000) == 0:
            t = time.time() - t0
            print '# %d, t = %d minutes' %(idx, t/60)
    h5file = h5py.File(save_path, 'w')
    h5file.create_dataset('feature', data=feat, dtype=np.float32)
    h5file.close()
    print 'saved to %s' %save_path

def compute_tfidf_cluster_feat(words, DF, nDoc, word2vec_distr):
    DIM = word2vec_distr.values()[0].shape[0]
    feat = np.zeros((DIM,), dtype=np.float32)
    TF = {}
    nWord = update_vocab(TF, words)
    for idx, word in enumerate(words):
        try:
            tfidf = 1.0 * TF[word] / nWord * np.log2(1.0 * nDoc / DF[word])
            feat += tfidf * word2vec_distr[word]
        except:
            pass
    return feat

def tfidf_cluster_feature(data_txt_path, word2vec_distr_path, save_path, df_path, nDoc):
    word2vec_distr = pk.load(open(word2vec_distr_path))
    docs = open(data_txt_path).readlines()
    DF = pk.load(open(df_path))
    N = len(docs)
    DIM = word2vec_distr.values()[0].shape[0]
    h5file = h5py.File(save_path, 'w')
    feat = h5file.create_dataset('feature', shape=(N, DIM), dtype=np.float32)
    t0 = time.time()
    for idx, doc in enumerate(docs):
        words = doc.strip().split(' ')
        feat[idx, :] = compute_tfidf_cluster_feat(words, DF, nDoc, word2vec_distr)
        if np.mod(idx, 10000) == 0:
            t = time.time() - t0
            print '#%d, t = %d mins' %(idx, t/60)
    h5file.close()
    print 'saved to %s' %save_path
    
    
if __name__ == '__main__':
    
    mode = 'test'
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    data_txt_path = root + '%s_txt/%s.txt' %(mode, mode)
    
    save_path = root + '%s/' %mode
    make_if_not_exist(save_path)
    save_path += '%s_tfidf_word2vec_cluster.h5' %mode
    
    # for computing tf-idf
    nDoc = 1249600
    df_path = root + 'train/DF_dict.pkl'    # from fusai train data
    
    word2vec_model = root + 'train/ourword2vec.pkl'
    
    #compute_tfidf_sorted_word2vec(data_txt_path, df_path, nDoc, word2vec_model, save_path)
    #doc2word2vec(data_txt_path, word2vec_model, save_path, dim=300, length=10)
    
    word2vec_distr_path = root + 'train/word2vec_distr_10w.pkl'
    tfidf_cluster_feature(data_txt_path, word2vec_distr_path, save_path, df_path, nDoc)
    
    


