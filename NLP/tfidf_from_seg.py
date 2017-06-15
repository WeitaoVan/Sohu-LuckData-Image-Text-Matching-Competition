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
import time
reload(sys)
sys.setdefaultencoding('utf-8')


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
    return  model.word_vec(word.decode('utf-8'), use_norm=True)
    #return model[word]

def compute_word2vec(docs, DF, nDoc, model, vecDim=300):
    N = len(docs)
    nonExist_vocab = {}
    feat = np.zeros((N, 300), dtype=np.float32)
    for idx, doc in enumerate(docs):
        nonExist_list = []
        TF = {}
        spt = doc.split(' ')
        nWord = len(spt)
        update_vocab(TF, spt)
        vec = np.zeros(vecDim, dtype=np.float32)
        for word, tf in TF.items():
            try:
                tfidf = 1.0 * tf / nWord * np.log2(1.0 * nDoc / DF[word])
                vec += tfidf * word2vec(model, word)
            except:
                nonExist_list.append(word)
                pass
        feat[idx, :] = vec
        update_vocab(nonExist_vocab, nonExist_list)
        if np.mod(idx, 10000) == 0:
            print '# %d' %idx
            print 'nonExist: %d' %len(nonExist_vocab.keys())
    return feat, nonExist_vocab

def compute_tfidf(doc, DIM, DF, nDoc, word2id):
    feat = np.zeros((DIM,), dtype=np.float32)
    TF = {}
    spt = doc.strip().split(' ')
    nWord = len(spt)
    update_vocab(TF, spt)
    for word, tf in TF.items():
        try:
            tfidf = 1.0 * tf / nWord * np.log2(1.0 * nDoc / DF[word])
            feat[word2id[word]] = tfidf
        except:
            print 'EXCEPTION: %s' %word
            pass
    return feat 

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path
        
def tfidf(data_txt_path, df_path, nDoc, word2id_path, save_path):
    t0 = time.time()
    docs = open(data_txt_path).readlines()
    word2id = pk.load(open(word2id_path, 'r'))
    N = len(docs)
    DIM = len(word2id.keys())
    h5file = h5py.File(save_path, 'w')
    h5set = h5file.create_dataset('feature', shape=(N, DIM), dtype=np.float32)
    print 'word2id loaded from %s' %word2id_path
    print 'dataset created, shape (%d, %d)' %(N, DIM)
    # load DF
    DF = pk.load(open(df_path))
    # compute tfidf
    for idx, doc in enumerate(docs):
        feat= compute_tfidf(doc, DIM, DF, nDoc, word2id)
        h5set[idx, :] = feat.copy()
        if np.mod(idx, 10000) ==0:
            t = time.time() - t0
            print '# %d, t = %f hours' %(idx, t / 3600.)
    h5file.close()
    print 'TF-IDF feature saved to %s' %save_path
    
if __name__ == '__main__':
    
    mode = 'train'
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    data_txt_path = root + '%s_txt/%s.txt' %(mode, mode)
    word2id_path = '/home/wwt/THULAC-Python/data/word2id_nv_4w.pkl'
    nDoc = 1249600
    save_path = root + '%s/' %mode
    make_if_not_exist(save_path)
    save_path += '%s_tfidf.h5' %mode
    
    df_path = root + 'train/DF_dict.pkl'    # from fusai train data
  
    tfidf(data_txt_path, df_path, nDoc, word2id_path, save_path)


