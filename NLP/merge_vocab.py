# Merge vocabs
# For very large dataset, we split the whole dataset into several parts
# and generate vocabulary for each part and then merge them
import pickle as pk
import time
import numpy as np

def save_vocab_txt(vocab, file_path):
    # sort and write vocabulary.txt
    vocab = sorted(vocab.iteritems(), key=lambda v:v[1], reverse=True)          
    fout = open(file_path, mode='w')
    for item in vocab:
        fout.write('%s %f\n' %(item[0], item[1])) # word(space)value
    fout.close()    
    return

def merge_dict(host, guest):
    total_count = 0
    for word, cnt in guest.iteritems():
        try:
            host[word] += cnt
        except:
            host[word] = cnt
        total_count += cnt
    return total_count

def merge_vocab(N, vocab_path):
    vocab = pk.load(open(vocab_path + '0.pkl'))
    t0 = time.time()
    for i in range(1, N):
        v = pk.load(open(vocab_path + str(i) + '.pkl'))
        merge_dict(vocab, v)
        t = time.time() - t0
        print '%d / %d merged. time %fs' %(i+1, N, t)
    pk.dump(vocab, open(vocab_path+'.pkl', 'w'))
    save_vocab_txt(vocab, vocab_path+'.txt')

def merge_TFIDF(N, vocab_path, TF_DF_prefix, total_doc):
    t0 = time.time()
    TF = {}
    DF = {}
    total_TF = 1e-10
    for i in range(N):
        new_TF = pk.load(open(TF_DF_prefix+str(i)+'TF.pkl'))
        new_DF = pk.load(open(TF_DF_prefix+str(i)+'DF.pkl'))
        total_TF += merge_dict(TF, new_TF)
        merge_dict(DF, new_DF)
        t = time.time() - t0
        print '%d / %d merged. time %fs' %(i+1, N, t)  
    pk.dump(TF, open(TF_DF_prefix+'TF.pkl', 'w'))
    pk.dump(DF, open(TF_DF_prefix+'DF.pkl', 'w'))
    TFIDF = TF.copy()
    #for word, value in TFIDF.iteritems():
        #TFIDF[word] = TF[word] * 1.0 / total_TF * np.log2(total_doc*1.0/DF[word])
    save_vocab_txt(TFIDF, vocab_path+'_tfidf.txt')

if __name__ == '__main__':
    N = 5 # total process
    mode = 2
    vocab_path = '/home/wwt/THULAC-Python/data/vocabulary'
    if mode == 1:
        # mode 1: ordinary. simply merge N vocabularies into 1.
        merge_vocab(N, vocab_path)
    elif mode == 2:
        # mode 2: advanced. merge TF and DF dicts first, then create vocabulary sorted by TF-IDF
        TF_DF_prefix = '/home/wwt/THULAC-Python/data/'
        total_doc = 100000
        merge_TFIDF(N, vocab_path, TF_DF_prefix, total_doc)
    
    
    

    
