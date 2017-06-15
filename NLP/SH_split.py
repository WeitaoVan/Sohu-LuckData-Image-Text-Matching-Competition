#-*- coding: utf-8 -*-
# For Souhu competetion
# Chinese documents segmenting & word counting.
# write output:
# - vocabulary.txt
# - vocabulary.pkl
# - the segmented docs (.txt)
# Coder: Weitao Wan
import sys
import thulac
import os
import numpy as np
import time
import pickle as pk
reload(sys)
sys.setdefaultencoding('utf-8')


def seg_doc(filepath, truncN, model):
    # segment doc
    lines = open(filepath).readlines()
    lines = lines[0].split('\t')[1:]
    chars = "".join(lines).decode('utf-8')
    #chars = lines[0].decode('utf-8')
    chars = chars[:truncN] + chars[-64:]
    seg = model.cut(chars, text=False)  
    return seg

def update_vocab(vocab, words):
    # update vocabulary
    for word in words:
        try:
            vocab[word] += 1
        except:
            vocab[word] = 1
    return

def save_vocab_txt(vocab, file_path):
    # sort and write vocabulary.txt
    vocab = sorted(vocab.iteritems(), key=lambda v:v[1], reverse=True)          
    fout = open(file_path, mode='w')
    for item in vocab:
        fout.write('%s %d\n' %(item[0], item[1])) # word(space)count
    fout.close()    
    return

def write_doc(file_path, seg, seg_only):
    # write segmented doc
    fout = open(file_path, 'w')
    s = ""
    for item in seg:
        s += (item[0] + ' ')
    fout.write('%s\n' %s)
    fout.close()    
    
def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path

def process_TF_DF(doc_list, TF_DF_prefix):
    TF = {}
    DF = {}
    for word_dict in doc_list:
        update_vocab(DF, word_dict.keys())
        for word, cnt in word_dict.iteritems():
            try:
                TF[word] += cnt
            except:
                TF[word] = 1
    pk.dump(TF, open(TF_DF_prefix+'TF.pkl', 'w'))
    pk.dump(DF, open(TF_DF_prefix+'DF.pkl', 'w'))

def segment(root, match_list, seg_root, vocab_path, truncN=10000, interval=None, SPLIT='      '):
    '''
    truncN = 1000 # truncate 'truncN' chars in each doc
    root = '/media/wwt/860G/data/formalCompetition4/News_info_train/' # documents folder
    seg_root = '/media/wwt/860G/data/formalCompetition4/train_seg/' # save segmented documents
    vocab_path = '/home/wwt/THULAC-Python/data/vocabulary' # write vocab to .txt and .pkl files
    interval: if specified, process only those files specified by interval. e.g. interval=range(0,200000)
    '''
    print 'vocabulary save path: %s' %vocab_path
    if not os.path.exists(seg_root):
        os.makedirs(seg_root)
    thu1 = thulac.thulac(seg_only=True, model_path="./models") # open source tool for word segmenting
    read_lines = open(match_list).readlines()
    if interval is not None:
        lines = [read_lines[i] for i in interval]
    else:
        lines = read_lines
    N = len(lines)    
    vocab = {}
    start_t = time.time()
    for idx, line in enumerate(lines):
        filename = line.strip().split(SPLIT)[1]
        filepath = root + filename
        folder = filename.split('/')[0]
        #make_if_not_exist(seg_root + folder)         
        seg = seg_doc(filepath, truncN, thu1) # segment
        #spt = [tup[0] for tup in seg]
        #update_vocab(vocab, spt)
        write_doc(seg_root+filename, seg, True)
        if np.mod(idx, 100) == 0:
            print '%d/%d files done' %(idx+1, N)
            t = time.time() - start_t
            print '%f hours' %(t/3600.0)
    # sort and write vocabulary.txt
    #save_vocab_txt(vocab, vocab_path + '.txt')
    # write vocabulary.pkl
    #pk.dump(vocab, open(vocab_path+'.pkl', 'w'))
    
def segment_tfidf(root, match_list, seg_root, TF_DF_prefix, truncN=10000, interval=None, seg_only=True, SPLIT='      '):
    print 'TF, DF save prefix: %s' %TF_DF_prefix
    valid_type = ['n', 'ns', 'np', 'ni']
    if not os.path.exists(seg_root):
        os.makedirs(seg_root)
    thu1 = thulac.thulac(seg_only=seg_only, model_path="./models") # open source tool for word segmenting
    read_lines = open(match_list).readlines()
    if interval is not None:
        lines = [read_lines[i] for i in interval]
    else:
        lines = read_lines
    N = len(lines)
    start_t = time.time()
    doc_list = [] # storing dict for each doc
    for idx, line in enumerate(lines):
        filename = line.strip().split(SPLIT)[1]
        word_dict = {}
        filepath = root + filename
        folder = filename.split('/')[0]
        make_if_not_exist(seg_root + folder)        
        seg = seg_doc(filepath, truncN, thu1) # segment
        spt = [tup[0] for tup in seg] if seg_only else [tup[0] for tup in seg if 'n' in tup[1]]
        update_vocab(word_dict, spt)    
        write_doc(seg_root+filename, seg, seg_only)
        if np.mod(idx, 100) == 0:
            print '%d/%d files done' %(idx+1, N)
            t = time.time() - start_t
            print '%f hours' %(t/3600.0)
        doc_list.append(word_dict)
    process_TF_DF(doc_list, TF_DF_prefix)
