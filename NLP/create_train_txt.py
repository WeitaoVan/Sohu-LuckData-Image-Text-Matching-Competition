#-*- coding: utf-8 -*-
# create clean, segmented, fixed-length docs in one file for training
# run 'run_SH_split.py' to get the segmented docs and 
# run 'merge_vocab' to get the whole vocabulary
import os
import cPickle as pk
import time
import numpy as np

def filter_words(words, vocab, DIM, filler):
    TAIL = 16
    assert DIM > TAIL
    HEAD = DIM - TAIL
    count = 0
    doc = ""
    for word in words:
        try:
            cnt = vocab[word]
            doc += (word + ' ')
            count += 1
        except:
            pass
        if count >= HEAD:
            break
    # traverse from the tail
    count_tail = 0
    if TAIL > 0:
        tail_words = []
        for word in reversed(words):
            try:
                cnt = vocab[word]
                tail_words.append(word)
                count_tail += 1
            except:
                pass
            if count_tail >= TAIL:
                break      
        for word in reversed(tail_words):
            doc += (word + ' ')
    # fill if length < DIM
    for i in range(DIM - (count + count_tail)):
        doc += (filler + ' ')
    return doc

def create_train_txt(seg_root, file_list, interval, vocab_path, DIM, save_path, SPLIT='      '):
    '''
    seg_root = '/media/wwt/860G/data/formalCompetition4/train_seg/
    DIM = 256 # fixed word length of each doc for training
    vocab_path = '/home/wwt/THULAC-Python/data/vocabulary'
    file_list = '' # generate data according to its order
    save_path = '/media/wwt/860G/data/formalCompetition4/train.txt'
    '''
    filler = '.' # filler for length < DIM
    if vocab_path is None:
        vocab = None
    else:
        vocab = pk.load(open(vocab_path+'.pkl'))
    fout = open(save_path, 'w')
    lines = open(file_list).readlines()
    N = len(lines)
    t0 = time.time()
    if interval is not None:
        lines = [lines[i] for i in interval]
    for idx, line in enumerate(lines):
        filename = line.strip().split(SPLIT)[1]
        if vocab is None:
            # do not filter, just write
            doc = open(seg_root + filename).readlines()[0].strip()
        else:
            words = open(seg_root + filename).readlines()[0].strip().split(' ')
            doc = filter_words(words, vocab, DIM, filler)
        fout.write('%s\n' %doc)
        if np.mod(idx, 10000) == 0:
            print '%d/%d, t = %ds' %(idx, N, time.time() - t0)
    fout.close()
    
        
    