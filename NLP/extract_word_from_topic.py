#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
topic_list = './data/lda_run2_topic.txt'
save_vocab = './data/vocab_topic.txt'
save_vocab2 = './data/vocab_topic_len2.txt'
vocab = {}
vocab2 = {}
for line in open(topic_list).readlines():
    spt = line.split('"')
    for k in range(10):
        word = spt[2*k+1]
        if word in vocab.keys():
            vocab[word] += 1
        else:
            vocab[word] = 1
        if word in vocab2.keys():
            vocab2[word] += 1
        elif len(word) >= 6:
            vocab2[word] = 1
fout = open(save_vocab, 'w')
fout2 = open(save_vocab2, 'w')
for word in vocab.keys():
    fout.write('%s\n' %(word))

for word in vocab2.keys():
    fout2.write('%s\n' %(word))

fout.close()
fout2.close()
