#
import cPickle as pk
import numpy as np

N = 97595
IDF = np.load('/media/wwt/860G/data/formalCompetition4/train_IDF.npy')
words = [line.split(' ')[0] for line in open('/home/wwt/THULAC-Python/data/vocabulary_1w2.txt')]
save_file = '/media/wwt/860G/data/formalCompetition4/train_DF_dict.npy'

DF = {}
for idx,word in enumerate(words):
    DF[word] = int(np.round(1.0*N / pow(10, IDF[idx])))
    
pk.dump(DF, open(save_file, 'w'))