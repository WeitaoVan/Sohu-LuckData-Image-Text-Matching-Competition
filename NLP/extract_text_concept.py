# extract concept words in text using sparse TF-IDF (i.e. (word, tfidfValue))
import cPickle as pk
import numpy as np
import time


def topK_concept(tfidf, K, id2word=None):
    sort_tfidf = []
    for doc in tfidf:
        sort_doc = sorted(doc, key=lambda v:v[1], reverse=True)[:K]
        if id2word is not None:
            str_doc = []
            for i in range(len(sort_doc)):
                str_doc.append((id2word[sort_doc[i][0]], sort_doc[i][1]))
            sort_tfidf.append(str_doc)
        else:
            sort_tfidf.append(sort_doc)
    return sort_tfidf

def save_to_txt(tfidf, save_path):
    fout = open(save_path, 'w')
    concept_vocab = {}
    for doc in tfidf:
        string = ""
        for tup in doc:
            string += (tup[0] + ' ')
            try:
                concept_vocab[tup[0]] += 1
            except:
                concept_vocab[tup[0]] = 1
        fout.write('%s\n' %string)
    fout.close()
    sort_concept = sorted(concept_vocab, key=lambda v:v[1], reverse=True)
    fout = open('/home/wwt/THULAC-Python/data/train_concept_vocab.txt', 'w')
    for tup in sort_concept:
        fout.write('%s %d\n' %(tup[0], tup[1]))
    fout.close()

if __name__ == '__main__':
    K = 5
    id2word_path = '/home/wwt/THULAC-Python/data/id2word.pkl'
    tfidf_path = '/media/wwt/860G/data/formalCompetition4/train/train_tfidf.pkl'
    save_path = '/home/wwt/THULAC-Python/data/train_concept.txt'
    id2word = pk.load(open(id2word_path))
    tfidf = pk.load(open(tfidf_path))
    print 'loaded'
    sort_tfidf = topK_concept(tfidf, K, id2word=id2word)
    save_to_txt(sort_tfidf, save_path)
