#
import cPickle as pk

def compute_df(docs):
    DF = {}
    for idx, doc in enumerate(docs):
        spt = doc.split(' ')
        doc_vocab = {}
        for word in spt:
            try:
                _ = DF[word]
            except:
                DF[word] = int(0)
            try:
                _ = doc_vocab[word]
            except:
                doc_vocab[word] = 1
                DF[word] += 1
    return DF

if __name__ == '__main__':
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    data_txt_path = root + 'train_txt/train.txt'
    save_root = root + 'train/'
    docs = open(data_txt_path).readlines()
    DF = compute_df(docs)
    pk.dump(DF, open(save_root + 'DF_dict.pkl', 'w'))
