import numpy as np
import h5py
import time

def test_match(image_embed_path, sentence_embed_path, pair_list, result_list):
    '''
       For each text, find its top 10 matching images.
       result_list format:
       (Repeat line:)
       textName, imgName1, imgName2, ..., imgName10
    '''
    CUT = 19997
    # read
    h5file = h5py.File(image_embed_path, 'r')
    image_embed = np.array(h5file['embed'])[:CUT]
    h5file.close()
    h5file = h5py.File(sentence_embed_path, 'r')
    sentence_embed = np.array(h5file['embed'])[:CUT]
    h5file.close()
    assert image_embed.shape[0] == sentence_embed.shape[0]
    lines = open(pair_list).readlines()[:CUT]
    image_files = [line.strip().split()[0] for line in lines]
    sentences = [line.strip().split()[1] for line in lines]
    # compute similarity by inner product
    # result S: row for image, column for text
    t0 = time.time()
    S = np.matmul(image_embed, sentence_embed.T)
    t = time.time() - t0
    print 'similarity computed, time cost %ds' %t
    # sorting
    t0 = time.time()
    rank = np.argsort(-S, axis=0) # desending order
    t = time.time() - t0
    print 'similarity sorted, time cost %ds' %t
    # write result
    t0 = time.time()
    N = S.shape[0]
    fout = open(result_list, 'w')
    for i in range(N):
        fout.write('%s' %(sentences[i].strip()))
        for k in range(10):
            picId = rank[k, i]
            fout.write(',%s' %image_files[picId])
        fout.write('\n')
    fout.close()
    similarity_path = '/media/wwt/860G/data/souhu_data/fusai/test/similarity/sim_unite_1w9.npy'
    np.save(similarity_path, S)
    print 'similarity matrix saved to\n %s' %similarity_path

def test_bagging(bagging_list, pair_list, result_list):
    print 'bagging'
    CUT = 19997
    bagging_files = open(bagging_list).readlines()
    S = 0.
    t0 = time.time()
    for idx, bagging_file in enumerate(bagging_files):
        S += np.load(bagging_file.strip())
        print 'read %d' %(idx+1)
    S = S/(idx + 1.)
    t = time.time() - t0
    print 'bagging %d results. %ds' %(idx + 1, t)
    # read image/text filename
    lines = open(pair_list).readlines()[:CUT]
    image_files = [line.strip().split()[0] for line in lines]
    sentences = [line.strip().split()[1] for line in lines]
    # sort
    rank = np.argsort(-S, axis=0) # desending order
    t = time.time() - t0
    print 'similarity sorted, time cost %ds' %t
    # write result
    N = S.shape[0]
    fout = open(result_list, 'w')
    for i in range(N):
        fout.write('%s' %(sentences[i].strip()))
        for k in range(10):
            picId = rank[0, i]
            fout.write(',%s' %image_files[picId])
        fout.write('\n')
    fout.close()    
    
if __name__ == '__main__':
    root = '/media/wwt/860G/data/souhu_data/fusai/test/'
    image_embed_path = root + 'embed/test_image_embed_unite_1w9.h5'
    sentence_embed_path = root + 'embed/test_text_embed_unite_1w9.h5'
    pair_list = root + 'testDummyMatching.txt'
    result_list = root + '/result/bagging_6tfidf_2rand.txt'
    #test_match(image_embed_path, sentence_embed_path, pair_list, result_list)
    
    bagging_list = root + 'similarity/bagging_list.txt'
    test_bagging(bagging_list, pair_list, result_list)
    # score = np.array([sum(S[:, v] > d[v]) for v in range(400)])
    
