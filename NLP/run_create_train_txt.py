# run 'create_train_txt.py'
# after 'nProcess' txts are created, sh
# 'for name in $(ls); do cat $name >> train.txt;done'
# to merge them into one train.txt
from create_train_txt import create_train_txt


nProcess = 1
i = 0
assert i < nProcess
assert i >= 0

DIM = 256 # fixed word length of each doc for training
root = '/media/wwt/860G/data/souhu_data/fusai/'
#root = '/media/wwt/860G/data/formalCompetition4/'
seg_root = root + 'test_seg/'
vocab_path = '/home/wwt/THULAC-Python/data/vocabulary_10w' #nv_4w
file_list = root + 'testDummyMatching.txt' #'testDummyMatching.txt' # trainMatching_shuffle_filter# generate data according to its order
save_path = root + 'test_txt/test_cont'
SPLIT = ' '# '      ' # split string for match list
print 'len(SPLIT) = %d' %len(SPLIT)

lines = open(file_list).readlines()
N = len(lines)
nEach = int(N / nProcess)
if i < nProcess - 1:
    interval = range(i * nEach, (i + 1) * nEach)
else:
    # the last process finishes the rest job
    interval = range(i * nEach, N)
create_train_txt(seg_root, file_list, interval, vocab_path, DIM, save_path+str(i)+'.txt', SPLIT=SPLIT)