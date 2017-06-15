#-*- coding: utf-8 -*-
# warper to run 'SH_split.py'
import sys
import SH_split
import os
import argparse
reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='from training data corpus, segment docs and create vocabulary')
# i = 0, 1, ..., nProcess-1. This specifies which part of dataset to process (for large dataset)
parser.add_argument('i', type=int, help='index for one of the nProcess parts')
args = parser.parse_args()

nProcess = 1
i = args.i 
truncN = 2000 # truncate 'truncN' chars in each doc
mode = 1 # 1: ordinary, 2: also store TF and DF

vocab_path = '/home/wwt/THULAC-Python/data/_vocabulary' # write vocab to .txt and .pkl files
r = '/media/wwt/860G/data/souhu_data/fusai/'
#root = r + '2016News/其他/' # documents folder
root = r + 'test/text/'
match_list = r + 'testDummyMatching.txt'#'trainMatching_shuffle.txt'
seg_root = r + 'test_seg/'#'train_seg/' # save segmented documents
TF_DF_prefix = '/home/wwt/THULAC-Python/data/'
SPLIT = ' '
print 'i=%d, N=%d, mode %d' %(i, nProcess, mode)
print 'len(SPLIT) = %d' %len(SPLIT)
assert i >= 0
assert i < nProcess

lines = open(match_list).readlines()
N = len(lines)
nEach = int(N / nProcess)
if i < nProcess - 1:
    interval = range(i * nEach, (i + 1) * nEach)
else:
    interval = range(i * nEach, N)
if mode == 1:
# ordinary segmenting: read word -> add to dict
    SH_split.segment(root, match_list, seg_root, vocab_path + str(i), truncN=truncN, interval=interval, SPLIT=SPLIT)
elif mode == 2:
# advanced segmenting: beyond ordinary, also store TF and DF
    SH_split.segment_tfidf(root, match_list, seg_root, TF_DF_prefix+str(i), 
                           truncN=truncN, interval=interval, seg_only=True, SPLIT=SPLIT)


