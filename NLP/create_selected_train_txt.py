# 
import numpy as np
import cPickle as pk
import os
root = '/media/wwt/860G/data/souhu_data/fusai/'
save_path = root + 'train_classification_list.txt'
MAX_LEN = 20000
class_list = open(root + 'doc_classification.txt').readlines()
fout = open(save_path, 'w')
labels = []
for idx, folder in enumerate(class_list):
    folder = folder.strip()
    dirs = os.listdir(root + 'train_seg/' + folder)
    for filename in dirs[:min(MAX_LEN, len(dirs))]:
        fout.write('none.jpg      %s\n' %(folder + '/' + filename))
        labels.append(idx)
fout.close()
labels = np.array(labels, dtype=np.uint8)
np.save(root+'doc_class_label.npy', labels)
    