# create dummy match list for val/test image-text
import os

if __name__ == '__main__':
    #root = '/media/wwt/860G/data/formalCompetition4/'
    root = '/media/wwt/860G/data/souhu_data/fusai/test/'
    img_root = root + 'image' #'News_pic_info_validate/'
    text_root = root + 'text' #'News_info_validate/'
    save_path = '/media/wwt/860G/data/souhu_data/fusai/testDummyMatching.txt'
    img_dirs = os.listdir(img_root)
    text_dirs = os.listdir(text_root)
    assert len(img_dirs) == len(text_dirs)
    text_dirs = sorted(text_dirs, key=lambda v:int(v[:-4]))
    fout = open(save_path, 'w')
    for idx,imgname in enumerate(img_dirs):
        fout.write('%s %s\n' %(imgname, text_dirs[idx]))
    fout.close()
    