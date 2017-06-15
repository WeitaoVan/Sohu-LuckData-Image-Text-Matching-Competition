# extract image features with VGG-16 model (pre-trained on ImageNet)
# saved in .h5 file with set name 'feature'
# Coder: Weitao Wan
import sys
sys.path.append('./python/')
import caffe
import os
os.environ['GLOG_minloglevel'] = '2' # suppress printing outcome
import h5py
import skimage.io
import skimage.transform as trans
import numpy as np
import time

def extract_feature_batch(net, filenames, batch_size, mode=1):
    output_blob = 'fc7';
    img_batch = get_img_batch(filenames, mode=mode)
    #print len(img_batch)
    #print len(img_batch[0])
    #print img_batch.shape
    net.blobs['data'].data[...] = img_batch
    output = net.forward()
    feature = np.array(output[output_blob], dtype=np.float32)
    feature = feature.reshape((batch_size, 3, 4096))
    return feature

def get_img_batch(filenames, mode=1):
    if type(mode) == type(int(1)):
	img_batch = np.array([get_image(filename, mode=mode) for filename in filenames])
    else:
	img_batch = []
	for filename in filenames:
	    for m in mode:
		img_batch.append(get_image(filename, mode=m))
	img_batch = np.array(img_batch)
    return img_batch
	    
def get_image(filename, mode=1):
    try:
        img = skimage.io.imread(filename)
	#print 'read max ', img.max()
    except:
        img = np.zeros((224, 224, 3), 'float')
	print 'ReadError: %s' %filename
    s = img.shape
    #print '%s shape:'%(filename) , s
    if len(s) == 4:
	img = img[0, ...]
	print 'len4 %s' %filename
    if len(s) == 2:
	img_rgb = np.zeros((s[0], s[1], 3), 'float')
	for i in range(3):
		img_rgb[..., i] = img
	img = img_rgb
	s = img.shape
	print 'len2 %s' %filename
    if s[0] == 1:
	img = np.zeros((224, 224, 3), 'float')
	print 'TooSmall: %s' %filename
    if s[2] == 4:
	img = img[:, :, :3]
    s = img.shape
    #img = img.astype(float)
    img_crop = crop_img(img, mode=mode)
    # resize
    #print 'mean before resize ', img_crop.mean()
    im_r = trans.resize(img_crop, (224, 224))
    # pre-process
    #print 'mean before whithening ', im_r.mean()
    im_r = (im_r - 0.5)/0.5
    im_r = im_r[:, :, ::-1] # RGB to BGR
    im_r = np.transpose(im_r, (2, 0, 1)) # HWC to CHW
    #print im_r.shape
    return im_r

def crop_img(img, mode=1):
    '''
    mode: 0(top-left), 1(center), 2(bottom-right)
    '''
    s = img.shape
    L = int(min(s[:2]))
    H = s[0]
    W = s[1]
    if mode == 0:
	img_crop = img[0:L, 0:L]
    elif mode == 1:
	h = int((H - L)/2.)
	w = int((W - L)/2.)	
	img_crop = img[h:L+h, w:L+w]
    elif mode == 2:
	img_crop = img[-L:, -L:]
    else:
	img_crop = img[0:L, 0:L]
	print 'Warning: invalid crop mode %d, return top-left crop' %mode
    return img_crop

if __name__ == '__main__': 
    batch_size = 50
    mode = [0, 1, 2] # 3 different crops for each image
    # model
    weights_path = './models/vgg_16layers/VGG_ILSVRC_16_layers.caffemodel'
    deploy_path = './models/vgg_16layers/vgg16_fc7_deploy.prototxt'
    # data
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    img_root = root + 'test/image/'
    match_list = root + 'testDummyMatching.txt'
    #match_list = '/media/wwt/860G/data/formalCompetition4/test/val_fusai_image.txt'
    #save
    output_file = root + 'test/test_img_feat_3crop_notAvg.h5'
    SPLIT = ' '
    
    caffe.set_mode_gpu()
    caffe.set_device(0) # gpu id
    net = caffe.Net(deploy_path, weights_path, caffe.TEST)
    features = []
    t0 = time.time()
    print 'start'
    lines = open(match_list).readlines()
    N = len(lines)
    maxiter = int(N / batch_size)
    assert np.mod(N, batch_size) == 0
    features = np.zeros((N, 3, 4096), dtype=np.float32)
    for i in range(maxiter):
	batch_lines = lines[i*batch_size : (i+1)*batch_size]
        filenames = [img_root + line.split(SPLIT)[0] for line in batch_lines]	
        feat_batch = extract_feature_batch(net, filenames, batch_size, mode=mode)
	# 3 crop
	#feat_mean3 = np.zeros((s[0]/3, s[1]), dtype=np.float32)
	#for k in range(batch_size):
	    #feat_mean3[k,...] = feat_batch[k*3:(k+1)*3, ...].mean(axis=0)
	#features.extend(feat_mean3.copy())    
        #features.extend(feat_batch.copy())
	features[i*batch_size:(i+1)*batch_size, :, :] = feat_batch
	print '%d\n' %(len(features))
        if np.mod(i+1, 10) == 0:
            t = time.time() - t0
            print ('Iter %d/%d, time %f hours' %(i+1, maxiter, t/3600.0))
    h5file = h5py.File(output_file, 'w')
    h5file.create_dataset('feature', data=features)
    h5file.close()
    print 'N=%d' %N 
        
