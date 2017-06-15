import tensorflow as tf
import numpy as np
import h5py
import time
import sys
import gensim
import os
import pickle as pkl
from sklearn.externals import joblib

def read_h5(file_path, dataset):
    h5file = h5py.File(file_path, 'r')
    data = h5file[dataset]
    return data
def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'folder %s created' %path

class BidirectionNet:
    def __init__(self,is_training=True,is_skip=False, batch_size= 100, is_TopKloss=True, 
                 word2vec_model='/media/wwt/860G/data/souhu_data/fusai/train/word2vec_11w.pkl'):
        # word2vec_model='/media/wwt/860G/model/word2vec/cn.cbow.bin'
        #self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True, unicode_errors='ignore')
        self.word2vec = pkl.load(open(word2vec_model,'r'))
        self.batch_size = batch_size
        self.weight_decay = 0.000001
        self.endpoint={}
        self.is_skip=is_skip
        self.is_TopKloss = is_TopKloss
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0
        self.build_input()
        #self.build_matchnet()
        #self.build_classify()
	#self.build_crossEnt_class()
	self.loss_weight = 0.
	self.build_unite(self.loss_weight)
        if is_training:
            #self.build_summary()
            #self.build_summary_crossEnt()
	    self.build_summary_unite()
    def build_input(self):
        # positive
        self.vector_size = 300
        self.sentence_len = 20
	self.num_classes = 1000
        self.softmax_labels = tf.placeholder(tf.int32, shape=[self.batch_size])
	#self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes])
	self.labels = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.raw_sentence= tf.placeholder(tf.float32, shape=[self.batch_size, self.sentence_len, self.vector_size],name='raw_sentence')
        self.sentence_emb =self.raw_sentence #tf.nn.embedding_lookup(tf.get_variable('word_embedding',[4096,512]),self.raw_sentence)
	self.docVec = tf.placeholder(tf.float32, shape=[self.batch_size, 512], name='doc_vector')
        if self.is_skip:
            self.image_feat = tf.placeholder(tf.float32,shape=[None,512], name='image_features')
        else:
            self.image_feat = tf.placeholder(tf.float32,shape=[self.batch_size, 4096], name='image_features')   
    def conv_layer(self, X, num_output, kernel_size, s, p='SAME'):
        return tf.contrib.layers.conv2d(X,num_output,kernel_size,s,\
                                        padding=p,weights_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),\
                                        normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':self.is_training,'updates_collections':None})
    def lstm(self, input_tensor, initial_embedding=None, num_units=1000, reuse=False):
        with tf.variable_scope('lstm', reuse=reuse) as scope:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
	    lstm_fc1 = tf.contrib.layers.fully_connected(input_tensor, num_units, scope='lstm_fc1')
	    #lstm_cell = tf.contrib.rnn.MultiRNNCell(
	    #    [tf.contrib.rnn.BasicLSTMCell(num_units) for _ in range(2)])
            #lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
	    input_list = tf.unstack(lstm_fc1, axis=1)
	    zero_state = lstm_cell.zero_state(
	        batch_size=input_tensor.get_shape()[0], dtype=tf.float32)
	    if initial_embedding is None:
		lstm1,_ = tf.contrib.rnn.static_rnn(lstm_cell, inputs=input_list, initial_state=zero_state)
	    else:
		_, initial_state = lstm_cell(initial_embedding, zero_state)
		lstm1,_ = tf.contrib.rnn.static_rnn(lstm_cell, inputs=input_list, initial_state=initial_state)
            lstm_output = lstm1
        self.endpoint['LSTM_output'] = lstm_output
        return lstm_output
    
    
    def lstm_classify(self, input_tensor, num_output, reuse=False):
        lstm_output_list = self.lstm(input_tensor, reuse=reuse)
        lstm_embed = tf.add_n(lstm_output_list) / len(lstm_output_list)
	#lstm_embed = tf.concat(lstm_output_list, axis=1)
	with tf.variable_scope('lstm_classify', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            #softmax_fc1 = tf.contrib.layers.fully_connected(lstm_embed, 4096, weights_regularizer=wd, scope='softmax_fc1')
	    #softmax_fc2 = tf.contrib.layers.fully_connected(softmax_fc1, 512, weights_regularizer=wd, scope='softmax_fc2')
            logits = tf.contrib.layers.fully_connected(lstm_embed, num_output, weights_regularizer=wd, scope='softmax_fc_logits',activation_fn=None)
            self.endpoint['softmax_logits'] = logits
        return logits, lstm_embed

    def sentencenet(self, input_tensor, reuse=False):
        with tf.variable_scope('sentence_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)

            sentence_fc1 = tf.contrib.layers.fully_connected(input_tensor, 2048, weights_regularizer=wd, scope='s_fc1')
            #drop_fc1 = tf.nn.dropout(sentence_fc1, self.keep_prob, name='drop_fc1')
            sentence_fc2 = tf.contrib.layers.fully_connected(sentence_fc1, 512,activation_fn=None, weights_regularizer=wd, scope='s_fc2')
            sentence_fc2_bn = tf.contrib.layers.batch_norm(sentence_fc2, center=True, scale=True, is_training=self.is_training,
                                                           reuse=reuse, decay=0.999, updates_collections=None, 
                                                           scope='s_fc2_bn')
            embed = sentence_fc2_bn/tf.norm(sentence_fc2_bn,axis= -1,keep_dims=True)
        self.endpoint['sentence_fc1'] = sentence_fc1
        self.endpoint['sentence_fc2'] = embed
        return embed
    
    
    def imagenet(self, image_feat, reuse=False,skip=False):
        if skip:
            return image_feat
        with tf.variable_scope('image_net', reuse=reuse) as scope:
            wd = tf.contrib.layers.l2_regularizer(self.weight_decay)
            image_fc1 = tf.contrib.layers.fully_connected(image_feat,2048, weights_regularizer=wd,scope='i_fc1')
            image_fc2 = tf.contrib.layers.fully_connected(image_fc1, 512, activation_fn=None, weights_regularizer=wd, scope='i_fc2')
	    image_fc2_bn = tf.contrib.layers.batch_norm(image_fc2, center=True, scale=True, is_training=self.is_training,
		                                                   reuse=reuse, decay=0.999, updates_collections=None, 
		                                                   scope='i_fc2_bn')
	    embed = image_fc2_bn/tf.norm(image_fc2_bn,axis= -1,keep_dims=True)
        self.endpoint['image_fc1'] = image_fc1
        self.endpoint['image_fc2'] = embed
        return embed
    
    def triplet_loss(self, common, pos, neg, margin=0.2):
        # d(common, pos) + margin < d(common, neg)
        self.d_pos = tf.reduce_sum(tf.squared_difference(common, pos),-1)
        self.d_neg =tf.reduce_sum(tf.squared_difference(common, neg),-1)
        return tf.reduce_sum(tf.nn.relu(self.d_pos + margin - self.d_neg, name = 'triplet_loss'))
    
    def crossEnt_loss(self, labels, logits):
	crossEnt_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits) #, label_smoothing=0.1
	#crossEnt_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
	#_labels = labels + 1e-8
	#crossEnt_loss += tf.reduce_mean(tf.reduce_sum(tf.multiply(_labels, tf.log(_labels)), axis=1))
	return crossEnt_loss
    
    def build_matchnet(self):
        self.sentence_fc2 = self.sentencenet(self.sentence_emb, reuse=False)
        self.image_fc2 = self.imagenet(self.image_feat, reuse=False,skip=self.is_skip)
        # compute loss
        if self.is_training:

            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.positiveloss=self.positive_loss(self.sentence_fc2,self.image_fc2)
            if not self.is_TopKloss:
                self.total_loss=tf.add_n([self.positive_loss(self.sentence_fc2,self.image_fc2)]+self.reg_loss)
            else:            
                self.total_loss =tf.add_n( list(self.top_K_loss(self.sentence_fc2,self.image_fc2))+self.reg_loss)
            self.saver = tf.train.Saver(max_to_keep=20)
     
    def build_classify(self):
        logits, embed = self.lstm_classify(self.raw_sentence, self.num_classes)
        if self.is_training:
            self.softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.softmax_labels ,logits=logits))
            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss=tf.add_n([self.softmax_loss]+self.reg_loss)   
        self.saver = tf.train.Saver(max_to_keep=30)
    
    def build_crossEnt_class(self):
	logits, embed = self.lstm_classify(self.raw_sentence, self.num_classes)
	if self.is_training:
	    self.crossEnt_loss = self.crossEnt_loss(self.labels, logits)
	    self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	    self.total_loss=tf.add_n([self.crossEnt_loss]+self.reg_loss)   
	self.saver = tf.train.Saver(max_to_keep=30)	
    
    def build_unite(self, loss_weight):
	lstm_output_list = self.lstm(self.raw_sentence, initial_embedding=self.docVec)
	lstm_embed = tf.add_n(lstm_output_list) / len(lstm_output_list)
	#lstm_embed = lstm_output_list[-1]
	self.sentence_fc2 = self.sentencenet(lstm_embed, reuse=False)
	self.image_fc2 = self.imagenet(self.image_feat, reuse=False,skip=self.is_skip)	
	if self.is_training:
	    #self.crossEnt_loss = self.crossEnt_loss(self.labels, logits) * loss_weight
	    self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	    if not self.is_TopKloss:
		self.total_loss=tf.add_n([self.positive_loss(self.sentence_fc2,self.image_fc2)]+self.reg_loss)
	    else:            
		self.total_loss =tf.add_n( list(self.top_K_loss(self.sentence_fc2, self.image_fc2)) + 
		                           self.reg_loss)
	    self.saver = tf.train.Saver(max_to_keep=20)	
    
    #def build_trainop(self,loss,lr=0.001, clipping_norm=10, optimizer =tf.train.AdamOptimizer, tvars=None,clip_vars=None):
        #if tvars is None:        
            #tvars = tf.trainable_variables()
        #if clip_vars is None:
            #clip_vars = tvars
        #g=tf.gradients(loss, tvars)
        #grads= [tf.clip_by_global_norm(v,clipping_norm) if v in clip_vars else v for v in g ]
        #opt = optimizer(lr)
        #for i,v in enumerate(tvars):
            #if grads[i] is not None:
                #tf.summary.histogram(name=v.name+'_gradient', values=grads[i])
        #return opt.apply_gradients(zip(grads,tvars))  
    def build_trainop(self, loss, lr=0.0001, clipping_norm=1e10, optimizer =tf.train.AdamOptimizer, t_vars = tf.trainable_variables()):     
	opt = optimizer(lr)
	gvs = opt.compute_gradients(loss, var_list=t_vars)
	clipped_gvs = [(tf.clip_by_norm(grad, clipping_norm), var) for grad,var in gvs]
	for grad,var in clipped_gvs:
	    tf.summary.histogram(var.name+'_gradient', grad)
	return opt.apply_gradients(clipped_gvs)       
           
    def build_summary(self):
        tf.summary.scalar('loss/reg_loss', tf.add_n(self.reg_loss))
        tf.summary.scalar('loss/total_loss', self.total_loss)
        if self.is_skip:
            tf.summary.histogram('activation/image_fc2',self.image_fc2)
        if self.is_TopKloss:
            tf.summary.scalar('msic/d_neg', tf.reduce_mean(self.d_neg))
            tf.summary.scalar('msic/d_pos', tf.reduce_mean(self.d_pos))        
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)

        t_var = tf.trainable_variables()
        watch_list = ['s_fc1', 's_fc2']
        if not self.is_skip:
            watch_list += ['i_fc1', 'i_fc2']        
        for watch_scope in watch_list:
            watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
            tf.summary.histogram('weights/'+watch_scope, watch_var[0])
            
    def build_summary_classify(self):
        tf.summary.scalar('loss/softmax_loss', self.softmax_loss)      
        for name, tensor in self.endpoint.items():
            tf.summary.histogram('activation/' + name, tensor)
        t_var = tf.trainable_variables()
        watch_list = ['softmax_fc1','softmax_fc2']     
        for watch_scope in watch_list:
            watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
	    if len(watch_var) > 0:
                tf.summary.histogram('weights/'+watch_scope, watch_var[0]) 
		
    def build_summary_crossEnt(self):
	tf.summary.scalar('loss/cross_entropy_loss', self.crossEnt_loss)      
	for name, tensor in self.endpoint.items():
	    tf.summary.histogram('activation/' + name, tensor)
	t_var = tf.trainable_variables()
	watch_list = []
	for watch_scope in watch_list:
	    watch_var = [var for var in t_var if watch_scope+'/weights' in var.name]
	    if len(watch_var) > 0:
		tf.summary.histogram('weights/'+watch_scope, watch_var[0])  
		
    def build_summary_unite(self):
	self.build_summary()
	#tf.summary.scalar('loss/cross_entropy_loss', self.crossEnt_loss)
        
    def positive_loss(self, sentence, image):
        image =tf.stack([image]*20,axis=1)
        diff = tf.reduce_sum(tf.squared_difference(sentence, image, name='positive_loss'),axis=-1) 
        diff = tf.reduce_min(diff,axis=-1)
        return tf.reduce_sum(diff)       
    def top_K_loss(self, sentence, image, K=50, margin=0.3):
	#  change: K=300, but i choose index 25 to 75 for training.
	#  so, the real 'K' is 50

       	sim_matrix = tf.matmul(sentence, image, transpose_b=True)
	s_square = tf.reduce_sum(tf.square(sentence), axis=1)
	im_square = tf.reduce_sum(tf.square(image), axis=1)
	d = tf.reshape(s_square,[-1,1]) - 2 * sim_matrix + tf.reshape(im_square, [1, -1])
	positive = tf.stack([tf.matrix_diag_part(d)] * K, axis=1)
	length = tf.shape(d)[-1]
	d = tf.matrix_set_diag(d, 8 * tf.ones([length]))
	sen_loss_K ,_ = tf.nn.top_k(-1.0 * d, K, sorted=False) # note: this is negative value
	im_loss_K,_ = tf.nn.top_k(tf.transpose(-1.0 * d), K, sorted=False) # note: this is negative value
	#sen_loss_K = sen_loss_K[:, 25:75]
	#im_loss_K = im_loss_K[:, 25:75]
	sentence_center_loss = tf.nn.relu(positive + sen_loss_K + margin)
	image_center_loss = tf.nn.relu(positive + im_loss_K + margin)
	self.d_neg = (sen_loss_K + im_loss_K)/-2.0
	self.d_pos = positive        
	self.endpoint['debug/im_distance_topK'] = -1.0 * im_loss_K
	self.endpoint['debug/sen_distance_topK'] = -1.0 * sen_loss_K 
	self.endpoint['debug/d_Matrix'] = d
	self.endpoint['debug/positive'] = positive
	self.endpoint['debug/s_center_loss'] = sentence_center_loss
	self.endpoint['debug/i_center_loss'] = image_center_loss
	self.endpoint['debug/S'] = sim_matrix
	self.endpoint['debug/sentence_square'] = s_square
	self.endpoint['debug/image_square'] = im_square
        return tf.reduce_sum(sentence_center_loss),tf.reduce_sum(image_center_loss)


    def train(self, sess, maxEpoch=300, lr=0.001,is_load=False,only_sentence=False,is_fixsentence=False, ckpt_path=''):
        logdir = './log/unite/run3'
        model_save_path='/media/wwt/860G/model/tf_souhu/unite3/'
	make_if_not_exist(model_save_path)
	model_save_path += 'ckpt'
	root = '/media/wwt/860G/data/souhu_data/fusai/'
	t_var = tf.trainable_variables()
	# read data
	nDataset = 1
	useDataset = 1
	REPEAT = 1
        sentence = np.array(open(root + 'train_txt/train.txt','r').readlines())
	#labels = np.array(read_h5(root + 'train/train_GMM_distr_nv4w_T1.h5', 'label'), dtype=np.float32).T
	print 'reading image feat'
        image_feat_all = read_h5(root + 'train_img_feat_3crop_mean_all.h5', 'feature')
	set_size = int(image_feat_all.shape[0] / nDataset)
	N = image_feat_all.shape[0]
	#assert N == labels.shape[0]
	docVec_all = read_h5(root + 'train/train_tfidf_word2vec_cluster.h5', 'feature')
	assert N == docVec_all.shape[0]
	assert N == sentence.shape[0]
	print 'read done. %d' %N
 		
	print 'building graph'
        #train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.total_loss)
	t_vars = tf.trainable_variables()
	#opt_var_list = [var for var in t_var if 'lstm' not in var.name]
	train_op = self.build_trainop(self.total_loss, lr=lr, t_vars=tf.trainable_variables())
	sess.run(tf.global_variables_initializer())
	if is_load:
	    var_list = [var for var in t_var if 'image' in var.name]
	    loader = tf.train.Saver(var_list = var_list)
	    loader.restore(sess, ckpt_path)
	    print '%s loaded' %ckpt_path	
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
	
	batch_size = self.batch_size
        step = 0
        t0 = time.time()
	img_feat_set = []
	print 'start from step %d, lr=%f' %(step, lr)
	iters_per_epoch = int(N/batch_size)
	print 'use dataset %d/%d' %(useDataset, nDataset)
	print 'repeat %d. %d iters/epoch. %d iters/(epoch*repeat)' %(REPEAT, iters_per_epoch, 
	                                                             iters_per_epoch*REPEAT)
	if useDataset == 1:
	    sentence_set = sentence[:set_size]
	    img_feat_set = image_feat_all[:set_size, :]
	    docVec_set = docVec_all[:set_size, :]
        for epoch in range(maxEpoch):
	    for setId in np.random.permutation(useDataset):
		if useDataset > 1:
		    del img_feat_set
		    set_range = range(setId*set_size, (setId + 1)*set_size)
		    sentence_set = sentence[set_range]
		    #label_set = labels[set_range, :]
		    img_feat_set = image_feat_all[set_range, :]
		    docVec_set = docVec_all[set_range, :]
		batch_idx = int(set_size / batch_size)
		for _ in range(REPEAT):
		    # shuffle
		    idxArr = np.random.permutation(set_size)
		    for idx in range(batch_idx):
			interval = range(idx*batch_size , (idx+1)*batch_size)
			idx_batch = idxArr[interval]
			raw_sentence = self.read_wordvector(sentence_set[idx_batch], batch_size)
			img_feat = img_feat_set[idx_batch, :]
			docVec = docVec_set[idx_batch, :]
		     #   label_batch = label_set[idx_batch, :]

			# train
			feed_dict = {self.raw_sentence: raw_sentence, self.image_feat: img_feat, self.docVec: docVec}
			_, summary, total_loss = sess.run([train_op, summary_op, self.total_loss], feed_dict=feed_dict)
    
			if np.mod(step, 2) == 0:
			    summary_writer.add_summary(summary, global_step=step)
			if np.mod(step+1, 1000) == 0:
			    self.saver.save(sess, model_save_path, global_step=step+1)
			    print 'model saved to %s-%d' %(model_save_path, step+1)
			if np.mod(step, 100) == 0:
			    t = (time.time() - t0)/3600
			    print '%.2f hours. Epoch %d, Iteration %d. total loss = %.4f' %(t, epoch, step, total_loss)
			step += 1
                
    def train_softmax(self, sess, maxEpoch=300, lr=0.001, is_load=False,ckpt_path=''):
        logdir = './log/lstm_crossEnt/run1'
        model_save_path='/media/wwt/860G/model/tf_souhu/lstm_crossEnt_T1/'
        make_if_not_exist(model_save_path)
        model_save_path += 'ckpt'
        root = '/media/wwt/860G/data/souhu_data/fusai/'
        sentence = np.array(open(root + 'train_txt/train_cont.txt','r').readlines())
        #labels = np.load(root + 'doc_class_label.npy')
        #assert max(labels) == self.num_classes - 1
        #assert min(labels) == 0
	#labels = np.array(read_h5(root + 'train/train_topOneHot_T1.h5', 'label'), dtype=np.float32)
	labels = joblib.load(root + 'train/kmeans_docVec_1000.model').labels_
        train_op =self.build_trainop(self.total_loss, lr=lr, clipping_norm=1e10, t_vars=tf.trainable_variables())
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()
        N = labels.shape[0]
        assert N == sentence.shape[0]
        batch_idx = int(N / self.batch_size)
        sess.run(tf.global_variables_initializer())
        if is_load:
	    t_vars = tf.trainable_variables()
	    var_list = [var for var in t_vars if 'classify' not in var.name]
	    loader = tf.train.Saver(var_list=var_list)
            self.saver.restore(sess, ckpt_path)
            print '%s loaded' %ckpt_path                                    
        step =0
        t0 = time.time()
        for epoch in range(maxEpoch):
            # shuffle
            idxArr = np.random.permutation(N)
            for idx in range(batch_idx):
                interval = range(idx*self.batch_size , (idx+1)*self.batch_size)
                raw_sentence = self.read_wordvector(sentence[idxArr[interval]], self.batch_size)
                batch_labels = labels[idxArr[interval]]
                feed_dict = {self.raw_sentence: raw_sentence, self.labels:batch_labels}
                _, summary, total_loss = sess.run([train_op, summary_op, self.crossEnt_loss], feed_dict=feed_dict)
    
                if np.mod(step, 2) == 0:
                    summary_writer.add_summary(summary, global_step=step)
                if np.mod(step, 500) == 0:
                    self.saver.save(sess, model_save_path, global_step=step)
                if np.mod(step, 20) == 0:
                    t = (time.time() - t0)/3600
                    print '%.2f hours. Iteration %d. total loss = %.4f' %(t, step, total_loss)
                step += 1    

    def test_embed(self, sess, feat_file, model_path, scope, save_path, h5dataset='embed', batch_size=100):
        '''
        For testing. Generate the final embedding of images for image-text matching.
        Dataset: 'embed'
        Scope: either 'image' or 'sentence'
        ''' 
        # read input features
        if scope == 'image':
            target_tensor = self.image_fc2
            input_tensor = self.image_feat
            h5file = h5py.File(feat_file, 'r')
            feat_all = np.array(h5file['feature'])
            h5file.close()            
        elif scope == 'sentence':
            target_tensor = self.sentence_fc2
            input_tensor = self.raw_sentence
            feat_all = np.array(open(feat_file).readlines())
            docVec_all = np.array(read_h5('/media/wwt/860G/data/souhu_data/fusai/test/test_tfidf_word2vec_cluster.h5', 'feature'))
        else:
            print 'invalid scope %s (must be either image or sentence)' %target_tensor.name
            sys.exit(1)   
        #feat_all = feat_all[:10000, ...]
        N = feat_all.shape[0]
        assert np.mod(N, batch_size) == 0
        # load model
        t_var = tf.global_variables()
        load_var = [var for var in t_var ]
        loader = tf.train.Saver(var_list = load_var)
        loader.restore(sess, model_path)
        # forward
        embed = []
        t0 = time.time()
        for idx in range(N/batch_size):
            interval = np.array(range(idx * batch_size, (idx + 1) * batch_size))
            feat_batch = feat_all[interval]
	    if scope == 'sentence':
		feat_batch = self.read_wordvector(feat_batch, batch_size)
                docVec_batch = docVec_all[interval]
                feed_dict = {input_tensor: feat_batch, self.docVec: docVec_batch}
            else:    
                feed_dict = {input_tensor: feat_batch}
            embed_batch = sess.run(target_tensor, feed_dict=feed_dict)
            embed.extend(embed_batch)
            if np.mod(idx, 10) == 0:
                t = (time.time() - t0)/60
                print '%.2f minutes. Iteration %d/%d' %(t, idx, N/batch_size)
        # save embed
        embed = np.array(embed)
        h5file = h5py.File(save_path, 'w')
        h5file.create_dataset(h5dataset, data=embed)
        h5file.close()
        print 'embed done for scope %s. Saved shape ' %scope, embed.shape


    def read_wordvector(self, batch_sentences, batch_size):
        #ss=time.time()
        batch_vectors=np.zeros([batch_size, self.sentence_len, self.vector_size])
        for i, v in enumerate(batch_sentences):
            sentence_matrix = np.zeros([self.sentence_len, self.vector_size])
            vsp = v.strip().split(' ')
            #assert len(vsp) >= self.sentence_len
            for j, word in enumerate(vsp):
                if j>= self.sentence_len:
                    break                
                try:
                    wordvec = self.word2vec[word]
                    sentence_matrix[j,:]=wordvec
                except:
                    #print 'ignore %s'%word
                    pass
                
            batch_vectors[i,:,:]=sentence_matrix
        return batch_vectors
        

    
if __name__ == '__main__':
    is_train = False
    is_load = True
    root = '/media/wwt/860G/data/souhu_data/fusai/'
    ckpt_path = '/media/wwt/860G/model/tf_souhu/concat/ckpt-15000' # for loading pre-trained model
    #ckpt_path = '/media/wwt/860G/model/tf_souhu/unite2/ckpt-9000'
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
	batch_size = 2000 if is_train else 2000
        model = BidirectionNet(is_training=True, batch_size = batch_size, word2vec_model=root+'train/ourword2vec.pkl') # 
        if is_train:
            model.train(sess, is_load=is_load, ckpt_path=ckpt_path)
            #model.train_softmax(sess, is_load=is_load, ckpt_path=ckpt_path)
        else:
	    # extract embedding
	    root = '/media/wwt/860G/data/souhu_data/fusai/'
	    model_path = '/media/wwt/860G/model/tf_souhu/temp_unite3/ckpt-60000'
	    save_path = root + 'test/embed/test_img_embed_unite3_6w.h5'
	    feat_file = root + 'test/test_img_feat_3crop_mean.h5'#'test_txt/test.txt'#'test_txt/test_cont.txt'
	    scope = 'image'
            model.test_embed(sess, feat_file, model_path, scope, save_path,batch_size=batch_size)	    







