Sohu Chinese Image-Text Matching Competition 2017.
We won the third prize.

# Requirements
Please clone the Chinese word-segmentation tools https://github.com/thunlp/THULAC-Python into ./NLP

And clone a caffe repo into ./caffe

Download a pre-trained VGG-16 model trained on ImageNet with caffe.

# Platform：
tensorflow 1.0

caffe



# Model Architecture：

	Image -> vgg16 -> 4096-d feature -> fc1 -> fc2 -> distance(image, text)
	
	Text TF-IDF or GMM feature -> fc1 ->  fc2 ->  distance(image, text)
	
# References：

《Learning two-branch neural networks for image-text matching tasks》

《Fisher Vectors Derived from Hybrid Gaussian-Laplacian Mixture Models for Image Annotation》

# Files：

    caffe-recurrent/
	- extract_feature.py # extract features using VGG-16

    Tensorflow/
	-BidirectionNet_tfidf.py # train using the tfidf features
	-BidirectionNet_lstm.py # train lstm
	-test_match_pairList.py # generate the matching result (top-10) using image/text embeddings.

    NLP/
	-run_SH_split.py # word segmentation and generate vocabulary

	-word_type.py # filter nouns and verbs

	-run_create_train_txt.py # generate one file .txt based on the word-segmented files and a vocabulary.

	-tfidf_from_seg.py # compute tfidf

	-kmeans_word2vec.py cluster based on word2vec (fasttext word2vec is recommended. Search 'fasttext' repo)

# Key of performance improvment：

model ensemble

OCR （which we did not use）
