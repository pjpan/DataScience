# This is a script I applied early in toe competition ~ LB = 3. With bounding box regression  
# (applying this to fishes rather than whole images achieves a much better score)
# I used this script to learn about CNNs, feature extraction and using features learned by the InceptionV3 CNN
# to perform classificaiton using a SVM architecture.
# Inspired (adapted heavily) from: http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

model_dir = 'latest_submission/'
# all training images
images_dir = 'SVM_training_set/'
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]


# setup tensorFlow graph initiation
def create_graph():
	with gfile.FastGFile(os.path.join(model_dir, 'output.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

# extract all features from pool layer of InceptionV3
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	labels = []
	create_graph()
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,
			{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
			labels.append(re.split('_\d+',image.split('/')[1])[0])
		return features, labels


features,labels = extract_features(list_images)

pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

features = pickle.load(open('features'))
labels = pickle.load(open('labels'))

# run a 10-fold CV SVM using probabilistic outputs. 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

# probabalistic SVM
clf =  sklearn.calibration.CalibratedClassifierCV(svm)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)


k_fold = KFold(len(labels),n_folds=10, shuffle=False, random_state=0)
C_array=[0.001,0.01,0.1,1,10]
C_scores=[]

for k in C_array:
	clf = svm.SVC(kernel='linear', C=k)
	scores= cross_val_score(clf, features, labels, cv=k_fold, n_jobs=-1)
	C_scores.append(scores.mean())
	print C_scores

#C = 0.1 is best

#clf = svm.LinearSVC(C=0.1)
clf = svm.SVC(kernel='linear', C=0.1,probability=True)

# final_model = clf.fit(features, labels)

final_model = CalibratedClassifierCV(clf,cv=10,method='sigmoid')
final_model = clf.fit(features, labels)


test_dir='latest_submission/test_stg1/test_stg1/'
list_images = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]


def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	create_graph()
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,
			{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
		return features


features_test = extract_features(list_images)

y_pred = final_model.predict_proba(features_test)
#y_pred = final_model.predict(features_test)
#y_pred = final_model.predict(features_test)


image_id = [i.split('/')[3] for i in list_images]

submit = open('submit.SVM.csv','w')
submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')

for idx, id_n in enumerate(image_id):
	probs=['%s' % p for p in list(y_pred[idx, :])]
	submit.write('%s,%s\n' % (str(image_id[idx]),','.join(probs)))

submit.close()