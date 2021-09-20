import numpy as np
import keras.models
from keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
import tensorflow as tf
import os

def init():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	# print('x',dir_path) 
	json_file = open(dir_path +'/mnist.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights(dir_path +"/weights.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.compat.v1.get_default_graph()

	return loaded_model,graph