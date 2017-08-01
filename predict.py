import glob
import os
import librosa
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# predict parameters
batch_size = 50
display_step = 200

# Network Parameters
n_input = 50 
n_steps = 50
n_hidden = 300
n_classes = 2 

# Shape tensors
x = tf.placeholder("float", [None, n_input, n_steps])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#Defining RNN
def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(n_hidden,state_is_tuple = True)
    cell = rnn_cell.MultiRNNCell([cell])
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = RNN(x, weight, bias)    

# Initializing the variables
init = tf.global_variables_initializer()

#Get data Function

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features_test(input_test ,bands = n_input, frames = n_steps):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    sound_clip,s = librosa.load(input_test)

    for (start,end) in windows(sound_clip,window_size):
        tmp_clip = sound_clip[int(start):int(end)]
        if len(tmp_clip) == window_size:
            signal = tmp_clip
            mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
            mfccs.append(mfcc)

    features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return np.array(features)

# Predict Function
def predict(sample_path="trainset/blacktop/blacktop-0-.flac", model_path="./model/small/model_small.ckt.meta"):

	saver = tf.train.import_meta_graph(model_path)
	ts_features = extract_features_test(sample_path)

	type0 = []
	type1 = []

	with tf.Session() as sess:
	    # Initialize variables
	    sess.run(init)
	    
	    # Restore model weights from previously saved model
	    offset = 0
	    while offset<len(ts_features):
	    	batch_x = ts_features[offset:(offset + batch_size), :, :]
	    	feed_dict = {x: batch_x}
	    	classification = sess.run(tf.argmax(prediction,1),feed_dict)
	    	tmp1 = 0.
	    	for ii in classification:
	    		if ii == 1:
	    			tmp1 = tmp1 + 1;
	    	tmptmp = float(tmp1)/len(classification)
	    	type0.append(1.-tmptmp)
	    	type1.append(tmptmp)
	    	offset= offset + batch_size
	return type0, type1
	    	
print(predict())