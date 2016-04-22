import tensorflow as tf
import numpy as np
import pickle
import math
import random


################################################################################
#Load Data from picked files
train = [2,3,4]
validate = 1

data = []
for i in train:
	fileName = '/Users/Leo/Desktop/clean/finalData/crossValidationPart'+str(i) +'.pickle'
	print "filename ", fileName
	pickledFile = open(fileName,'r')
	tempData = pickle.load(pickledFile)
	tempData = tempData.astype(np.float32)
	print tempData.shape
	tempData = tempData.tolist()
	data = data+tempData
	pickledFile.close()

data = np.asarray(data)
print data.shape

validationSetFileName = fileName = '/Users/Leo/Desktop/clean/finalData/crossValidationPart'+str(validate) +'.pickle'
pickledFile = open(fileName,'r')
validationData = pickle.load(pickledFile)
validationData = validationData.astype(np.float32)

################################################################################
# Utility Functions

def oneHotEncoding(n):
	res = []
	for each in n:
		s = [0 for i in xrange(7)]
		s[int(each)-1] = 1
		res.append(s)
	return np.asarray(res)

def nextBatch(data,datasize):
	result = set()
	totalDataSize = data[0].shape[0]
	for x in range (0, datasize):
	    num = random.randint(0, totalDataSize-1)
	    while num in result:
	        num = random.randint(0, totalDataSize-1)
	    result.add(num)
	x = []
	y_ = []
	for each in result:
		x.append(data[0][each,:])
		y_.append(data[1][each,:])
	return np.asarray(x), np.asarray(y_)

def init_weights(shape, namedAs, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=namedAs)
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32), name=namedAs)
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(1.0/(fan_in + fan_out)) # {sigmoid:4, nn.relu:1} 
        high = 4*np.sqrt(1.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32), name=namedAs)

################################################################################
# Define Input and Label data variables

input_data = data[:,1:4]
label_data = oneHotEncoding(data[:,4]) 
train_size = input_data.shape[0]
acc_data = []
acc_data.append(input_data)
acc_data.append(label_data)

validation_test_data = validationData[:,1:4]
validation_label_data = oneHotEncoding(validationData[:,4])

################################################################################
#Define input and output tensor placeholders

input_size = 3
output_size = 7 #Number of classes


x = tf.placeholder(tf.float32, [None, input_size], 'x')
y_ = tf.placeholder(tf.int32, [None, output_size], 'y_')

################################################################################

#Hidden Layer 1
hidden1_units = 4
weights1 = init_weights(
        [input_size, hidden1_units], 'weights1',
        'xavier',
        xavier_params=(input_size, hidden1_units))

biases1 = init_weights([1,hidden1_units], 'biases1','zeros')
hidden1 = tf.tanh(tf.matmul(x, weights1) + biases1)

################################################################################
#Hidden Layer 2
hidden2_units = 5
weights2 = init_weights(
        [hidden1_units, hidden2_units], 'weights2',
        'xavier',
        xavier_params=(hidden1_units, hidden2_units))

biases2 = init_weights([1,hidden2_units], 'biases2','zeros')

hidden2 = tf.tanh(tf.matmul(hidden1, weights2))

################################################################################
#Hidden Layer 3
hidden3_units = 6
weights3 = init_weights(
        [hidden2_units, hidden3_units], 'weights3',
        'xavier',
        xavier_params=(hidden2_units, hidden3_units))

biases3 = init_weights([1,hidden3_units], 'biases3','zeros')

hidden3 = tf.tanh(tf.matmul(hidden2, weights3) + biases3)

################################################################################
#Output Layer. Not nn.relu here. Linear operation. 
weights4 = init_weights(
        [hidden3_units, output_size], 'weights4',
        'xavier',
        xavier_params=(hidden3_units, output_size))
biases4 = init_weights([1,output_size], 'biases4','zeros')

logits = tf.matmul(hidden3, weights4) + biases4

################################################################################
# Define Loss here
def loss(logits, labels,regularizers):
	labels = tf.to_float(labels)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,labels,name='xentropy')
	loss = tf.reduce_mean(cross_entropy,name='xentropy_mean')
	loss += 5e-4 * regularizers
	return loss

################################################################################
# Define for training
global_step = tf.Variable(0, trainable=False)
learning_rate = 2e-2

optimizer = tf.train.RMSPropOptimizer(learning_rate,decay=0.99, momentum=0.9, epsilon=1e-10)

regularizers = (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) +
                  tf.nn.l2_loss(weights3) + tf.nn.l2_loss(weights4))

computedLoss = loss(logits,y_,regularizers)

train = optimizer.minimize(computedLoss,global_step=global_step)

#init = tf.initialize_all_variables()

saver = tf.train.Saver()

# Launch the graph.
sess = tf.Session()
#sess.run(init)
saver.restore(sess,"/Users/Leo/Desktop/clean/models/newCrossValidation1/model_10000_1.13671_a64-44.ckpt")

for step in xrange(100001):
	next_x, next_y_ = nextBatch(acc_data,4000)
	_, losss, logitss = sess.run([train, computedLoss, logits], feed_dict={x: next_x, y_: next_y_})
	if step%100 == 0:
		print step ,
		print "  Loss: ",losss
	if step%1000 == 0:
		save_path = saver.save(sess, "/Users/Leo/Desktop/clean/models/newCrossValidation1/attempt_4/model_"+str(step)+"_"+str(losss)+".ckpt")
 		print("Model saved in file: %s" % save_path)
# 	if step%10000 == 0:	
 		correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print "ACCURACY : ", 
		print(sess.run(accuracy, feed_dict={x: validation_test_data, y_: validation_label_data}))

correct_prediction = tf.equal(tf.argmax(logits,1)+1, tf.argmax(y_,1)+1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: validation_test_data, y_: validation_label_data}))