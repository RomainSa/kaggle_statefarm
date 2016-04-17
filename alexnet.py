'''
AlexNet implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
AlexNet Paper (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import tensorflow as tf
from input_data import DataSet

# Preprocessing Parameters
input_size = (128, 128, 1)   # new_height, new_width, nb_channels = 1 for black and white
mean_to_substract = None   # does not work well yet

# Training Parameters
learning_rate = 1e-5
n_epochs = 40
batch_size = 64
display_step = 10
dropout = 0.5
initial_bias = 0.01

# Neural network architecture
c1 = 32   # number of convolution filters
c2 = 64
c3 = 128
f1 = 6   # convolution filter size
f2 = 4
f3 = 2
p1 = 4   # pooling filter size
p2 = 2
p3 = 2
d1 = 256   # fully connected layer size
d2 = 128
assert input_size[0] % (p1 * p2 * p3) == 0 and input_size[1] % (p1 * p2 * p3) == 0

# get the data
#data = DataSet(folder='/Users/roms/Documents/Kaggle/StateFarm/Data/imgs', new_size=input_size, substract_mean=False, subsample_size=100)
data = DataSet(folder='/home/ubuntu/data/kaggle_statefarm', new_size=input_size, substract_mean=False, subsample_size=None)

# tf Graph input
training_iters = n_epochs * len(data.labels) // batch_size + 1
n_classes = len(np.unique(data.labels))
x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, input_size[0], input_size[1], input_size[2]])
    # 1st layer (CONVOLUTION)
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])   # Convolution filter
    pool1 = max_pool('pool1', conv1, k=p1)   # Max-Pooling
    norm1 = norm('norm1', pool1, lsize=2)   # Normalization
    norm1_dropout = tf.nn.dropout(norm1, _dropout)   # DropOut
    # 2nd layer (CONVOLUTION)
    conv2 = conv2d('conv2', norm1_dropout, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=p2)
    norm2 = norm('norm2', pool2, lsize=2)
    norm2_dropout = tf.nn.dropout(norm2, _dropout)
    # 3rd layer (CONVOLUTION)
    conv3 = conv2d('conv3', norm2_dropout, _weights['wc3'], _biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=p3)
    norm3 = norm('norm3', pool3, lsize=2)
    norm3_dropout = tf.nn.dropout(norm3, _dropout)
    norm3_dropout_reshaped = tf.reshape(norm3_dropout, [-1, input_size[0] / (p1 * p2 * p3) * input_size[1] / (p1 * p2 * p3) * c3]) # Reshape conv3 output to fit dense layer input
    # 4th layer (FULLY CONNECTED)
    dense1 = tf.nn.relu(tf.matmul(norm3_dropout_reshaped, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
    dense1_dropout = tf.nn.dropout(dense1, _dropout)
    # 5th layer (FULLY CONNECTED)
    dense2 = tf.nn.relu(tf.matmul(dense1_dropout, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    dense2_dropout = tf.nn.dropout(dense2, _dropout)
    # Output, class prediction
    out = tf.nn.softmax(tf.matmul(dense2_dropout, _weights['out']) + _biases['out'])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([f1, f1, input_size[2], c1])),
    'wc2': tf.Variable(tf.truncated_normal([f2, f2, c1, c2])),
    'wc3': tf.Variable(tf.truncated_normal([f3, f3, c2, c3])),
    'wd1': tf.Variable(tf.truncated_normal([input_size[0] / (p1 * p2 * p3) * input_size[1] / (p1 * p2 * p3) * c3, d1])),
    'wd2': tf.Variable(tf.truncated_normal([d1, d2])),
    'out': tf.Variable(tf.truncated_normal([d2, 10]))
}
biases = {
    'bc1': tf.Variable(tf.constant(initial_bias, shape=[c1])),
    'bc2': tf.Variable(tf.constant(initial_bias, shape=[c2])),
    'bc3': tf.Variable(tf.constant(initial_bias, shape=[c3])),
    'bd1': tf.Variable(tf.constant(initial_bias, shape=[d1])),
    'bd2': tf.Variable(tf.constant(initial_bias, shape=[d2])),
    'out': tf.Variable(tf.constant(initial_bias, shape=[n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = -tf.reduce_mean(y * tf.log(pred + 1e-9))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Add summary ops to collect data
wc1_hist = tf.histogram_summary("weights conv 1", weights['wc1'])
wc2_hist = tf.histogram_summary("weights conv 2", weights['wc2'])
wc3_hist = tf.histogram_summary("weights conv 3", weights['wc3'])
wd1_hist = tf.histogram_summary("weights dense 1", weights['wd1'])
wd2_hist = tf.histogram_summary("weights dense 2", weights['wd2'])
wout_hist = tf.histogram_summary("weights output", weights['out'])
bc1_hist = tf.histogram_summary("biases conv 1", biases['bc1'])
bc2_hist = tf.histogram_summary("biases conv 2", biases['bc2'])
bc3_hist = tf.histogram_summary("biases conv 3", biases['bc3'])
bd1_hist = tf.histogram_summary("biases dense 1", biases['bd1'])
bd2_hist = tf.histogram_summary("biases dense 2", biases['bd2'])
bout_hist = tf.histogram_summary("biases output", biases['out'])
y_hist = tf.histogram_summary("predictions", pred)

ce_summ = tf.scalar_summary("cost", cost)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)

merged = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    writer = tf.train.SummaryWriter("/tmp/alexnet_logs", sess.graph)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        batch_xs, batch_ys = data.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            result = sess.run([merged, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 256 test images
    #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: data.test.images[:256], y: data.test.labels[:256], keep_prob: 1.})
