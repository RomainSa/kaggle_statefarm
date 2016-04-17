import numpy as np
import pandas as pd
from input_data import DataSet
import tensorflow as tf

input_size = (28, 28, 1)
images_folder = '/Users/roms/Documents/Kaggle/StateFarm/Data/imgs'
images_folder = '/home/ubuntu/data/kaggle_statefarm'

test = pd.read_csv('test_labels.csv', names=['image', 'label'])
test.image = test.image.apply(lambda x: images_folder + '/test/' + x)
mnist = DataSet(folder=images_folder, new_size=input_size,
                substract_mean=False, subsample_size=None, test=test)

lr = 1e-4


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   # SAME => output size = input size


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, input_size[0], input_size[1], input_size[2]])   # 784 pixels = 28 * 28 image
y_ = tf.placeholder("float", shape=[None, 10])

n1 = 32
n2 = 64
n3 = 1024
parameters = 5 * 5 * n1 + 5 * 5 * n2 + 7 * 7 * n2 * n3
print 'Number of parameters', parameters

keep_prob = tf.placeholder("float")

W_conv1 = weight_variable([2, 2, input_size[2], n1])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv1 = bias_variable([n1])

W_conv2 = weight_variable([2, 2, n1, n2])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv2 = bias_variable([n2])

W_fc1 = weight_variable([input_size[0] / 4 * input_size[1] / 4 * n2, n3])
b_fc1 = bias_variable([n3])

W_fc2 = weight_variable([n3, 10])   # output size: 10
b_fc2 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)   # output size: None, 28, 28, n1
h_pool1 = max_pool_2x2(h_conv1)   # output size: None, 14, 14, n1
h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

h_conv2 = tf.nn.relu(conv2d(h_pool1_drop,W_conv2) + b_conv2)   # output size: None, 14, 14, n2
h_pool2 = max_pool_2x2(h_conv2)   # output size: None, 7, 7, n2
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

h_pool2_flat = tf.reshape(h_pool2_drop, [-1, input_size[0] / 4 * input_size[1] / 4 * n2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # output size: n3
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y + 1e-9))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# add summary variables
wc1_hist = tf.histogram_summary("weights conv 1", W_conv1)
wc2_hist = tf.histogram_summary("weights conv 2", W_conv2)
wd1_hist = tf.histogram_summary("weights dense 1", W_fc1)
wd2_hist = tf.histogram_summary("weights dense 2", W_fc2)
bc1_hist = tf.histogram_summary("biases conv 1", b_conv1)
bc2_hist = tf.histogram_summary("biases conv 2", b_conv2)
bd1_hist = tf.histogram_summary("biases dense 1", b_fc1)
bd2_hist = tf.histogram_summary("biases dense 2", b_fc2)
y_hist = tf.histogram_summary("predictions", y)
ce_summ = tf.scalar_summary("cost", cross_entropy)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("/tmp/alexnet_logs", sess.graph)
    for i in range(15000+1):
        writer = tf.train.SummaryWriter("/tmp/alexnet_logs", sess.graph)
        batch = mnist.next_batch(50)
        if i % 100 == 0:
            print('[Step', str(i) + '] TRAIN error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                  '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),')')
            result = sess.run([merged, accuracy], feed_dict={x: batch[0], y: batch[1], keep_prob: 1.})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
        if i % 1000 == 0:
            batch = mnist.next_test_batch(100)
            print('TEST error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                  '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),')')
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print 'Making predictions on test set...'
predictions_ = np.empty((0, 10))
j = 0
while predictions_.shape[0] < len(mnist.prediction_files):
    j += 1
    print j
    predictions_ = np.concatenate((predictions_, y.eval(feed_dict={x: mnist.next_prediction_batch(500), keep_prob: 1.0})))

print 'Saving predictions to csv...'
with open('submission_' + str(np.random.rand())[2:] + '.csv', 'w+') as f:
    f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    for i, p in enumerate(predictions_):
        f.write(mnist.prediction_files[i].split('/')[-1] + ',' + ','.join([str(x) for x in p]) + '\n')
