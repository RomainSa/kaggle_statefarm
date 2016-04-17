import numpy as np
import pandas as pd
from input_data import DataSet
import tensorflow as tf

input_size = (224, 224, 3)
images_folder = '/Users/roms/Documents/Kaggle/StateFarm/Data/imgs'
images_folder = '/home/ubuntu/data/kaggle_statefarm'

test = pd.read_csv('test_labels.csv', names=['image', 'label'])
test.image = test.image.apply(lambda x: images_folder + '/test/' + x)
mnist = DataSet(folder=images_folder, new_size=input_size,
                substract_mean=False, subsample_size=1000, test=test)

lr = 1e-4
batch_size = 64


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s=1):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')   # SAME => output size = input size


def max_pool(l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


def norm(l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


x = tf.placeholder("float", shape=[None, input_size[0], input_size[1], input_size[2]])   # 784 pixels = 28 * 28 image
y_ = tf.placeholder("float", shape=[None, 10])

c1 = 16
c2 = 32
c3 = 64
c4 = 64
c5 = 32
f6 = 512
f7 = 512

keep_prob = tf.placeholder("float")

# Weights
W_conv1 = weight_variable([11, 11, input_size[2], c1])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv1 = bias_variable([c1])

W_conv2 = weight_variable([5, 5, c1, c2])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv2 = bias_variable([c2])

W_conv3 = weight_variable([3, 3, c2, c3])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv3 = bias_variable([c3])

W_conv4 = weight_variable([3, 3, c3, c4])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv4 = bias_variable([c4])

W_conv5 = weight_variable([3, 3, c4, c5])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv5 = bias_variable([c5])

W_fc6 = weight_variable([1568, f6])   # output size: 10
b_fc6 = bias_variable([f6])

W_fc7 = weight_variable([f6, f7])   # output size: 10
b_fc7 = bias_variable([f7])

W_fc8 = weight_variable([f7, 10])   # output size: 10
b_fc8 = bias_variable([10])

# Model
conv1 = conv2d(x, W_conv1, s=4)
relu1 = tf.nn.relu(conv1 + b_conv1)
norm1 = norm(relu1, lsize=5)
pool1 = max_pool(norm1, k=3, s=2)

conv2 = conv2d(pool1, W_conv2)
relu2 = tf.nn.relu(conv2 + b_conv2)
norm2 = norm(relu2, lsize=5)
pool2 = max_pool(norm2, k=3, s=2)

conv3 = conv2d(pool2, W_conv3)
relu3 = tf.nn.relu(conv3 + b_conv3)

conv4 = conv2d(relu3, W_conv4)
relu4 = tf.nn.relu(conv4 + b_conv4)

conv5 = conv2d(relu4, W_conv5)
relu5 = tf.nn.relu(conv5 + b_conv5)
pool5 = max_pool(relu5, k=3, s=2)
pool5_flat = tf.reshape(pool5, [-1, 1568])

fc6 = tf.matmul(pool5_flat, W_fc6)
relu6 = tf.nn.relu(fc6 + b_fc6)
dropout6 = tf.nn.dropout(relu6, keep_prob)

fc7 = tf.matmul(dropout6, W_fc7)
relu7 = tf.nn.relu(fc7 + b_fc7)
dropout7 = tf.nn.dropout(relu7, keep_prob)

y = tf.nn.softmax(tf.matmul(dropout7, W_fc8) + b_fc8)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y + 1e-9))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# add summary variables
wc1_hist = tf.histogram_summary("weights conv 1", W_conv1)
wc2_hist = tf.histogram_summary("weights conv 2", W_conv2)
wc3_hist = tf.histogram_summary("weights conv 3", W_conv3)
wc4_hist = tf.histogram_summary("weights conv 4", W_conv4)
wc5_hist = tf.histogram_summary("weights conv 5", W_conv5)
wd1_hist = tf.histogram_summary("weights dense 1", W_fc6)
wd2_hist = tf.histogram_summary("weights dense 2", W_fc7)
wout_hist = tf.histogram_summary("weights output", W_fc8)
bc1_hist = tf.histogram_summary("biases conv 1", b_conv1)
bc2_hist = tf.histogram_summary("biases conv 2", b_conv2)
bc3_hist = tf.histogram_summary("biases conv 3", b_conv3)
bc4_hist = tf.histogram_summary("biases conv 4", b_conv4)
bc5_hist = tf.histogram_summary("biases conv 5", b_conv5)
bd1_hist = tf.histogram_summary("biases dense 1", b_fc6)
bd2_hist = tf.histogram_summary("biases dense 2", b_fc7)
bout_hist = tf.histogram_summary("biases output", b_fc8)
y_hist = tf.histogram_summary("predictions", y)
ce_summ = tf.scalar_summary("cost", cross_entropy)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("/tmp/alexnetreduced_logs", sess.graph)
    for i in range(15000+1):
        batch = mnist.next_batch(batch_size)
        if i % 100 == 0:
            print('[Step', str(i) + '] TRAIN error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                  '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}), ')')
            result = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
        #if i % 1000 == 0:
        #    batch = mnist.next_test_batch(batch_size)
        #    print('TEST error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
        #          '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}), ')')
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print 'Making predictions on test set...'
    predictions_ = np.empty((0, 10))
    j = 0
    while predictions_.shape[0] < len(mnist.prediction_files):
        j += 1
        print j
        predictions_ = np.concatenate((predictions_, y.eval(feed_dict={x: mnist.next_prediction_batch(batch_size), keep_prob: 1.0})))

    print 'Saving predictions to csv...'
    with open('submission_' + str(np.random.rand())[2:] + '.csv', 'w+') as f:
        f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
        for i, p in enumerate(predictions_):
            f.write(mnist.prediction_files[i].split('/')[-1] + ',' + ','.join([str(x) for x in p]) + '\n')
