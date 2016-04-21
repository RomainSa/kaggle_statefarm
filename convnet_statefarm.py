import numpy as np
import pandas as pd
from input_data import DataSet
from sklearn.metrics import confusion_matrix
import tensorflow as tf

input_size = (84, 84, 3)
images_folder = '/Users/roms/Documents/Kaggle/StateFarm/Data/imgs'
images_folder = '/home/ubuntu/data/kaggle_statefarm'

test = pd.read_csv('test_labels.csv', names=['image', 'label'])
test.image = test.image.apply(lambda x: images_folder + '/test/' + x)
mnist = DataSet(folder=images_folder, new_size=input_size,
                substract_mean=False, subsample_size=None, test=test)

lr = 1e-4
keep_prob_ = 0.5
lambda_ = 0.


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s=1):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')   # SAME => output size = input size


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, input_size[0], input_size[1], input_size[2]])   # 784 pixels = 28 * 28 image
y_ = tf.placeholder("float", shape=[None, 10])

n1 = 32
n2 = 64
n3 = 64
n4 = 512

keep_prob = tf.placeholder("float")

W_conv1 = weight_variable([8, 8, input_size[2], n1])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv1 = bias_variable([n1])

W_conv2 = weight_variable([4, 4, n1, n2])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv2 = bias_variable([n2])

W_conv3 = weight_variable([3, 3, n2, n3])   # [patch_size1, patch_size2, input_channels, output_channels]
b_conv3 = bias_variable([n3])

W_fc1 = weight_variable([input_size[0] * input_size[1] * n3, n4])
b_fc1 = bias_variable([n4])

W_fc2 = weight_variable([n4, 10])   # output size: 10
b_fc2 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1, s=4) + b_conv1)   # output size: None, 28, 28, n1
h_pool1_drop = tf.nn.dropout(h_conv1, keep_prob)

h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2, s=2) + b_conv2)   # output size: None, 14, 14, n2
h_pool2_drop = tf.nn.dropout(h_conv2, keep_prob)

h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3, s=1) + b_conv3)   # output size: None, 14, 14, n2
h_pool3_drop = tf.nn.dropout(h_conv3, keep_prob)

h_pool3_flat = tf.reshape(h_pool3_drop, [-1, input_size[0] * input_size[1] * n3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)   # output size: n3
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
              tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc1))

cross_entropy = -tf.reduce_mean(y_ * tf.log(y + 1e-9))
cross_entropy += lambda_ * regularizers   # Add the regularization term to the loss.
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()

# add summary variables
wc1_hist = tf.histogram_summary("weights conv 1", W_conv1)
wc2_hist = tf.histogram_summary("weights conv 2", W_conv2)
wc3_hist = tf.histogram_summary("weights conv 3", W_conv3)
wd1_hist = tf.histogram_summary("weights dense 1", W_fc1)
wd2_hist = tf.histogram_summary("weights dense 2", W_fc2)
bc1_hist = tf.histogram_summary("biases conv 1", b_conv1)
bc2_hist = tf.histogram_summary("biases conv 2", b_conv2)
bc3_hist = tf.histogram_summary("biases conv 3", b_conv3)
bd1_hist = tf.histogram_summary("biases dense 1", b_fc1)
bd2_hist = tf.histogram_summary("biases dense 2", b_fc2)
y_hist = tf.histogram_summary("predictions", y)
ce_summ = tf.scalar_summary("cost", cross_entropy)
accuracy_summary = tf.scalar_summary("accuracy", accuracy)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter("/tmp/convnet_logs", sess.graph)
    for i in range(15000+1):
        batch = mnist.next_batch(50)
        if i % 100 == 0:
            print('[Step', str(i) + '] TRAIN error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                  '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),')')
            result = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, i)
        if i % 1000 == 0:
            batch = mnist.next_test_batch(600)
            print('TEST error:', 1-accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),
                  '(Crossentropy:', cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}),')')
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_prob_})

    # save the model
    save_path = saver.save(sess, "/tmp/convnet_statefarm.ckpt")

    # plotting confusion matrix
    batch = mnist.next_test_batch(600)
    test_predictions = y.eval(feed_dict={x: batch[0], keep_prob: 1.0})
    test_predictions = tf.argmax(test_predictions, 1).eval()
    truth = tf.argmax(batch[1], 1).eval()
    cm = confusion_matrix(truth, test_predictions)
    print(cm)

    print 'Making predictions on test set...'
    predictions_ = np.empty((0, 10))
    j = 0
    while predictions_.shape[0] < len(mnist.prediction_files):
        j += 1
        print j
        predictions_ = np.concatenate((predictions_, y.eval(feed_dict={x: mnist.next_prediction_batch(500), keep_prob: 1.0})))
    np.save('mnist_predictions.npy', predictions_)
    np.save('mnist_predictions_names.npy', mnist.prediction_files)

    print 'Saving predictions to csv...'
    with open('submission_' + str(np.random.rand())[2:] + '.csv', 'w+') as f:
        f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
        for i, p in enumerate(predictions_):
            f.write(mnist.prediction_files[i].split('/')[-1] + ',' + ','.join([str(x) for x in p]) + '\n')
