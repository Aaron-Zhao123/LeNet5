'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 100
display_step = 1
# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# Store layers weight & bias
weights = {
    'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1)),
    'cov2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    'fc1': tf.Variable(tf.random_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])),
    'fc2': tf.Variable(tf.random_normal([512, NUM_LABELS]))
}
biases = {
    'cov1': tf.Variable(tf.random_normal([32])),
    'cov2': tf.Variable(tf.random_normal([64])),
    'fc1': tf.Variable(tf.random_normal([512])),
    'fc2': tf.Variable(tf.random_normal([10]))
}

# Create model
def conv_network(x, weights, biases):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'SAME')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'SAME')
    '''get pool shape'''
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    output = tf.matmul(hidden, weights['fc2']) + biases['fc2']
    return output
def plot_weights():
        weights_cov1 = weights['cov1'].eval().flatten()
        weights_cov2 = weights['cov2'].eval().flatten()
        weights_fc1 = weights['fc1'].eval().flatten()
        weights_fc2 = weights['fc2'].eval().flatten()
        plt.figure(1)
        plt.subplot(411)
        plt.hist(weights_cov1, bins = 20)
        plt.title('cov1')
        plt.subplot(412)
        plt.hist(weights_cov2, bins = 20)
        plt.title('cov2')
        plt.subplot(413)
        plt.hist(weights_fc1, bins = 20)
        plt.title('fc1')
        plt.subplot(414)
        plt.hist(weights_fc2, bins = 20)
        plt.title('fc2')
        plt.show()

def plot_biases():
        biases_cov1 = biases['cov1'].eval().flatten()
        biases_cov2 = biases['cov2'].eval().flatten()
        biases_fc1 = biases['fc1'].eval().flatten()
        biases_fc2 = biases['fc2'].eval().flatten()
        plt.figure(2)
        plt.subplot(411)
        plt.hist(biases_cov1, bins = 20)
        plt.title('cov1')
        plt.subplot(412)
        plt.hist(biases_cov2, bins = 20)
        plt.title('cov2')
        plt.subplot(413)
        plt.hist(biases_fc1, bins = 20)
        plt.title('fc1')
        plt.subplot(414)
        plt.hist(biases_fc2, bins = 20)
        plt.title('fc2')
        plt.show()
def plot_training_data():
    raw_data = np.loadtxt('log/data.txt', delimiter = ',')
    plt.subplot(211)
    plt.plot(raw_data[:,0],raw_data[:,1],'b')
    plt.title('training accuracy')
    plt.subplot(212)
    plt.plot(raw_data[:,0],raw_data[:,2],'r')
    plt.title('cross entropy')
    plt.show()

#def plot_biases():
def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    x_image = tf.reshape(x,[-1,28,28,1])

    # Construct model
    pred= conv_network(x_image, weights, biases)

    # Define loss and optimizer

    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.scalar_summary('cross entropy', cost)

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # restore model if exists
        if (os.path.isfile("tmp/model.ckpt")):
            saver.restore(sess, "tmp/model.ckpt")
            print ("model found and restored")
        # Test model
        # Calculate accuracy
        '''accuracy testing
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        '''
        '''histogram plots
        plot_weights()
        plot_biases()
        '''
        plot_training_data()

if __name__ == '__main__':
    main()
