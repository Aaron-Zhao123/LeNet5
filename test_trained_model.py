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
        fig, axrr = plt.subplots( 4, sharex = True )  # create figure & 1 axis
        weights_cov1 = weights['cov1'].eval().flatten()
        weights_cov2 = weights['cov2'].eval().flatten()
        weights_fc1 = weights['fc1'].eval().flatten()
        weights_fc2 = weights['fc2'].eval().flatten()
        weights_cov1 = weights['cov1'].eval().flatten()
        weights_cov2 = weights['cov2'].eval().flatten()
        weights_fc1 = weights['fc1'].eval().flatten()
        weights_fc2 = weights['fc2'].eval().flatten()
        axrr[0].hist(weights_cov1, bins = 20)
        axrr[0].set_title('cov1')
        axrr[1].hist(weights_cov2, bins = 20)
        axrr[1].set_title('cov2')
        axrr[2].hist(weights_fc1, bins = 20)
        axrr[2].set_title('fc1')
        axrr[3].hist(weights_fc2, bins = 20)
        axrr[3].set_title('fc2')
        fig.savefig('fig/weights info')
        plt.close(fig)

def plot_biases():
        fig, axrr = plt.subplots( 4, sharex = True )  # create figure & 1 axis
        biases_cov1 = biases['cov1'].eval().flatten()
        biases_cov2 = biases['cov2'].eval().flatten()
        biases_fc1 = biases['fc1'].eval().flatten()
        biases_fc2 = biases['fc2'].eval().flatten()
        axrr[0].hist(biases_cov1, bins = 20)
        axrr[0].set_title('cov1')
        axrr[1].hist(biases_cov2, bins = 20)
        axrr[1].set_title('cov2')
        axrr[2].hist(biases_fc1, bins = 20)
        axrr[2].set_title('fc1')
        axrr[3].hist(biases_fc2, bins = 20)
        axrr[3].set_title('fc2')
        fig.savefig('fig/biases info')
        plt.close(fig)

def plot_training_data():
    raw_data = np.loadtxt('log/data.txt', delimiter = ',')
    fig, axrr = plt.subplots( 2, sharex = True )  # create figure & 1 axis
    axrr[0].plot(raw_data[:,0],raw_data[:,1],'b')
    axrr[0].set_title('training accuracy')
    axrr[1].plot(raw_data[:,0],raw_data[:,2],'r')
    axrr[1].set_title('cross entropy')
    fig.savefig('fig/train info')
    plt.close(fig)


def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

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
        if (os.path.isfile("tmp_20160105/model.meta")):
            op = tf.train.import_meta_graph("tmp_20160105/model.meta")
            op.restore(sess,tf.train.latest_checkpoint('tmp_20160105/'))
            # saver.restore(sess, "tmp_20160105/model.meta")
            print ("model found and restored")

        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))

        # Test model
        # Calculate accuracy
        # '''accuracy testing
        # '''
        # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        #
        '''histogram plots
        '''
        #plot_weights()
        #plot_biases()
        #plot_training_data()

if __name__ == '__main__':
    main()
