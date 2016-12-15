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

#pruning Parameters
prune_threshold = 0.1


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

weights_mask = {
    'cov1': np.zeros([5, 5, NUM_CHANNELS, 32]),
    'cov2': np.zeros([5, 5, 32, 64]),
    'fc1': np.zeros([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 *64, 512]),
    'fc2': np.zeros([512, NUM_LABELS])
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
    return output , reshape

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''
def calculate_non_zero_weights(key, weight):
    count = (weight != 0).sum()
    print("The number of zeros in {} is {}".format(key,count))
def prune_weights(sess):
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight = weights[key].eval(sess)
        calculate_non_zero_weights(key+' pre prune', weight)
        # plot_weights(weight, key+'original')
        weights_mask[key] = abs(weight) > prune_threshold
        prunned_weight = weight * weights_mask[key]
        calculate_non_zero_weights(key+' post prune', prunned_weight)
        # plot_weights(prunned_weight, key)

#def mask_gradients(sess):


def plot_weights(weight, name):
        fig, axrr = plt.subplots( 1, sharex = True )  # create figure & 1 axis
        weight = weight.flatten()
        axrr.hist(weight)
        axrr.set_title(name)
        fig.savefig('fig/weights info'+ name)
        plt.close(fig)

def test():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    x_image = tf.reshape(x,[-1,28,28,1])
    # Construct model
    pred, pool = conv_network(x_image, weights, biases)

    # Define loss and optimizer

    with tf.name_scope('cross_entropy'):

    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
       # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
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
        prune_weights(sess)
def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    x_image = tf.reshape(x,[-1,28,28,1])
    # Construct model
    pred, pool = conv_network(x_image, weights, biases)

    # Define loss and optimizer

    with tf.name_scope('cross_entropy'):

    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
       # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
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


        # Training cycle
        training_cnt = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                pred_val = pred.eval(feed_dict={x:batch_x, y: batch_y})
                # print (pred_val)
                training_cnt = training_cnt + 1
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
                print (c)
                with open('log/data.txt',"a") as output_file:
            		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
		saver.save(sess, "tmp/model.ckpt")
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
if __name__ == '__main__':
    main()
