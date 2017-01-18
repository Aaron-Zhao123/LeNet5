from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
learning_rate = 0.001
training_epochs = 10
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

'''
pruning Parameters
'''
# sets the threshold
prune_threshold_cov = 0.08
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 100
ENABLE_PRUNING = 0


# Store layers weight & bias
# def initialize_tf_variables(first_time_training):
#     if (first_time_training):
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

#store the masks
# weights_mask = {
#     'cov1': tf.Variable(tf.ones([5, 5, NUM_CHANNELS, 32]), trainable = False),
#     'cov2': tf.Variable(tf.ones([5, 5, 32, 64]), trainable = False),
#     'fc1': tf.Variable(tf.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]), trainable = False),
#     'fc2': tf.Variable(tf.ones([512, NUM_LABELS]), trainable = False)
# }
    # else:
    #     with open('assets.pkl','rb') as f:
    #         (weights, biases, weights_mask) = pickle.load(f)

# weights_mask = {
#     'cov1': np.ones([5, 5, NUM_CHANNELS, 32]),
#     'cov2': np.ones([5, 5, 32, 64]),
#     'fc1': np.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
#     'fc2': np.ones([512, NUM_LABELS])
# }
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

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''

def prune_weights(sess, threshold, weights, weight_mask):
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight_mask[key] = np.abs(weights[key].eval(sess)) > threshold[key]
    with open('mask.pkl', 'wb') as f:
        pickle.dump(weight_mask, f)

def mask_weights():
    mask_ops = []
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight = weights[key]
        mask = weights_mask[key]
        mask_ops.append(weights[key].assign(tf.multiply(weight,mask)))
    return mask_ops
'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(grads_and_names, weight_masks):
    new_grads = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

'''
plot weights and store the fig
'''
def plot_weights(sess,pruning_info):
        keys = ['cov1','cov2','fc1','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval(sess).flatten()
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

'''
Define a training strategy
'''
def main(argv = None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:],'hp:')
            for opt, val in opts:
                if (opt == '-p'):
                    pruning_number = val
                    break
            print('pruning count is {}'.format(pruning_number))
        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        pruning_number = int(pruning_number)
        if (pruning_number == 0):
            weights_mask = {
                'cov1': np.ones([5, 5, NUM_CHANNELS, 32]),
                'cov2': np.ones([5, 5, 32, 64]),
                'fc1': np.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
                'fc2': np.ones([512, NUM_LABELS])
            }
        else:
            with open('mask.pkl','rb') as f:
                weights_mask = pickle.load(f)

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        # Construct model
        pred, pool = conv_network(x_image, weights, biases)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        merged = tf.merge_all_summaries()
        saver = tf.train.Saver()

        # I need to fetch this value
        variables = [weights['cov1'], weights['cov2'], weights['fc1'], weights['fc2'],
                    biases['cov1'], biases['cov2'], biases['fc1'], biases['fc2']]
        org_grads = trainer.compute_gradients(cost, var_list = variables, gate_gradients = trainer.GATE_OP)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        new_grads = mask_gradients(org_grads, weights_mask)

        train_step = trainer.apply_gradients(new_grads)


        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exists
            # if (os.path.isfile("tmp_20160118/model.meta")):
            #     op = tf.train.import_meta_graph("tmp_20160108/model.meta")
            #     op.restore(sess,tf.train.latest_checkpoint('tmp_20160105/'))
            #     print ("model found and restored")

            keys = ['cov1','cov2','fc1','fc2']

            # retain the masks on the weights
            for key in keys:
                sess.run(weights[key].assign(weights[key].eval()*weights_mask[key]))
            # print('Before training....')
            prune_info(weights,0)
            plot_weights(sess, 'before training'+ str(pruning_number))
            # Training cycle
            training_cnt = 0
            pruning_cnt = 0
            train_accuracy = 0
            accuracy_list = np.zeros(10)

            print('Training starts ...')
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    # execute a pruning
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # print('before apply gradients:')
                    # prune_info(weights,1)
                    [_, c, train_accuracy] = sess.run([train_step, cost, accuracy], feed_dict = {
                            x: batch_x,
                            y: batch_y})
                    # print('after apply gradients:')
                    # prune_info(weights,1)
                    weights_info(training_cnt, c, train_accuracy)
                    training_cnt = training_cnt + 1
                    accuracy_list = np.concatenate((np.array([train_accuracy]),accuracy_list[0:9]))
                    accuracy_mean = np.mean(accuracy_list)
                    # if (training_cnt == 10):
                    if (accuracy_mean > 0.9):
                        print('Training ends')
                        saver.save(sess, "tmp_20160118/model")
                        plot_weights(sess, 'after pruning'+ str(pruning_number))
                        print("saving model ...")
                        threshold = {
                            'cov1' : 0.08,
                            'cov2' : 0.08,
                            'fc1' : 1,
                            'fc2' : 1
                        }
                        prune_weights(sess, threshold, weights, weights_mask)
                        # mask_info(weights_mask)
                        sys.exit()



                    with open('log/data0118.txt',"a") as output_file:
                		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

                # if epoch % display_step == 0:
            	# 	saver.save(sess, "tmp_20161225/prunned_model.ckpt")
            print("Optimization Finished!")
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2

def weights_info(iter,  c, train_accuracy):
    print('This is the {}th iteration, cost is {}, accuracy is {}'.format(
        iter,
        c,
        train_accuracy
    ))

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))

def mask_info(weights):
    (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    sys.exit(main())
