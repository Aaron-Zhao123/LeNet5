from __future__ import print_function

# Import MNIST data
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

# Parameters
learning_rate = 0.001
training_epochs = 1000
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
prune_freq = 300
ENABLE_PRUNING = 0


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

#store the masks
weights_mask = {
    'cov1': tf.Variable(tf.ones([5, 5, NUM_CHANNELS, 32]), trainable = False),
    'cov2': tf.Variable(tf.ones([5, 5, 32, 64]), trainable = False),
    'fc1': tf.Variable(tf.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]), trainable = False),
    'fc2': tf.Variable(tf.ones([512, NUM_LABELS]), trainable = False)
}

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

def calculate_non_zero_weights(key, weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    print("{}, total elements are {} and nonzeros are {}".format(key,size,count))

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''
def prune_weights(sess, pruning_cnt):
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight = weights[key].eval(sess)
        calculate_non_zero_weights(key+' pre prune', weight)
        ''' pruning thresholds for cov layer is different from fc layers'''
        if (key == 'cov1' or key == 'cov2'):
            mask = abs(weight) > prune_threshold_cov
        elif (key == 'fc1' or key == 'fc2'):
            mask = abs(weight) > prune_threshold_fc
        prunned_weight = weight * mask
        sess.run(weights[key].assign(prunned_weight))
        sess.run(weights_mask[key].assign(mask))
        calculate_non_zero_weights(key+' post prune' + str(pruning_cnt), weights[key].eval(sess))
        # calculate_non_zero_weights(key+' post prune, mask' + str(pruning_cnt), weights_mask[key].eval(sess))
def mask_weights(sess):
    keys = ['cov1','cov2','fc1','fc2']
    for key in keys:
        weight = weights[key].eval(sess)
        mask = weights_mask[key].eval(sess)
        prunned_weight = weight * mask
        sess.run(weights[key].assign(prunned_weight))
'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(grads_and_names, masks_and_names):
    new_grads = []
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        grad = np.array(grad)
        for mask, mask_name in masks_and_names:
            if (mask_name == var_name):
                # print("found a match, name:{}".format(mask_name))
                # print ('shape of grad is {}'.format(np.shape(grad)))
                mask = np.array(mask)
                # print ('shape of mask is {}'.format(np.shape(mask)))
                new_grads.append(grad*mask)
                # print ('shape of grad* mask is {}'.format(np.shape(grad * mask)))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append(grad)
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
        fig.savefig('fig/weights'+pruning_info)
        plt.close(fig)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

'''
Define a training strategy
'''
def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # keys = ['cov1', 'cov2', 'fc1', 'fc2']
    # for key in keys:
    #     weights_mask[key].__init__(trainable = False)
    keys = ['cov1','cov2','fc1','fc2']
    # for key in keys:
    #     weights[key] = tf.assign(weights, tf.mul(weights[key], weights_mask[key]))

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

    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # I need to fetch this value
    variables = [weights['cov1'], weights['cov2'], weights['fc1'], weights['fc2'],
                biases['cov1'], biases['cov2'], biases['fc1'], biases['fc2']]
    org_grads = trainer.compute_gradients(cost, var_list = variables, gate_gradients = trainer.GATE_GRAPH)
    org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]

    grad_placeholder = []
    for grad_var in org_grads:
        grad_placeholder.append((tf.placeholder('float', shape=grad_var[0].get_shape()) ,grad_var[1]))
    train_step = trainer.apply_gradients(grad_placeholder)

    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # restore model if exists
        # if (os.path.isfile("tmp/model.ckpt")):
            # saver.restore(sess, "tmp/model.ckpt")
            # print ("model found and restored")


        # Training cycle
        training_cnt = 0
        pruning_cnt = 0
        from tempfile import TemporaryFile
        outfile = TemporaryFile()

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # execute a pruning
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                if (training_cnt % prune_freq == 0 and training_cnt != 0):
                    plot_weights(sess, 'pre pruning'+ str(pruning_cnt))
                    prune_weights(sess,pruning_cnt)
                    plot_weights(sess, 'after pruning' + str(pruning_cnt))
                    pruning_cnt = pruning_cnt + 1
                # evaluate all gradients
                grads_and_varnames = []
                for i in range(len(org_grads)):
                    [g,v] = sess.run([org_grads[i][0], org_grads[i][1]],feed_dict={
                        x: batch_x,
                        y: batch_y})
                    grads_and_varnames.append((g,org_grads[i][1].name))
                    # if (i == 0):
                    #     print ('inspect org grad vars: {}'.format((v !=0).sum()))
                    #     print ('inspect org grad vars: {}'.format(np.shape(v)))
                # evaluate the masks
                masks_and_names = []
                keys = ['cov1','cov2','fc1','fc2']
                for key in keys:
                    m = sess.run(weights_mask[key], feed_dict = {})
                    masks_and_names.append((m,weights[key].name))
                    # if key == 'cov1':
                    #     print ('inspect mask: {}'.format((m !=0).sum()))
                        # inspect_m = m
                # mask gradients
                new_grads = mask_gradients(grads_and_varnames, masks_and_names)

                feed_dict = {}
                for i, grad_var in enumerate(org_grads):
                    # calculate_non_zero_weights('newgrad'+str(i), new_grads[i])
                    feed_dict[grad_placeholder[i][0]] = new_grads[i]
                    # print ('inspect newgrad[{}]: {}'.format(i,np.shape(new_grads[i])))
                    # if (i == 0):
                    #     print ('inspect newgrad[{}]: {}'.format(i,(new_grads[i] != 0).sum()))
                    # print (np.shape(new_grads[i]))
                # inspect_grad = np.logical_or(new_grads[0]!=0, inspect_m !=0)

                train_step.run(feed_dict = feed_dict)
                mask_weights(sess)
                # calculate_non_zero_weights('weights cov 1'+str(i), weights['cov1'].eval())

                c = sess.run(cost,  feed_dict={
                        x: batch_x,
                        y: batch_y})
                # wcov1 = weights['cov1'].eval()
                # print ('trainng weights cov1: {}'.format((wcov1 != 0).sum()))
                training_cnt = training_cnt + 1
                #
                #
                #
                # c, g2 = sess.run([cost, grads[0][0]], feed_dict={
                #     x: batch_x,
                #     y: batch_y})
                # _, c, g3= sess.run([train_step, cost, grads[0][0]], feed_dict={
                #     x: batch_x,
                #     y: batch_y})
                # pred_val = pred.eval(feed_dict={x:batch_x, y: batch_y})
                #
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
                print ("training count is {}".format(training_cnt))
                print (c)
                print (train_accuracy)
                # # mask = weights_mask['cov1'].eval(sess)
                # # print((mask != 0).sum())
                # # print((mask_in_grad!= 0).sum())
                #
                with open('log/data1216.txt',"a") as output_file:
            		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            if epoch % display_step == 0:
        		saver.save(sess, "tmp_20161216/prunned_model.ckpt")
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    main()
