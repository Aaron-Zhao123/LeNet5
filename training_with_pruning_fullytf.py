from __future__ import print_function

# Import MNIST data
import input_data
import os.path
import tensorflow as tf
import numpy as np
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
prune_threshold_cov = 0.1
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 4
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
        calculate_non_zero_weights(key+' post prune, mask' + str(pruning_cnt), weights_mask[key].eval(sess))
'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(grads):
    keys = ['cov1','cov2','fc1','fc2']
    # keys = ['Variable_1','Variable_2','Variable_3','Variable_4']
    # grad_placeholder_cov = [
    # tf.placeholder("float", shape=weights['cov1'].get_shape()),
    # tf.placeholder("float", shape=weights['cov2'].get_shape())]
    record = {}
    record_index = {}
    for key in keys:
        index = 0
        for grad, var in grads:
            weight = weights[key]
            mask = weights_mask[key]
            # mask = weights_mask[key]
            if var.name == weights[key].name:
                tmp = tf.to_float(mask)
                new_grad = tf.mul(grad,tmp)
                record[key]= new_grad
                record_index[key] = index
                grads[index] = (new_grad, var)
                print ("new gradient is: {}".format(grads[index]))
                # grads[index] = (tmp, var)
            # grad_placeholder[index].assign(grad)
            index += 1
    return (grads, record, record_index)
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
# def grad_mask_test(sess, grads, org_grads):
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
    org_grads = trainer.compute_gradients(cost)

    (grads, tmp, tmp_index)= mask_gradients(org_grads)
    grads = [(ClipIfNotNone(grad), var) for grad, var in grads]
    train_step = trainer.apply_gradients(grads)

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
                if (training_cnt == prune_freq):
                    pruning_cnt = pruning_cnt + 1
                    # plot_weights(sess,str(pruning_cnt)+"pre_pruning")
                    prune_weights(sess,pruning_cnt)
                    # plot_weights(sess,str(pruning_cnt)+"post_pruning")
                    # print('pre training')
                    update_grad = grads[0][0].eval(feed_dict = {x:batch_x, y:batch_y})
                    # pickle.dump( update_grad, open( "save1.p", "wb" ) )
                    # weights_cov1 = weights['cov1'].eval()
                    # pickle.dump( weights_cov1, open( "save2.p", "wb" ) )
                    # mask_cov1 = weights_mask['cov1'].eval(sess)
                    # print ((np.logical_or(update_grad != 0, weights_cov1 != 0) == (weights_cov1 != 0)).all())
                    # print ((weights_cov1 != 0).sum())
                    # print ((mask_cov1 != 0).sum())

                    c, g1= sess.run([cost, grads[0][0]], feed_dict={
                        x: batch_x,
                        y: batch_y})

                    keys = ['cov1','cov2','fc1','fc2']
                    r1 = {}
                    for key in keys:
                        r1[key]= sess.run([tmp[key]], feed_dict={
                            x: batch_x,
                            y: batch_y})
                        print (tmp_index[key])

                    print("test one")
                    w1 = weights['cov1'].eval(sess)
                    print((w1 != 0).sum())
                    print((( g1!=0 ) == (r1['cov1']!=0)).all())
                    # write_numpy_to_file(g1,"g1.txt")
                    # print(r1['cov1'])
                    r1_ele = r1['cov1']
                    # write_numpy_to_file(r1_ele,"r1.txt")
                    print("shape of g1 is {}".format(np.shape(g1)))
                    print("shape of r1 is {}".format(np.shape(r1['cov1'])))
                    # print ((np.logical_or(g1 != 0, w1 != 0) == (w1!= 0)).all())
                else:
                    c, g2 = sess.run([cost, grads[0][0]], feed_dict={
                        x: batch_x,
                        y: batch_y})

                    if (training_cnt == prune_freq+1):
                        print("test two")
                        w2 = weights['cov1'].eval(sess)
                        print((w2 != 0).sum())
                        print((g2!= 0).sum())
                        print ((np.logical_or(g2 != 0, w2 != 0) == (w2!= 0)).all())
                        write_numpy_to_file(w2,"w2.txt")
                        write_numpy_to_file(g2,"g2.txt")

                    _, c, g3= sess.run([train_step, cost, grads[0][0]], feed_dict={
                        x: batch_x,
                        y: batch_y})
                    pred_val = pred.eval(feed_dict={x:batch_x, y: batch_y})

                    if (training_cnt == prune_freq+1):
                        print("test three")
                        w3 = weights['cov1'].eval(sess)
                        write_numpy_to_file(w3,"w3.txt")
                        print((w3 != 0).sum())
                        write_numpy_to_file(np.logical_xor(w3!=0,w2!=0),"diff.txt")
                        print((g3 != 0).sum())
                        write_numpy_to_file(g3,"g3.txt")
                        print ((np.logical_or(g3 != 0, w3 != 0) == (w3!= 0)).all())
                    if (training_cnt == prune_freq+2):
                        print("test four")
                        w4 = weights['cov1'].eval(sess)
                        print((w4 != 0).sum())
                        print((g3 != 0).sum())
                        exit()

                training_cnt = training_cnt + 1
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y: batch_y})
                print (c)
                # mask = weights_mask['cov1'].eval(sess)
                # print((mask != 0).sum())
                # print((mask_in_grad!= 0).sum())

                with open('log/data.txt',"a") as output_file:
            		output_file.write("{},{},{}\n".format(training_cnt,train_accuracy, c))
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            if epoch % display_step == 0:
        		saver.save(sess, "tmp_20161214/prunned_model.ckpt")
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            for data_slice_two in data_slice:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
                np.savetxt(outfile, data_slice_two)

            # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')


if __name__ == '__main__':
    main()
