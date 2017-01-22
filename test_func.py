import tensorflow as tf


a = tf.Variable(10.1234)
e = tf.Variable(2.0)
b = tf.multiply(a,0)

lower_bound = tf.constant(
    [[ 0.,   0.,   0. ],
     [ 0.,   0.,   0. ],
     [ 1.0,  0.5,  0.5],
     [ 0.5,  0.5,  0.5],])
inter = tf.equal(lower_bound, 1)
inter = tf.to_float(inter)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    res = sess.run(inter)
    print(res)
