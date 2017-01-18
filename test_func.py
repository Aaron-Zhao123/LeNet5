import tensorflow as tf


a = tf.Variable(10)
b = tf.multiply(a,0)
c = 4
x = a > c
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    res = sess.run(x)
    c = 20
    res = sess.run(x)
    print(res)
