import tensorflow as tf

a = tf.placeholder(tf.float32, shape = [3], name = 'a')
b = tf.Variable(3, dtype = tf.float32, name = 'b')
x = tf.add(a, b, name = 'add');

writer = tf.summary.FileWriter('./graphs/lecture02/lazy', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        print(sess.run(tf.add(a, b), {a: [1, 2, 3]}));
    print(sess.run(x, {a:[1,2,3]}));

writer.close()