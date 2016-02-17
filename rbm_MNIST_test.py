import tensorflow as tf
import numpy as np
import rbm
import input_data


def build_model(X, w1, b1, wo, bo):
    h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
    model = tf.nn.sigmoid(tf.matmul(h1, wo) + bo)
    return model


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))


def init_bias(dim):
    return tf.Variable(tf.zeros([dim]))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_layer = rbm.RBM("mnist", 784, 500)

trX_constant = tf.constant(trX)
# for i in range(1):
# print "RBM CD: ", i
# rbm_layer.cd1(trX_constant)

rbm_w, rbm_vb, rbm_hb = rbm_layer.cd1(trX_constant)


wo = init_weight([500, 10])
bo = init_bias(10)
py_x = build_model(X, rbm_w, rbm_hb, wo, bo)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(10):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
