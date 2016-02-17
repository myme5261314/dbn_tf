import tensorflow as tf
import numpy as np
import rbm
import input_data
import math
import Image
from util import tile_raster_images


def build_model(X, w1, b1, wo, bo):
    h1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
    model = tf.nn.sigmoid(tf.matmul(h1, wo) + bo)
    return model


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))


def init_bias(dim):
    return tf.Variable(tf.zeros([dim]))

alpha = 1.0

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_w = tf.Variable(tf.zeros([784, 500]), name="rbm_w")
rbm_vb = tf.Variable(tf.zeros([784]), name="rbm_vb")
rbm_hb = tf.Variable(tf.zeros([500]), name="rbm_hb")
h0 = rbm.sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb, name="h0"))
v1 = rbm.sample_prob(tf.nn.sigmoid(
    tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb, name="v1"))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb, name="h1")
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad)
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

dataset = tf.placeholder("float", [None, 784])
h = tf.nn.sigmoid(tf.matmul(dataset, rbm_w) + rbm_hb)
h_sample = tf.nn.relu(tf.sign(h - tf.random_uniform(tf.shape(h))))
v = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb)
v_sample = tf.nn.relu(tf.sign(v - tf.random_uniform(tf.shape(v))))
err = dataset - v_sample
err_sum = tf.reduce_mean(err * err)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


print "Initialize Success!!!!!!"
print sess.run(err_sum, feed_dict={dataset: trX})

for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
    # sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    batch = trX[start:end]
    r_w = sess.run(update_w, feed_dict={X: batch})
    r_vb = sess.run(update_vb, feed_dict={X: batch})
    r_hb = sess.run(update_hb, feed_dict={X: batch})
    sess.run(rbm_w.assign(r_w))
    sess.run(rbm_vb.assign(r_vb))
    sess.run(rbm_hb.assign(r_hb))
    # print "Reconstruction Error is: ", sess.run(err_sum)
    if start % 10000 == 0:
        print sess.run(err_sum, feed_dict={dataset: trX})

        image = Image.fromarray(
            tile_raster_images(
                X=r_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("rbm_%d.png" % (start / 10000))


w0 = tf.Variable(r_w)
b0 = tf.Variable(r_hb)
a1 = tf.nn.sigmoid(tf.matmul(X, w0) + b0)
w1 = tf.Variable(
    (tf.random_uniform([500, 10]) - 0.5) * 8 * math.sqrt(6 / (500 + 10)))
b1 = tf.Variable(tf.zeros([10]))
a2 = tf.nn.sigmoid(tf.matmul(a1, w1) + b1)

# rbm_layer = rbm.RBM("mnist", 784, 500)

# trX_constant = tf.constant(trX)
# for i in range(1):
# print "RBM CD: ", i
# rbm_layer.cd1(trX_constant)

# rbm_w, rbm_vb, rbm_hb = rbm_layer.cd1(trX_constant)


# wo = init_weight([500, 10])
# bo = init_bias(10)
# py_x = build_model(X, rbm_w, rbm_hb, wo, bo)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a2, Y))
train_op = tf.train.GradientDescentOptimizer(1.0).minimize(cost)
predict_op = tf.argmax(a2, 1)

# sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if start % 10000 == 0:
            print i, np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX, Y: teY}))
