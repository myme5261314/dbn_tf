import tensorflow as tf
import input_data
import Image
from util import tile_raster_images


def sample_prob(probs):
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))

alpha = 1.0
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_w = tf.Variable(tf.zeros([784, 500]), name="rbm_w")
rbm_vb = tf.Variable(tf.zeros([784]), name="rbm_vb")
rbm_hb = tf.Variable(tf.zeros([500]), name="rbm_hb")
h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb, name="h0"))
v1 = sample_prob(tf.nn.sigmoid(
    tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb, name="v1"))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb, name="h1")
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad)
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

h = tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb)
h_sample = tf.nn.relu(tf.sign(h - tf.random_uniform(tf.shape(h))))
v = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb)
v_sample = tf.nn.relu(tf.sign(v - tf.random_uniform(tf.shape(v))))
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print sess.run(err_sum, feed_dict={X: trX})

for start, end in zip(
        range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    n_w = sess.run(update_w, feed_dict={X: batch})
    n_vb = sess.run(update_vb, feed_dict={X: batch})
    n_hb = sess.run(update_hb, feed_dict={X: batch})
    sess.run(rbm_w.assign(n_w))
    sess.run(rbm_vb.assign(n_vb))
    sess.run(rbm_hb.assign(n_hb))
    # print "Reconstruction Error is: ", sess.run(err_sum)
    if start % 10000 == 0:
        print sess.run(err_sum, feed_dict={X: trX})

        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("rbm_%d.png" % (start / 10000))
