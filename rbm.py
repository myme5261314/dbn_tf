""" An rbm implementation for TensorFlow, based closely on the one in Theano """
import tensorflow as tf
import math


def sample_prob(probs):
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))


class RBM(object):

    def __init__(self, name, input_size, output_size):
        with tf.name_scope("rbm_" + name):
            self.weights = tf.Variable(
                tf.truncated_normal(
                    [input_size, output_size],
                    stddev=1.0 / math.sqrt(float(input_size))), name="weights")
            self.v_bias = tf.Variable(tf.zeros([input_size]), name="v_bias")
            self.h_bias = tf.Variable(tf.zeros([output_size]), name="h_bias")

    def propup(self, visible):
        return tf.nn.sigmoid(tf.matmul(visible, self.weights) + self.h_bias)

    def propdown(self, hidden):
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(self.weights)) + self.v_bias)

    def sample_h_given_v(self, v_sample):
        return sample_prob(self.propup(v_sample))

    def sample_v_given_h(self, h_sample):
        return sample_prob(self.propdown(h_sample))

    def gibbs_hvh(self, h0_sample):
        v_sample = self.sample_v_given_h(h0_sample)
        h_sample = self.sample_h_given_v(v_sample)
        return [v_sample, h_sample]

    def gibbs_vhv(self, v0_sample):
        h_sample = self.sample_h_given_v(v0_sample)
        v_sample = self.sample_v_given_h(h_sample)
        return [h_sample, v_sample]

    def cd1(self, visibles, learning_rate=0.1):
        h_start = self.propup(visibles)
        v_end = self.propdown(h_start)
        h_end = self.propup(v_end)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h_start)
        w_negative_grad = tf.matmul(tf.transpose(v_end), h_end)
        update_w = self.weights.assign_add(
            learning_rate * (w_positive_grad - w_negative_grad))
        update_vb = self.v_bias.assign_add(
            learning_rate * tf.reduce_mean(visibles - v_end, 0))
        update_hb = self.h_bias.assign_add(
            learning_rate * tf.reduce_mean(h_start - h_end, 0))
        return [update_w, update_vb, update_hb]

    def reconstruction_error(self, dataset):
        err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1])
        return tf.reduce_sum(err * err)
