""" Main model definition  """

import numpy as np
import tensorflow as tf
from da_rnn.model.encoder_decoder import Encoder, Decoder


class DualStageRNN:
    def __init__(self, encoder_dim, decoder_dim, num_steps, num_series):
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_steps = num_steps
        self.num_series = num_series

    def build(self):
        """ Build up the network  """
        # (batch_size, num_steps, num_series)
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.num_series])

        # (batch_size, num_steps, 1)
        self.input_label = tf.placeholder(tf.float32, shape=[None, self.num_steps, 1])

        # (batch_size, 1)
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        self.lr = tf.placeholder(tf.float32)

        self.is_training = tf.placeholder(tf.bool)

        self.encoder = Encoder(self.encoder_dim, self.num_steps)

        self.decoder = Decoder(self.decoder_dim, self.num_steps)

        self.pred, self.loss = self.forward()

        self.opt_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        return

    def forward(self):
        """ Forward through time axis """

        # (batch_size, num_steps, encoder_dim)
        encoder_states = self.encoder(self.input_x)

        # (batch_size, 1)
        pred = self.decoder(encoder_states, self.input_label)

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - self.labels), axis=-1))

        return pred, loss

    def train(self, sess, batch_data, lr):
        """ Define the train process """
        batch_x, batch_y = batch_data
        feed_dict = {self.input_x: batch_x[:, :, :-1],
                     self.input_label: batch_x[:, :, -1:],
                     self.labels: batch_y,
                     self.lr: lr,
                     self.is_training: True}

        _, loss, prediction = sess.run([self.opt_op, self.loss, self.pred], feed_dict=feed_dict)

        return loss, prediction

    def predict(self, sess, batch_data):
        """ Define the prediction process """
        batch_x, batch_y = batch_data
        feed_dict = {self.input_x: batch_x[:, :, :-1],
                     self.input_label: batch_x[:, :, -1:],
                     self.labels: batch_y,
                     self.is_training: False}

        loss, prediction = sess.run([self.loss, self.pred], feed_dict=feed_dict)

        return loss, prediction
