""" Model Implementation  """

import tensorflow as tf
from tensorflow.keras import layers


class Attention(layers.Layer):
    def __init__(self, cell_units, reuse=tf.AUTO_REUSE):
        self.cell_units = cell_units
        with tf.variable_scope('Attention_Layer', reuse=reuse):
            self.attention_w1 = layers.Dense(self.cell_units, name='W1')
            self.attention_w2 = layers.Dense(self.cell_units, name='W2')
            self.attention_v = layers.Dense(self.cell_units, name='V')

    def call(self, decoder_state, encoder_states, pos_mask=None):
        return


class Encoder(layers.Model):
    def __int__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.encoder_layer = layers.LSTM(self.hidden_dim,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, inputs):
        return


class Decoder(layers.Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.decoder_layer = layers.LSTM(self.hidden_dim,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, inputs):
        return
