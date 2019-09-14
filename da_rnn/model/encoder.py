""" Model Implementation  """

import tensorflow as tf
from tensorflow.keras import layers


class InputAttention(layers.Layer):
    def __init__(self, num_steps, reuse=tf.AUTO_REUSE):
        self.num_steps = num_steps
        with tf.variable_scope('Attention_Layer', reuse=reuse):
            self.attention_w = layers.Dense(self.num_steps, name='W')
            self.attention_u = layers.Dense(self.num_steps, name='U')
            self.attention_v = layers.Dense(1, name='V')

    def call(self, input_x, prev_state_tuple):
        """
        Compute the attention weight for input series
        hidden_state, cell_state (batch_size, hidden_dim)
        input_x (batch_size, num_series, num_steps)
        """
        prev_hidden_state, prev_cell_state = prev_state_tuple
        # (batch_size, 1, hidden_dim * 2)
        concat_state = tf.expand_dims(tf.concat([prev_hidden_state, prev_cell_state], axis=-1),
                                      axis=1)

        # (batch_size, num_series, num_steps)
        score_ = self.attention_w(concat_state) + self.attention_v(input_x)

        # (batch_size, num_series, 1)
        score = self.attention_v(tf.nn.tanh(score_))

        # (batch_size, num_series)
        weight = tf.squeeze(tf.nn.softmax(score, axis=1))

        return weight


class Encoder(layers.Model):
    def __int__(self, hidden_dim, num_steps):
        self.hidden_dim = hidden_dim
        self.attention_layer = InputAttention(num_steps)

    def call(self, inputs):
        def one_step(self, prev_state_tuple, current_input):
            """ Move along the time axis by one step  """
            hidden_state, cell_state = prev_state_tuple

            # (batch_size, num_series)
            weight = self.attention_layer(inputs, prev_state_tuple)

            weighted_current_input = weight * current_input

            # (batch_size, hidden_dim + num_series)
            concat_input = tf.concat([hidden_state, weighted_current_input], axis=-1)

            # (batch_size * 4, hidden_dim + num_series)
            concat_input_tiled = tf.tile(concat_input, [4, 1])

            forget_, input_, output_, cell_bar = tf.split(self.layer_fc(concat_input_tiled),
                                                          axis=0,
                                                          num_or_size_splits=4)

            # (batch_size, hidden_dim)
            cell_state = tf.nn.sigmoid(forget_) * cell_state + \
                         tf.nn.sigmoid(input_) * tf.nn.tanh(cell_bar)

            hidden_state = tf.nn.sigmoid(output_) * tf.nn.tanh(cell_state)

            return (hidden_state, cell_state)

        # (num_steps, batch_size, num_series)
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple = tf.scan(one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state,
                                           self.init_cell_state,
                                           0))

        # (batch_size, num_steps, hidden_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state


class Decoder(layers.Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.decoder_layer = layers.LSTM(self.hidden_dim,
                                         return_sequences=True,
                                         return_state=True)

    def call(self, inputs):
        return
