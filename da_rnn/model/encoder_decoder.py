""" Model Implementation  """

import tensorflow as tf
from tensorflow.keras import layers


class Attention(layers.Model):
    def __init__(self, input_dim, var_scope, reuse=tf.AUTO_REUSE):
        self.input_dim = input_dim
        with tf.variable_scope(var_scope, reuse=reuse):
            self.attention_w = layers.Dense(self.input_dim, name='W')
            self.attention_u = layers.Dense(self.input_dim, name='U')
            self.attention_v = layers.Dense(1, name='V')

    def call(self, input_x, prev_state_tuple):
        """
        Compute the attention weight for input series
        hidden_state, cell_state (batch_size, hidden_dim)
        input_x (batch_size, num_series, input_dim),
        input_dim = num_steps for input attention
        """
        prev_hidden_state, prev_cell_state = prev_state_tuple
        # (batch_size, 1, hidden_dim * 2)
        concat_state = tf.expand_dims(tf.concat([prev_hidden_state, prev_cell_state], axis=-1),
                                      axis=1)

        # (batch_size, num_series, input_dim)
        score_ = self.attention_w(concat_state) + self.attention_u(input_x)

        # (batch_size, num_series, 1)
        # Equation (8)
        score = self.attention_v(tf.nn.tanh(score_))

        # (batch_size, num_series)
        # Equation (9)
        weight = tf.squeeze(tf.nn.softmax(score, axis=1), axis=-1)

        return weight


class LSTMCell(layers.Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.layer_fc = layers.Dense(self.hidden_dim)

    def call(self, input_x, prev_state_tuple):
        """ Return next step's hidden state and cell state  """
        hidden_state, cell_state = prev_state_tuple

        # (batch_size, hidden_dim + input_dim)
        concat_input = tf.concat([hidden_state, input_x], axis=-1)

        # (batch_size * 4, hidden_dim + input_dim)
        concat_input_tiled = tf.tile(concat_input, [4, 1])

        # Equation (3) - (6) without activation
        forget_, input_, output_, cell_bar = tf.split(self.layer_fc(concat_input_tiled),
                                                      axis=0,
                                                      num_or_size_splits=4)

        # (batch_size, hidden_dim)
        # Equation (6)
        cell_state = tf.nn.sigmoid(forget_) * cell_state + \
                     tf.nn.sigmoid(input_) * tf.nn.tanh(cell_bar)

        # Equation (7)
        hidden_state = tf.nn.sigmoid(output_) * tf.nn.tanh(cell_state)
        return (hidden_state, cell_state)


class Encoder(layers.Model):
    def __int__(self, encoder_dim, num_steps):
        self.encoder_dim = encoder_dim
        self.attention_layer = Attention(num_steps, var_scope='input_attention')
        self.lstm = LSTMCell(encoder_dim)

    def call(self, inputs):
        """
        inputs: (batch_size, num_steps, num_series)
        """

        def one_step(self, prev_state_tuple, current_input):
            """ Move along the time axis by one step  """

            # (batch_size, num_series)
            weight = self.attention_layer(inputs, prev_state_tuple)

            weighted_current_input = weight * current_input

            return self.LSTMCell(weighted_current_input, prev_state_tuple)

        # Get the batch size from inputs
        self.batch_size = tf.shape(inputs)[0]
        self.num_steps = inputs.get_shape().as_list()[1]

        self.init_hidden_state = tf.random_normal([self.batch_size, self.hidden_dim])
        self.init_cell_state = tf.random_normal([self.batch_size, self.hidden_dim])

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
    def __init__(self, decoder_dim, num_steps):
        self.decoder_dim = decoder_dim
        self.attention_layer = Attention(num_steps, var_scope='temporal_attention')
        self.lstm = LSTMCell(decoder_dim)
        self.layer_prediction_fc_1 = layers.Dense(decoder_dim)
        self.layer_prediction_fc_2 = layers.Dense(1)

    def call(self, encoder_states, labels):
        """
        encoder_states: (batch_size, num_steps, encoder_dim)
        labels: (batch_size, num_steps)
        """

        def one_step(self, accumulator, current_label):
            """ Move along the time axis by one step  """

            prev_state_tuple, context = accumulator
            # (batch_size, num_steps)
            # Equation (12) (13)
            weight = self.attention_layer(encoder_states, prev_state_tuple)

            # Equation (14)
            # (batch_size, encoder_dim)
            context = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * encoder_states,
                                    axis=1)

            # Equation (15)
            # (batch_size, 1)
            y_tilde = self.layers_fc_context(tf.concat([current_label, context], axis=-1))

            # Equation (16)
            return self.LSTMCell(y_tilde, prev_state_tuple), context

        # Get the batch size from inputs
        self.batch_size = tf.shape(encoder_states)[0]
        self.num_steps = encoder_states.get_shape().as_list()[1]

        init_hidden_state = tf.random_normal([self.batch_size, self.decoder_dim])
        init_cell_state = tf.random_normal([self.batch_size, self.decoder_dim])

        # (num_steps, batch_size, num_series)
        inputs_ = tf.transpose(encoder_states, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple, all_context = tf.scan(one_step,
                                           elems=inputs_,
                                           initializer=(init_hidden_state,
                                                        init_cell_state,
                                                        None))

        # (batch_size, num_steps, decoder_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])

        # (batch_size, num_steps, encoder_dim)
        all_context = tf.transpose(state_tuple[0], perm=[1, 0, 2])

        last_hidden_state = all_hidden_state[:, -1, :]
        last_context = all_context[:, -1, :]

        # (batch_size, 1)
        # Equation (22)
        pred_ = self.layer_prediction_fc_1(tf.concat([last_hidden_state, last_context], axis=-1))
        pred = self.layer_prediction_fc_2(pred_)

        return pred
