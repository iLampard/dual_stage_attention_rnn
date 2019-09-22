""" A script to run the model """

import os
import sys

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from absl import app
from absl import flags
from da_rnn.model import DualStageRNN
from da_rnn.model_runner import ModelRunner
from da_rnn.data_loader import BaseLoader

FLAGS = flags.FLAGS

# Data input params
flags.DEFINE_string('data_set', 'data_nasdaq', 'Source data set for training')

# Model runner params
flags.DEFINE_bool('write_summary', False, 'Whether to write summary of epoch in training using Tensorboard')
flags.DEFINE_integer('max_epoch', 300, 'Max epoch number of training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 128, 'Batch size of data fed into model')
flags.DEFINE_bool('plot_prediction', False, 'Whether to plot predictions after model evaluation')

# Model params
flags.DEFINE_bool('use_cur_exg', False, 'Whether to use current exogenous factor for prediction')
flags.DEFINE_bool('shuffle_train', False, 'Whether to shuffle the training set to avoid overfitting')
flags.DEFINE_integer('encoder_dim', 32, 'Dimension of encoder LSTM cell')
flags.DEFINE_integer('decoder_dim', 32, 'Dimension of decoder LSTM cell')
flags.DEFINE_integer('num_steps', 10, 'Num of time steps for input x data')
flags.DEFINE_string('save_dir', 'logs', 'Root path to save logs and models')


def main(argv):
    data_loader = BaseLoader.get_loader_from_flags(FLAGS.data_set)

    train_set, valid_set, test_set = data_loader.load_dataset(FLAGS.num_steps, FLAGS.shuffle_train)

    model = DualStageRNN(encoder_dim=FLAGS.encoder_dim,
                         decoder_dim=FLAGS.decoder_dim,
                         num_steps=FLAGS.num_steps,
                         num_series=data_loader.num_series,
                         use_cur_exg=FLAGS.use_cur_exg)
    model_runner = ModelRunner(model, data_loader.label_scaler, FLAGS)

    model_runner.train(train_set, valid_set, test_set, FLAGS.max_epoch)

    model_runner.evaluate(test_set, plot=FLAGS.plot_prediction)

    return


if __name__ == '__main__':
    app.run(main)
