""" A script to run the model """

import os
import sys

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from absl import app
from absl import flags
from tensorflow.examples.tutorials.mnist import input_data
from da_rnn.model_runner import ModelRunner

FLAGS = flags.FLAGS

# model runner_params
flags.DEFINE_bool('write_summary', False, 'Whether to write summary of epoch in training using Tensorboard')
flags.DEFINE_integer('max_epoch', 10, 'Max epoch number of training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 128, 'Batch size of data fed into model')

# model params
flags.DEFINE_bool('batch_norm', True, 'Whether to apply batch normalization')
flags.DEFINE_integer('rnn_dim', 32, 'Dimension of LSTM cell')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate of the model')
flags.DEFINE_string('save_dir', 'logs', 'Root path to save logs and models')


def main(argv):
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_set, valid_set, test_set = dataset.train, dataset.validation, dataset.test

    model = ClassificationModel(hidden_dim=FLAGS.rnn_dim,
                                input_x_dim=FLAGS.input_x_dim,
                                num_class=10,
                                apply_bn=FLAGS.batch_norm)
    model_runner = ModelRunner(model, FLAGS)

    model_runner.train(train_set, valid_set, test_set, FLAGS.max_epoch)

    return


if __name__ == '__main__':
    app.run(main)
