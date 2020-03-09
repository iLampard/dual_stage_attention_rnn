""" Data loaders  """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataSet:
    def __init__(self, data, num_steps, do_shuffle=False):
        # num_steps = steps for x
        # Add 1 => predict one step after num_steps, i.e., T+1
        self.processed_data = self.window_rolling(data, num_steps + 1)
        self.shuffle_idx = None

        # Do a shuffle to avoid overfitting
        if do_shuffle:
            self.shuffle_idx = np.random.permutation(len(self.processed_data))
            self.processed_data = self.processed_data[self.shuffle_idx]

        # (None, num_steps, x_dim)
        self.input_x = self.processed_data[:, :-1, :-1]

        # last col is y, i.e., y_1, y_2,...y_T which will also be used in training
        # (None, num_steps, 1)
        self.input_label = self.processed_data[:, :-1, -1]
        self.input_label = self.input_label[:, :, np.newaxis]

        # (None, 1, x_dim)
        # Exogenous factors at the same time step with predicted label
        self.input_exg_x = self.processed_data[:, -1, :-1]
        self.input_exg_x = self.input_exg_x[:, np.newaxis]

        self.labels = self.processed_data[:, -1, -1]
        # (None, 1)
        self.labels = self.labels[:, np.newaxis]

        self.batch_idx = 0

    def next_batch(self, batch_size):
        """ Return batch data - x and label  """
        if self.batch_idx >= self.num_samples // batch_size:
            self.batch_idx -= self.num_samples // batch_size

        start_idx = self.batch_idx * batch_size
        end_idx = min((self.batch_idx + 1) * batch_size, self.num_samples)
        batch_x = self.input_x[start_idx: end_idx]
        batch_x_label = self.input_label[start_idx: end_idx]
        batch_x_exg = self.input_exg_x[start_idx: end_idx]
        batch_y = self.labels[start_idx: end_idx]
        yield (batch_x, batch_x_label, batch_x_exg, batch_y)
        self.batch_idx += 1

    @property
    def num_samples(self):
        """ Num of total samples """
        return len(self.labels)

    @staticmethod
    def window_rolling(data, window_size):
        """ Slice data based on window size  """
        dim = data.shape[-1]
        output = []
        for i in range(window_size):
            end = i - window_size + 1
            if end == 0:
                output.append(data[i:])
            else:
                output.append(data[i: end])

        output = np.hstack(output)

        return output.reshape([-1, window_size, dim])

    @staticmethod
    def inverse_standard_scale(val, mean, std):
        """ Inverse standard scale """
        return val * std + mean


class BaseLoader:
    name = 'baseloader'

    def __init__(self):
        self.train_data, self.valid_data, self.test_data, self.label_scaler = self.process_data()

    def load_dataset(self, num_steps, do_shuffle):
        train_dataset = DataSet(self.train_data, num_steps, do_shuffle=do_shuffle)
        valid_dataset = DataSet(self.valid_data, num_steps, do_shuffle=do_shuffle)
        test_dataset = DataSet(self.test_data, num_steps)
        return train_dataset, valid_dataset, test_dataset

    def process_data(self):
        """ To overwrite in derived class  """
        train_data = []
        valid_data = []
        test_data = []
        label_scaler = None
        return train_data, valid_data, test_data, label_scaler

    @staticmethod
    def get_loader_from_flags(dataset_name):
        loader_cls = None
        for sub_cls in BaseLoader.__subclasses__():
            if sub_cls.name == dataset_name:
                loader_cls = sub_cls

        if loader_cls is None:
            raise RuntimeError('Unknown dataset - ' + dataset_name)

        return loader_cls()

    @staticmethod
    def split_train_test(total_data, valid_start_ratio=0.8, test_start_ratio=0.9):
        """ Split the train, valid and test set """
        idx_train_end = int(valid_start_ratio * len(total_data))
        idx_valid_end = int(test_start_ratio * len(total_data))
        train_data = total_data[0: idx_train_end]
        valid_data = total_data[idx_train_end: idx_valid_end]
        test_data = total_data[idx_valid_end:]
        return train_data, valid_data, test_data

    @property
    def num_series(self):
        """ Return number of driving series """
        return self.train_data.shape[-1] - 1


class NasdaqLoader(BaseLoader):
    name = 'data_nasdaq'

    def process_data(self):
        csv_file = 'data/data_nasdaq/nasdaq100_padding.csv'
        raw_data = pd.read_csv(csv_file, header=0)

        train_data, valid_data, test_data = self.split_train_test(raw_data.values)
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)
        label_scaler = (scaler.mean_[-1], np.sqrt(scaler.var_[-1]))

        return train_data, valid_data, test_data, label_scaler
