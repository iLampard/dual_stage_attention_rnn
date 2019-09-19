""" Data loaders  """
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataSet:
    def __init__(self, data, num_steps):
        # num_steps = steps for x
        # Add 1 => predict one step after num_steps, i.e., T+1
        self.processed_data = self.window_rolling(data, num_steps + 1)

        # (None, num_steps, x_dim + 1)
        # x_dim + 1 because last col is y, i.e., y_1, y_2,...y_T which will also be used in training
        self.input_x = self.processed_data[:, :-1, :]

        self.labels = self.processed_data[:, -1, -1]
        # (None, 1)
        self.labels = self.labels[:, np.newaxis]

        self.batch_idx = 0

    def next_batch(self, batch_size):
        """ Return batch data - x and label  """
        if self.batch_idx >= self.num_samples // batch_size:
            self.batch_idx = self.batch_idx - self.num_samples // batch_size
        batch_x = self.input_x[self.batch_idx: self.batch_idx + batch_size]
        batch_y = self.labels[self.batch_idx: self.batch_idx + batch_size]
        yield (batch_x, batch_y)
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


class BaseLoader:
    name = 'baseloader'

    def __init__(self):
        self.train_data, self.valid_data, self.test_data, self.scaler = self.process_data()

    def load_dataset(self, num_steps):
        train_dataset = DataSet(self.train_data, num_steps)
        valid_dataset = DataSet(self.valid_data, num_steps)
        test_dataset = DataSet(self.test_data, num_steps)
        return train_dataset, valid_dataset, test_dataset

    def process_data(self):
        """ To overwrite in derived class  """
        train_data = []
        valid_data = []
        test_data = []
        return train_data, valid_data, test_data

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
        scaler = MinMaxScaler((0, 100))
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)

        return train_data, valid_data, test_data, scaler
