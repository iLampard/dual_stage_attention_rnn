""" Data loaders  """
import pandas as pd


class DataSet:
    def __init__(self, input_x, labels):
        self.input_x = input_x
        self.labels = labels
        self.batch_idx = 0
        assert len(self.input_x) == len(self.labels), 'len of input x must match that of labels'

    def next_batch(self, batch_size):
        if self.batch_idx >= self.num_samples // batch_size:
            self.batch_idx = self.batch_idx - self.num_samples // batch_size
        batch_x = self.input_x[self.batch_idx: self.batch_idx + batch_size]
        batch_y = self.labels[self.batch_idx: self.batch_idx + batch_size]
        yield (batch_x, batch_y)
        self.batch_idx += 1

    @property
    def num_samples(self):
        return len(self.labels)


class BaseLoader:
    def __init__(self):
        self.train_data, self.valid_data, self.test_data = self.process_data()

    def load_dataset(self):
        train_dataset = DataSet(self.train_data[:, :-1], self.train_data[:, -1])
        valid_dataset = DataSet(self.valid_data[:, :-1], self.valid_data[:, -1])
        test_dataset = DataSet(self.test_data[:, :-1], self.test_data[:, -1])
        return train_dataset, valid_dataset, test_dataset

    def process_data(self):
        """ To overwrite in derived class  """
        train_data = []
        valid_data = []
        test_data = []
        return train_data, valid_data, test_data

    @staticmethod
    def split_train_test(total_data, valid_start_ratio=0.8, test_start_ratio=0.9):
        """ Split the train, valid and test set """
        idx_train_end = int(valid_start_ratio * len(total_data))
        idx_valid_end = int(test_start_ratio * len(total_data))
        train_data = total_data[0: idx_train_end]
        valid_data = total_data[idx_train_end: idx_valid_end]
        test_data = total_data[idx_valid_end:]
        return train_data, valid_data, test_data


class NasdaqLoader(BaseLoader):
    name = 'nasdaq'

    def process_data(self):
        csv_file = 'data/data_nasdaq/nasdaq100_padding.csv'
        raw_data = pd.read_csv(csv_file, header=0)

        train_data, valid_data, test_data = self.split_train_test(raw_data.values)

        return train_data, valid_data, test_data
