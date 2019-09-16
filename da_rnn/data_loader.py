


class BaseLoader:
    def __init__(self):
        self.train_data, self.valid_data, self.test_data = self.process_data()

    def process_data(self):
        """ To overwrite in derived class  """
        train_data = []
        valid_data = []
        test_data = []
        return train_data, valid_data, test_data


class NasdaqLoader(BaseLoader):
    name = 'nasdaq'

    def process_data(self):


        return