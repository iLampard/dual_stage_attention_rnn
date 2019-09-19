""" Metrics utilities """

import numpy as np


class Metrics:
    """ Metrics to evaluate prediction performance """

    def __init__(self):
        pass

    def get_metrics_dict(self, predictions, labels):
        """ Return the metrics result in dict """
        res = dict()

        res['rmse'] = self.rmse(predictions, labels)
        res['mae'] = self.mae(predictions, labels)

        return res

    @staticmethod
    def rmse(predictions, labels):
        """ RMSE ratio """
        return np.sqrt(np.mean(np.subtract(predictions, labels) ** 2))

    @staticmethod
    def mae(predictions, labels):
        """ MAE ratio """
        return np.mean(np.abs(predictions - labels))

    @staticmethod
    def metrics_dict_to_str(metrics_dict):
        """ Convert metrics to a string to show in the console """
        eval_info = ''
        for key, value in metrics_dict.items():
            eval_info += '{0} : {1}'.format(key, value)

        return eval_info[:-1]
