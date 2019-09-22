""" Scaler class to scale data  """

import numpy as np


class BaseScaler:
    def __init__(self, scale_range):
        self.scale_min, self.scale_max = scale_range[0], scale_range[1]
        self.scalers = None

    def fit(self, seqs):
        return

    def transform(self, seqs):
        return seqs

    def inverse_transform(self, scaled_seqs):
        return scaled_seqs

    def fit_transform(self, seqs):
        self.fit(seqs)
        return self.transform(seqs)


class ZeroMaxScaler(BaseScaler):
    def __init__(self, scale_range):
        super(ZeroMaxScaler, self).__init__(scale_range)

    def fit(self, seqs):
        """ seqs - 2-dim array or list of list  """

        if self.scalers is not None:
            return

        num_seqs = len(seqs)
        # an array of [min_val, max_val]
        self.scalers = np.zeros([num_seqs, 2])

        for i in range(num_seqs):
            seq = np.array(seqs[i])
            _, max_val = self.get_min_max_val(seq)
            self.scalers[i] = [0.0, max_val]

    def transform(self, seqs):
        """ Scale all the sequences """
        return [self.transform_seq(seq, i) for i, seq in enumerate(seqs)]

    def transform_seq(self, seq, seq_idx):
        """ Scale single sequence """
        min_val = self.scalers[seq_idx][0]
        max_val = self.scalers[seq_idx][1]
        if min_val == max_val:
            raise RuntimeError('the seq with zero min and max value is unable to scale')
        else:
            return (seq - min_val) / (max_val - min_val) * (self.scale_max - self.scale_min)

    def get_min_max_val(self, seq):
        """ Compute min and max without considering outliers """
        std_mul = 3
        mean_val = np.mean(seq)
        std_val = np.val(seq)

        mask = (seq <= mean_val + std_mul * std_val) & (seq >= mean_val - std_mul * std_val)
        seq_mask = seq[mask]
        min_val = np.min(seq_mask)
        max_val = np.max(seq_mask)

        return min_val, max_val

    def inverse_transform(self, scaled_seqs):
        """ Scale-back all the sequences """
        return [self.inverse_transform_seq(seq, i) for i, seq in enumerate(scaled_seqs)]

    def inverse_transform_seq(self, scaled_seq, seq_idx):
        """ Scale-back single sequence """
        min_val = self.scalers[seq_idx][0]
        max_val = self.scalers[seq_idx][1]
        return (scaled_seq - self.scale_min) / (self.scale_max - self.scale_min) * \
               (max_val - min_val) + min_val
