import numpy as np


class Process:

    @staticmethod
    def check_data_sanctity(arrays):
        for array in arrays:
            assert not np.isnan(array).any()

    @staticmethod
    def split_data(array, test_size):
        test_data_len = int(len(array) * test_size)
        train_data_len = len(array) - test_data_len
        train_data, test_data = array[:train_data_len, :], array[train_data_len:, :]
        return train_data, test_data
