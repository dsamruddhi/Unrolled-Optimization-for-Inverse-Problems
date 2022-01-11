import random
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from dataloader.load_data import Load
from dataloader.process_data import Process

from utils.plot_utils import PlotUtils


class DataLoader:

    def __init__(self):

        self.test_size = Config.config["data"]["test_size"]

    @staticmethod
    def check_data(train_input, train_output):

        plot_cmap = PlotUtils.get_cmap()
        plot_extent = PlotUtils.get_doi_extent()

        for i in random.sample(range(0, train_input.shape[0]), 5):
            print(i)
            fig, (ax1, ax3) = plt.subplots(ncols=2)

            original = ax1.imshow(train_output[i, :, :], cmap=plot_cmap, extent=plot_extent)
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Output: ground truth")

            guess_imag = ax3.imshow(train_input[i, :, :], cmap=plot_cmap, extent=plot_extent)
            fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Initial guess: imaginary")

            plt.show()

    def load_denoising(self, show_data):
        real_data = Load.get_real_data()
        real_data = np.asarray(real_data)

        real_data_train, real_data_test = Process.split_data(real_data, self.test_size)

        return real_data_train, real_data_test

    def main(self, show_data):

        X_imag = Load.get_generated_data()
        gen_data = np.asarray(X_imag)

        real_data = Load.get_real_data()
        real_data = np.asarray(real_data)

        measurements = Load.get_measurements()
        measurements = np.asarray(measurements)

        Process.check_data_sanctity([gen_data, real_data, measurements])

        gen_data_train, gen_data_test = Process.split_data(gen_data, self.test_size)
        real_data_train, real_data_test = Process.split_data(real_data, self.test_size)
        measurements_train, measurements_test = Process.split_data(measurements, self.test_size)

        print(f"Gen train: {gen_data_train.shape}, Gen test: {gen_data_test.shape} "
              f"real train: {real_data_train.shape}, real test: {real_data_test.shape}",
              f"measurements train: {measurements_train.shape}, measurements test: {measurements_test.shape}")

        if show_data:
            DataLoader.check_data(gen_data_train, real_data_train)

        return gen_data_train, gen_data_test, real_data_train, real_data_test, measurements_train, measurements_test

    @staticmethod
    def rytov_model():
        return Load.get_rytov_model()


if __name__ == '__main__':

    # To test the data pipeline
    loader = DataLoader()
    data = loader.main(show_data=True)
