import os
import numpy as np
from scipy.io import loadmat

from config import Config


class Load:

    gen_path = Config.config["data"]["gen_data_path"]
    real_path = Config.config["data"]["real_data_path"]
    total_power_path = Config.config["data"]["total_power_path"]
    direct_power_path = Config.config["data"]["direct_power_path"]

    num_samples = Config.config["data"]['num_samples']

    @staticmethod
    def get_files(filepath):
        files = os.listdir(filepath)
        files.sort(key=lambda x: int(x.strip(".mat")))
        return files

    @staticmethod
    def get_generated_data():
        imag_data = []
        files = Load.get_files(Load.gen_path)
        num_files = Load.num_samples if Load.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(Load.gen_path, file)
            guess = loadmat(filename)["guess"]
            imag_data.append(guess[0][0][0])
        return imag_data

    @staticmethod
    def get_real_data():
        scatterers = []
        files = Load.get_files(Load.real_path)
        num_files = Load.num_samples if Load.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(Load.real_path, file)
            scatterer = loadmat(filename)["scatterer"]
            scatterers.append(np.real(scatterer))
        return scatterers

    @staticmethod
    def get_direct_power():
        filename = os.path.join(Load.direct_power_path, "direct_power")
        direct_power = loadmat(filename)["direct_power"]
        return direct_power

    @staticmethod
    def get_total_power():
        powers = []
        files = Load.get_files(Load.total_power_path)
        num_files = Load.num_samples if Load.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(Load.total_power_path, file)
            total_power = loadmat(filename)["total_power"]
            powers.append(np.real(total_power))
        return powers

    @staticmethod
    def get_measurements():

        def _get_rytov_data(total_power, direct_power):
            data = (total_power - direct_power) / (10 * np.log10(np.exp(2)))
            data = data.reshape(data.size, order='F')
            return data

        direct_power = Load.get_direct_power()
        total_powers = Load.get_total_power()
        measurements = [_get_rytov_data(power, direct_power) for power in total_powers]
        return measurements

    @staticmethod
    def get_rytov_model():
        A = loadmat(r"A_imag.mat")
        A = A["A_imag"]
        return A
