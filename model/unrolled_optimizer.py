from abc import ABC
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, BatchNormalization, Input, MaxPooling2D, Flatten, LeakyReLU

from config import Config
from base_model import BaseModel
from layers.prox_l1_layer import ProximalL1Layer
from dataloader.data_loader import DataLoader
from utils.plot_utils import PlotUtils
from metric_functions.metrics import Metrics


class UnrolledOptimization(BaseModel, ABC):

    def __init__(self):
        super().__init__()

        # Data and its attributes
        self.train_dataset = None
        self.test_dataset = None

        self.data_generator = ImageDataGenerator()

        # Model and its attributes
        self.model_path = Config.config["model"]["model_path"]
        self.experiment_name = Config.config["model"]["experiment_name"]
        self.model = None
        self.optimizer = None

        self.eta = 0.01
        self.num_iters = 10

        # Training
        self.batch_size = Config.config["train"]["batch_size"]

        # Logging
        self.file_writer = None

    def load_data(self, show_data=False):
        gen_data_train, gen_data_test,\
            real_data_train, real_data_test, \
            measurements_train, measurements_test = DataLoader().main(show_data)

        train_dataset = tf.data.Dataset.from_tensor_slices((gen_data_train, real_data_train, measurements_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((gen_data_test, real_data_test, measurements_test))

        self.train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True)
        self.test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)

        self.A = DataLoader.rytov_model()

    def build(self):

        def _create_model():
            X = tf.keras.layers.Input((2500, 1), dtype=tf.float32)
            Y = tf.keras.layers.Input((1560, 1), dtype=tf.float32)
            X_, Y_ = X, Y
            x_recs = []
            for iter in range(0, self.num_iters):
                X_, Y_ = ProximalL1Layer(self.A, self.eta)([X_, Y_])
                x_recs.append(X_)
            model = tf.keras.Model(inputs=[X, Y], outputs=[X_, Y_])
            return model

        self.model = _create_model()
        print(self.model.summary())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    def log(self):
        log_dir = os.path.join(Config.config["model"]["model_path"], "logs")
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                         Config.config['model']['experiment_name'],
                                                                         datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self, steps):

        train_ds = self.train_dataset.repeat(100).as_numpy_iterator()

        for step in tf.range(steps):
            tf.print("Step: ", step)
            step = tf.cast(step, tf.int64)

            gen_batch, real_batch, measurement_batch = train_ds.next()
            gen_batch[gen_batch < 0] = 0
            gen_batch = gen_batch.reshape((gen_batch.shape[0], 2500, 1))
            real_batch = real_batch.reshape((real_batch.shape[0], 2500, 1))

            with tf.GradientTape() as tape:
                out_batch, _ = self.model([gen_batch, measurement_batch])
                loss = tf.reduce_mean(tf.square(out_batch - real_batch))

            network_gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(network_gradients, self.model.trainable_variables))

            tf.print("loss value: ", loss)

            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)

        print("Trainable variables: ", self.model.trainable_variables)

    def evaluate(self, index):

        train_ds = self.train_dataset.repeat(1).as_numpy_iterator()

        gen_batch, real_batch, measurement_batch = train_ds.next()
        gen_batch[gen_batch < 0] = 0

        gen_batch = gen_batch.reshape((gen_batch.shape[0], 2500, 1))
        real_batch = real_batch.reshape((real_batch.shape[0], 2500, 1))

        out_batch, _ = self.model([gen_batch, measurement_batch])

        gt = real_batch[index].reshape((50, 50), order='F')
        start = gen_batch[index].reshape((50, 50), order='F')
        current = np.asarray(out_batch[index]).reshape((50, 50), order='F')

        psnr_start = Metrics.psnr(np.asarray(gt), np.asarray(start))
        psnr_current = Metrics.psnr(np.asarray(gt), np.asarray(current))

        PlotUtils.plot_output(gt, start, current, psnr_start, psnr_current)


if __name__ == '__main__':

    """ TF / GPU config """
    tf.random.set_seed(1234)
    tf.keras.backend.clear_session()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = InteractiveSession(config=config)

    experiment = UnrolledOptimization()
    experiment.load_data(show_data=False)
    experiment.build()
    experiment.log()
    experiment.train(100)
    index = 6
    experiment.evaluate(index)
