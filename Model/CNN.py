import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from keras import Sequential
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras.optimizers import Adam


class CNN:
    def __init__(self, filters: int, kernel_size: int, activation: str, x_data, y_data, valid_data, epoch, batch_size):
        self.model = Sequential()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.x_data = x_data
        self.y_data = y_data
        self.valid_data = valid_data
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.x_data) // batch_size
        self.input_shape = x_data.shape[-2:]
        self.cyclic_lr = self.set_optimizer()
        self.build_model()
        self.optimizer = Adam(learning_rate=self.cyclic_lr, amsgrad=True)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def set_optimizer(self):
        return tfa.optimizers.CyclicalLearningRate(initial_learning_rate=4e-07, maximal_learning_rate=1e-03,
                                                   scale_fn=lambda x: 1 / (2 ** (x - 1)),
                                                   step_size=6 * self.steps_per_epoch)

    def plot_learning_rate(self):
        """
        Plot Cyclical Learning Rate
        """
        step = np.arange(0, self.epoch * self.steps_per_epoch)
        lr = self.cyclic_lr(step)
        plt.plot(step, lr)
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Shape of Cyclical Learning Rate")
        plt.show()

    def build_model(self):
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                              activation=self.activation, input_shape=self.input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation=self.activation))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1))

    def compile_model(self, loss):
        self.model.compile(self.optimizer, loss)

    def summary_model(self):
        self.model.summary()

    def train_model(self):
        return self.model.fit(self.x_data, self.y_data, validation_data=self.valid_data, epochs=self.epoch,
                              batch_size=self.batch_size, callbacks=[self.callback])

    def predict_model(self, predict_x):
        return self.model.predict(predict_x)
