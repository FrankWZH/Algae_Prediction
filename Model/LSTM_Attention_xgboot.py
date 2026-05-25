import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from keras.models import Model
from keras.layers import Dense, Flatten, Masking, Input, dot
import keras.layers as layers
from keras.optimizers import Adam


class LSTM_Attention:
    def __init__(self, input_unit: int, hist_size: int, activation: str, x_data, y_data, valid_data, epoch, batch_size):
        self.model = Model()
        self.input_unit = input_unit
        self.hist_size = hist_size
        self.activation = activation
        self.x_data = x_data
        self.y_data = y_data
        self.valid_data = valid_data
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.x_data) // batch_size
        self.input_shape = x_data.shape[2]
        self.cyclic_lr = self.set_optimizer()
        self.build_model()
        self.optimizer = Adam(learning_rate=self.cyclic_lr, amsgrad=True)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def set_optimizer(self):
        return tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-06, maximal_learning_rate=1e-03,
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
        inputs = Input(shape=(self.hist_size, self.input_shape))
        # By masking the zeros, the model can learn to ignore the missing values and focus on the valid data.
        masked = Masking(mask_value=0.)(inputs)
        lstm = layers.LSTM(self.input_unit, return_sequences=True)(masked)
        attention = dot([lstm, lstm], axes=[2, 2])
        # extracting weight for every observation in the history size!
        attention = Dense(self.hist_size, activation='softmax')(attention)
        # assinging weight to lstm by dot product
        context = dot([attention, lstm], axes=[2, 1])
        flattened = Flatten()(context)
        output = Dense(1)(flattened)
        self.model = Model(inputs=inputs, outputs=output)

    def compile_model(self, loss):
        self.model.compile(self.optimizer, loss)

    def summary_model(self):
        self.model.summary()

    def train_model(self):
        return self.model.fit(self.x_data, self.y_data, validation_data=self.valid_data, epochs=self.epoch,
                              batch_size=self.batch_size, callbacks=[self.callback])

    def predict_model(self, predict_x):
        return self.model.predict(predict_x)
