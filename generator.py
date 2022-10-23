import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

class Generator(keras.Model):

    def __init__(self, random_noise_size=100):
        super().__init__(name='generator')
        # layers
        self.input_layer = keras.layers.Dense(units=random_noise_size)
        self.dense_1 = keras.layers.Dense(units=128)
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_2 = keras.layers.Dense(units=128)
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_3 = keras.layers.Dense(units=256)
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.01)
        self.output_layer = keras.layers.Dense(units=784, activation="tanh")

    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        return self.output_layer(x)

    def generate_noise(self, batch_size, random_noise_size):
        return np.random.uniform(-1, 1, size=(batch_size, random_noise_size))

    