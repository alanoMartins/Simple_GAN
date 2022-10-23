import glob
from gan import GAN
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from train import Trainer

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def run():
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    LOG_DIR = 'tensorboard'
    CHECKPOINT_DIR = './checkpoints/'

    # Initialize callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        monitor='generator_loss',
        mode='min',
        save_freq=5,
        save_best_only=True)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = np.expand_dims(train_images, -1).astype("float32")
    test_images = np.expand_dims(test_images, -1).astype("float32")

    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2],
                                        train_images.shape[3]).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],
                                      test_images.shape[3]).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    trainer = Trainer()
    trainer.train(train_dataset,10)

    images = trainer.predict()

    for i in images:
        plt.imshow(i)
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
