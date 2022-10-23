import tensorflow as tf
from tensorflow import keras
from models import *


class GAN(keras.Model):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.noise_dim = 100
        self.BATCH_SIZE = 256



    def random_sample(self):
        noise = tf.random.normal([10, self.noise_dim])
        generated_images = self.generator(noise, training=False)
        return generated_images

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"generator_loss": gen_loss, "discriminator_loss": disc_loss}

    @tf.function
    def test_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_images = self.generator(noise, training=False)

        real_output = self.discriminator(images, training=False)
        fake_output = self.discriminator(generated_images, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        return {"generator_loss": gen_loss, "discriminator_loss": disc_loss}