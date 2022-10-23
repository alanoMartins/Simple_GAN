import tensorflow as tf
from tensorflow import keras
from models import *
from tqdm import tqdm

class Trainer:

    def __init__(self):
        super(Trainer, self).__init__()
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.noise_dim = 100
        self.BATCH_SIZE = 256

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = tape.gradient(target=gen_loss, sources=self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(target=disc_loss, sources=self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return { 'generator_loss': gen_loss,  'discriminator_loss': disc_loss }

    def train(
            self,
            dataset,
            num_epochs: int,
    ):
        dataset_tqdm = tqdm(
            iterable=dataset,
            desc="Batches",
            leave=True
        )

        for epoch in tqdm(range(0, num_epochs), desc='Epochs'):
            for batch in dataset_tqdm:
                losses = self.train_step(batch)
                print(losses['discriminator_loss'])


    def predict(self):
        noise = tf.random.normal([10, self.noise_dim])
        return self.generator(noise)