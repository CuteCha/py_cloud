import tensorflow.keras as keras
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio

BS = 64
CHANNELS = 1
CLASSES = 10
IMSIZE = 28
GENDIM = 128


def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = keras.utils.to_categorical(all_labels, 10)

    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(BS)

    print(f"Shape of training images: {all_digits.shape}")
    print(f"Shape of training labels: {all_labels.shape}")

    return dataset


def build_discriminator(channels):
    model = keras.Sequential([
        keras.layers.InputLayer((28, 28, channels)),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1)],
        name="discriminator")

    return model


def build_generator(channels):
    model = keras.Sequential([
        keras.layers.InputLayer((channels,)),
        keras.layers.Dense(7 * 7 * channels),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Reshape((7, 7, channels)),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")],
        name="generator")

    return model


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, gen_input_dim):
        super().__init__()
        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = gen_input_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_images, one_hot_labels = data
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[IMSIZE * IMSIZE]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, IMSIZE, IMSIZE, CLASSES)
        )

        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        generated_images = self.generator(random_vector_labels)

        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


def interpolate_class(generator, first_number, second_number, num_interpolation, interpolation_noise):
    first_label = keras.utils.to_categorical([first_number], CLASSES)
    second_label = keras.utils.to_categorical([second_number], CLASSES)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (first_label * (1 - percent_second_label) + second_label * percent_second_label)

    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = generator.predict(noise_and_labels)
    return fake


def gen_image(generator, start_class, end_class, num_interpolation, interpolation_noise):
    fake_images = interpolate_class(generator, start_class, end_class, num_interpolation, interpolation_noise)
    fake_images *= 255.0
    converted_images = fake_images.astype(np.uint8)
    converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
    imageio.mimsave("./logs/animation00.gif", converted_images, fps=1)
    # embed.embed_file("./logs/animation00.gif")


def inference():
    generator = keras.models.load_model("./logs/cond_gan_gen.h5")
    num_interpolation = 9
    interpolation_noise = tf.random.normal(shape=(1, GENDIM))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, GENDIM))

    start_class = 1
    end_class = 5
    gen_image(generator, start_class, end_class, num_interpolation, interpolation_noise)


def main():
    discriminator_in_channels = CHANNELS + CLASSES
    generator_in_channels = GENDIM + CLASSES
    discriminator = build_discriminator(discriminator_in_channels)
    generator = build_generator(generator_in_channels)
    cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator, gen_input_dim=GENDIM)
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    dataset = load_dataset()
    cond_gan.fit(dataset, epochs=2)
    cond_gan.generator.save("./logs/c_gan_gen.h5")

    num_interpolation = 9
    interpolation_noise = tf.random.normal(shape=(1, GENDIM))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, GENDIM))
    start_class = 1
    end_class = 5
    gen_image(generator, start_class, end_class, num_interpolation, interpolation_noise)


if __name__ == '__main__':
    main()
