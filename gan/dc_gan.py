import tensorflow as tf
import tensorflow.keras as keras

from matplotlib import pyplot as plt

LATENT_DIM = 100
WEIGHT_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
CHANNELS = 1
LR = 0.0002
EPOCHS = 1


def load_data():
    (train_images, train_labels), (_, _) = keras.datasets.fashion_mnist.load_data()
    print("train_images.shape: {}, train_labels.shape: {}".format(train_images.shape, train_labels.shape))
    print(type(train_images))

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return train_images, train_labels


def build_generator():
    model = keras.Sequential(name="generator")

    model.add(keras.layers.Dense(7 * 7 * 256, input_dim=LATENT_DIM))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(keras.layers.BatchNormalization())
    model.add((keras.layers.ReLU()))

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(CHANNELS, (5, 5), padding="same", activation="tanh"))

    return model


def build_discriminator(width, height, depth, alpha=0.2):
    model = keras.Sequential(name="discriminator")
    input_shape = (height, width, depth)

    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=alpha))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=alpha))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    return model


class DCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_images, training=True)
            d_loss_real = self.loss_fn(tf.ones((batch_size, 1)), pred_real)

            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images, training=True)
            d_loss_fake = self.loss_fn(tf.zeros((batch_size, 1)), pred_fake)

            d_loss = (d_loss_real + d_loss_fake) / 2
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=100):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim

        self.seed = tf.random.normal([16, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.seed)
        generated_images = (generated_images * 127.5) + 127.5
        generated_images.numpy()

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i + 1)
            img = keras.utils.array_to_img(generated_images[i])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.savefig('./logs/fig/epoch_{:03d}.png'.format(epoch))
        plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save('./logs/dc_gan_gen.h5')


def main():
    discriminator = build_discriminator(28, 28, 1)
    generator = build_generator()
    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

    dcgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=LR, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=LR, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    train_images, train_labels = load_data()
    dcgan.fit(train_images, epochs=EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])


if __name__ == '__main__':
    main()
