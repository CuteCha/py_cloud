import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def load_data():
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    return x_train, y_train, x_valid, y_valid


def input_fn(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32)
    return dataset


def load_dataset():
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    dataset_train = input_fn(x_train, y_train)
    dataset_valid = input_fn(x_valid, y_valid)

    return dataset_train, dataset_valid


def get_model01():
    model = CustomModel(name="model")

    return model


def get_model02():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model


def get_model03():
    inputs = keras.Input(shape=(28, 28))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='model')

    return model


def get_model04():
    inputs = keras.Input(shape=(28, 28))
    outputs = NNBlock()(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name='model')

    return model


class CustomModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(28, 28))
        self.hidden_layer = keras.layers.Dense(128, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x

    def get_config(self):
        config = super().get_config()
        return config


class NNBlock(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.layer01 = keras.layers.Flatten()
        self.layer02 = keras.layers.Dense(128, activation='relu')
        self.layer03 = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, *args, **kwargs):
        x = self.layer01(inputs)
        x = self.layer02(x)
        x = self.layer03(x)

        return x


def train01():
    x_train, y_train, x_valid, y_valid = load_data()
    model = get_model01()
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_history = model.fit(x=x_train, y=y_train, epochs=5)
    print(train_history.history)

    eval_history = model.evaluate(x_valid, y_valid)
    print(eval_history)

    y_pred = model.predict(x_valid[:2])
    y_index = np.argmax(y_pred, axis=1)
    print(y_index)


def train01():
    dataset_train, dataset_valid = load_dataset()
    model = get_model01()
    optimizer = keras.optimizers.Adam(0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss = 0.0
    for epoch in range(5):
        for features, labels in dataset_train:
            loss = train_step01(model, loss_fn, optimizer, features, labels)
        print("Finished epoch: {}, loss: {}".format(epoch, loss))


@tf.function
def train_step01(model, loss_fn, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
        # regularization_loss = tf.math.add_n(model.losses)
        # total_loss = pred_loss + regularization_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train02():
    dataset_train, dataset_valid = load_dataset()
    model = get_model01()
    model.losses
    optimizer = keras.optimizers.Adam(0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(5):
        loss = train_step02(model, loss_fn, optimizer, dataset_train)
        print("Finished epoch: {}, loss: {}".format(epoch, loss))


@tf.function
def train_step02(model, loss_fn, optimizer, dataset):
    res = 0.0
    for x, y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
            res = loss
            regularization_loss = tf.math.add_n(model.losses)
            # total_loss = loss + regularization_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return res


def main():
    train02()


if __name__ == '__main__':
    main()
