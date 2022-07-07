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
    x_train, y_train, x_valid, y_valid = load_data()

    dataset_train = input_fn(x_train, y_train)
    dataset_valid = input_fn(x_valid, y_valid)

    return dataset_train, dataset_valid


def main():
    train02()


if __name__ == '__main__':
    main()
