# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def main01():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    # ds_numpy = tfds.as_numpy(ds_train)
    pred = model(ds_train).numpy()
    np.savetxt("./logs/num_emb.txt", pred)
    np.savetxt("./logs/num_val.txt", ds_train)
    print("done")


def main02():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # pred = probability_model(x_train).numpy()
    pred = model(x_train).numpy()
    np.savetxt("./logs/num_emb.txt", pred)
    np.savetxt("./logs/num_val.txt", y_train)
    print("done")


def show_result():
    emb = np.loadtxt("./logs/num_emb.txt")
    val = np.loadtxt("./logs/num_val.txt")
    pca = PCA(n_components=2)
    r_emb = pca.fit_transform(emb[:10000])
    # color = [plt.cm.Set3(i) for i in range(10)]
    color = ["red", "green", "blue", "yellow", "magenta", "cyan", "black", "gold", "purple", "silver"]

    res = dict()
    for i in range(len(r_emb)):
        k = int(val[i])
        res.setdefault(k, {"x": list(), "y": list()})
        res[k]["x"].append(r_emb[i][0])
        res[k]["y"].append(r_emb[i][1])

    fig, ax = plt.subplots()
    # plt.figure(figsize=(12, 12))

    for k in range(10):
        # plt.scatter(res[k]["x"], res[k]["y"], c=color[k], marker="x")
        ax.scatter(res[k]["x"], res[k]["y"], c=color[k])

    # for k in range(10):
    #     for i in range(len(res[k]["x"])):
    #         ax.annotate(i, (res[k]["x"][i], res[k]["y"][i]))

    plt.show()


if __name__ == '__main__':
    # main02()
    show_result()
