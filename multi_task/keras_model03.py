import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)

    return x_train, y_train, x_test, y_test


def input_fn(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) \
        .shuffle(buffer_size=1000).batch(32) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    return dataset


def load_dataset():
    x_train, y_train, x_test, y_test = load_data()
    max_words = x_train.max() + 1
    cat_num = y_train.max() + 1

    ds_train = input_fn(x_train, y_train)
    ds_test = input_fn(x_test, y_test)

    return ds_train, ds_test, max_words, cat_num


def get_model(max_words, cat_num):
    model = keras.Sequential([
        keras.layers.Embedding(max_words, 7, input_length=300),
        keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
        keras.layers.MaxPool1D(2),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        keras.layers.MaxPool1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(cat_num, activation="softmax")
    ])

    return model


def train01():
    ds_train, ds_test, max_words, cat_num = load_dataset()
    print("max_words:{}, cat_num:{}".format(max_words, cat_num))
    model = get_model(max_words, cat_num)
    # model.summary()
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)]
                  )
    history = model.fit(ds_train, validation_data=ds_test, epochs=10)
    print(history.history)


def train02():
    ds_train, ds_test, max_words, cat_num = load_dataset()
    print("max_words:{}, cat_num:{}".format(max_words, cat_num))
    model = get_model(max_words, cat_num)
    # model.summary()
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)]
                  )
    for epoch in tf.range(1, 11):
        model.reset_metrics()

        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr / 2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_test:
            test_result = model.test_on_batch(x, y, reset_metrics=False)

        print("epoch:{}, train:{}, test:{}".format(epoch,
                                                   dict(zip(model.metrics_names, train_result)),
                                                   dict(zip(model.metrics_names, test_result))))


def train03():
    ds_train, ds_test, max_words, cat_num = load_dataset()
    print("max_words:{}, cat_num:{}".format(max_words, cat_num))
    model = get_model(max_words, cat_num)
    # model.summary()
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)]
                  )

    optimizer = keras.optimizers.Nadam()
    loss_func = keras.losses.SparseCategoricalCrossentropy()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_metric = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = keras.metrics.Mean(name='test_loss')
    test_metric = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in tf.range(1, 11):
        for features, labels in ds_train:
            train_step(model, loss_func, optimizer, train_loss, train_metric, features, labels)

        for features, labels in ds_test:
            test_step(model, loss_func, test_loss, test_metric, features, labels)

        tf.print(tf.strings.format("Epoch={}, Train Loss:{}, Train Accuracy:{}, Test Loss:{}, Test Accuracy:{}",
                                   (epoch, train_loss.result(), train_metric.result(),
                                    test_loss.result(), test_metric.result())))

        train_loss.reset_states()
        test_loss.reset_states()
        train_metric.reset_states()
        test_metric.reset_states()


@tf.function
def train_step(model, loss_func, optimizer, train_loss, train_metric, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def test_step(model, loss_func, test_loss, test_metric, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    test_loss.update_state(batch_loss)
    test_metric.update_state(labels, predictions)


def main():
    train03()


if __name__ == '__main__':
    main()
