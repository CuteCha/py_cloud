import time
import tensorflow.keras as keras
import tensorflow as tf


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return tf.expand_dims(image, -1), label


def load_data():
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.mnist.load_data()

    return x_train, y_train, x_valid, y_valid


def input_fn(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) \
        .map(scale).shuffle(buffer_size=1024).batch(32) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    return dataset


def load_dataset():
    x_train, y_train, x_valid, y_valid = load_data()

    ds_train = input_fn(x_train, y_train)
    ds_test = input_fn(x_valid, y_valid)

    return ds_train, ds_test


def get_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.02),
                            input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def net01(x):
    x = keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(10, activation="softmax")(x)

    return x


def net02(x):
    x = keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02))(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(10, activation="softmax")(x)

    return x


def train01():
    input00 = keras.Input(shape=(28, 28, 1), name='input00')
    layer01 = net01(input00)
    layer02 = net02(input00)

    model01 = keras.Model(input00, layer01)
    model02 = keras.Model(input00, layer02)

    multiply_layer = keras.layers.Multiply()([model01(input00), model02(input00)])
    fc = keras.layers.Dense(15, activation='softmax', name='fc1')(multiply_layer)

    model03 = keras.Model(input00, fc)

    # model03.summary()

    opt01 = keras.optimizers.SGD(0.0001, momentum=0.01, nesterov=True)
    opt02 = keras.optimizers.SGD(0.0003, momentum=0.01, nesterov=True)

    ds_train, ds_valid = load_dataset()
    for epoch in range(2):
        for k, (x, y) in enumerate(ds_train):
            if k > 10: break
            print("train model01")
            model01.trainable = True
            model02.trainable = False
            model03.compile(
                optimizer=opt01,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)])

            train_result01 = model03.train_on_batch(x, y)
            print("epoch:{}/{}, train:{}".format(k, epoch, dict(zip(model03.metrics_names, train_result01))))

            print("train model02")
            model01.trainable = False
            model02.trainable = True
            model03.compile(
                optimizer=opt02,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)])

            train_result02 = model03.train_on_batch(x, y)
            print("epoch:{}/{}, train:{}".format(k, epoch, dict(zip(model03.metrics_names, train_result02))))


def train02():
    ds_train, ds_valid = load_dataset()
    model = get_model()
    # model.summary()
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(5)]
                  )

    ts = time.time()
    for epoch in tf.range(1, 6):
        model.reset_metrics()

        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr / 2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y, reset_metrics=False)

        print("epoch:{}, train:{}, valid:{}".format(epoch,
                                                    dict(zip(model.metrics_names, train_result)),
                                                    dict(zip(model.metrics_names, valid_result))))

    print("cost: {}ms".format(time.time() - ts))


def train03():
    ds_train, ds_valid = load_dataset()
    model = get_model()

    optimizer = keras.optimizers.Nadam()
    loss_func = keras.losses.SparseCategoricalCrossentropy()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_metric = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = keras.metrics.Mean(name='valid_loss')
    valid_metric = keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    ts = time.time()
    for epoch in tf.range(5):
        for features, labels in ds_train:
            train_step(model, loss_func, optimizer, train_loss, train_metric, features, labels)

        for features, labels in ds_valid:
            valid_step(model, loss_func, valid_loss, valid_metric, features, labels)

        tf.print(tf.strings.format("epoch:{}, train loss:{}, train accuracy:{}, valid loss:{}, valid accuracy:{}",
                                   (epoch + 1, train_loss.result(), train_metric.result(),
                                    valid_loss.result(), valid_metric.result())))

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

    print("cost: {}ms".format(time.time() - ts))


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
def valid_step(model, loss_func, valid_loss, valid_metric, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def main():
    train01()


if __name__ == '__main__':
    main()
