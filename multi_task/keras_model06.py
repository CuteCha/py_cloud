import copy
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
    input00 = keras.Input(shape=(28, 28, 1), name='input00')
    layer01 = net01(input00)
    layer02 = net02(input00)

    model01 = keras.Model(input00, layer01)
    model02 = keras.Model(input00, layer02)

    multiply_layer = keras.layers.Multiply()([model01(input00), model02(input00)])
    fc = keras.layers.Dense(15, activation='softmax', name='fc1')(multiply_layer)

    model03 = keras.Model(input00, fc)

    # model03.summary()

    return model01, model02, model03


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
    model01, model02, model03 = get_model()

    opt01 = keras.optimizers.SGD(0.0001, momentum=0.01, nesterov=True)
    opt02 = keras.optimizers.SGD(0.0003, momentum=0.01, nesterov=True)

    ds_train, ds_valid = load_dataset()
    print(len(ds_train))
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


def train_model():
    @tf.function
    def train(model, loss_func, optimizer, train_loss, train_metric, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_func(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        train_metric.update_state(labels, predictions)

    return train


# @tf.function #optimizer.apply_gradients tf.function-decorated function tried to create variables on non-first call.
def train_step(model, loss_func, optimizer, train_loss, train_metric, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


def train02():
    model01, model02, model03 = get_model()

    opt01 = keras.optimizers.SGD(0.0001, momentum=0.01, nesterov=True)
    opt02 = keras.optimizers.SGD(0.0003, momentum=0.01, nesterov=True)

    loss_func = keras.losses.SparseCategoricalCrossentropy()

    train_loss = keras.metrics.Mean(name='train_loss')
    train_metric = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    ds_train, ds_valid = load_dataset()
    print(len(ds_train))
    for epoch in range(2):
        for k, (feature, label) in enumerate(ds_train):
            if k > 10: break
            print("train model01")
            model01.trainable = True
            model02.trainable = False
            train_step01 = train_model()
            train_step01(model03, loss_func, opt01, train_loss, train_metric, feature, label)
            print("epoch:{}/{}, loss:{}, acc:{}".format(k, epoch, train_loss.result(), train_metric.result()))

            print("train model02")
            model01.trainable = False
            model02.trainable = True
            train_step02 = train_model()
            train_step02(model03, loss_func, opt02, train_loss, train_metric, feature, label)
            print("epoch:{}/{}, loss:{}, acc:{}".format(k, epoch, train_loss.result(), train_metric.result()))


def main():
    train02()


if __name__ == '__main__':
    main()
