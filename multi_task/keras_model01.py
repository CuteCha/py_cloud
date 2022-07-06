from abc import ABC
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def model01():
    model = keras.Sequential()
    # 指定input维度，模型持续创建
    model.add(keras.layers.Dense(64, activation="relu", input_shape=(16, 0)))
    # 等价方法
    # model.add(keras.layers.Dense(64, activation="relu", input_dim=16))
    # model.add(keras.layers.Dense(64, activation="relu", batch_input_shape=(None, 16)))

    model.add(keras.layers.Dense(64, activation="gelu"))
    model.add(keras.layers.Dense(10))

    # 模型已经初始化了，可以查看weights的信息
    print(model.weights)

    return model


def model02():
    model = keras.Sequential()
    # 未指定input维度，模型延迟创建
    model.add(keras.layers.Dense(64, activation="relu"))

    model.add(keras.layers.Dense(64, activation="gelu"))
    model.add(keras.layers.Dense(10))

    # 模型未初始化了，不能查看weights的信息；模型在训练方法或者评估方法调用是才能创建

    return model


def model01b():
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(16, 0)),
        keras.layers.Dense(64, activation="gelu"),
        keras.layers.Dense(10)
    ])

    print(model.weights)

    return model


def model03():
    inputs = keras.Input(shape=(16,))
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="gelu")(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model")
    model.summary()

    return model


def model04():
    categorical_input = keras.Input(shape=(16,))
    numeric_input = keras.Input(shape=(32,))

    categorical_features = keras.layers.Embedding(
        input_dim=100,
        output_dim=64,
        input_length=16
    )(categorical_input)
    categorical_features = keras.layers.Reshape([16 * 64])(categorical_features)

    numeric_features = keras.layers.Dense(64, activation='relu')(numeric_input)

    x = keras.layers.Concatenate(axis=-1)([categorical_features, numeric_features])
    x = keras.layers.Dense(128, activation='relu')(x)

    binary_pred = keras.layers.Dense(1, activation='sigmoid', name="ctr")(x)
    categorical_pred = keras.layers.Dense(3, activation='softmax', name="level")(x)

    model = keras.Model(
        inputs=[categorical_input, numeric_input],
        outputs=[binary_pred, categorical_pred]
    )
    model.summary()

    return model


def model05():
    categorical_input_one = keras.Input(shape=(16,))
    categorical_input_two = keras.Input(shape=(24,))

    shared_embedding = keras.layers.Embedding(100, 64)

    categorical_features_one = shared_embedding(categorical_input_one)
    categorical_features_two = shared_embedding(categorical_input_two)

    categorical_features_one = keras.layers.Reshape([16 * 64])(categorical_features_one)
    categorical_features_two = keras.layers.Reshape([24 * 64])(categorical_features_two)

    x = keras.layers.Concatenate(axis=-1)([categorical_features_one, categorical_features_two])
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(
        inputs=[categorical_input_one, categorical_input_two],
        outputs=outputs
    )
    model.summary()

    return model


class CustomDense(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model06():
    inputs = keras.Input((4,))
    layer = CustomDense(10)
    outputs = layer(inputs)

    model = keras.Model(inputs, outputs)
    model.summary()

    # layer recreate
    config = layer.get_config()
    new_layer = CustomDense.from_config(config)
    new_outputs = new_layer(inputs)
    print(new_layer.weights)
    print(new_layer.non_trainable_weights)
    print(new_layer.trainable_weights)

    # model recreate
    config = model.get_config()
    new_model = keras.Model.from_config(
        config,
        custom_objects={'CustomDense': CustomDense},
    )
    new_model.summary()


class MLP(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense_1 = keras.layers.Dense(64, activation='relu')
        self.dense_2 = keras.layers.Dense(64, activation='relu')
        self.dense_3 = keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x


def model07():
    inputs = keras.Input((16,))
    mlp = MLP()

    y = mlp(inputs)
    print('weights:', len(mlp.weights))
    print('trainable weights:', len(mlp.trainable_weights))


class CustomModel(keras.Model, ABC):
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(28, 28))
        self.hidden_layer = keras.layers.Dense(128, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x


def model08():
    """
    # 查看模型结构
    model = CustomModel(name="model")
    model.build(input_shape=(None, 28, 28))
    print(model.weights)
    print("*" * 72)
    model.summary()
    """

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    model = CustomModel(name="model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)
    model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test[:1])
    print(y_pred)
    print(np.argmax(y_pred, axis=1))


class MMoE(keras.Model):
    def __init__(self, units, num_experts, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass


def model09():
    pass


class Encoder(keras.layers.Layer):
    def __init__(self, l2_rate=1e-3):
        super().__init__()
        self.l2_rate = l2_rate
        self.dense1 = None
        self.dense2 = None
        self.dense3 = None

    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.dense2 = keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.dense3 = keras.layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, l2_rate=1e-3):
        super().__init__()
        self.l2_rate = l2_rate
        self.dense1 = None
        self.dense2 = None
        self.dense3 = None

    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(
            units=64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.dense2 = keras.layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )
        self.dense3 = keras.layers.Dense(
            units=16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_rate),
        )

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class AutoEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def get_config(self):
        pass


def model10():
    model = AutoEncoder()
    model.build((None, 16))
    model.summary()
    print(model.layers)
    print(model.weights)


if __name__ == '__main__':
    model10()
