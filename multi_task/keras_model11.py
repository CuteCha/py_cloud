import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = keras.backend


class MyLayer(keras.layers.Layer):
    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.kernel = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super().build(input_shape)

    def call(self, x, *args, **kwargs):
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a - b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


def FM(feature_dim):
    inputs = keras.Input((feature_dim,))
    liner = keras.layers.Dense(units=1,
                               bias_regularizer=keras.regularizers.l2(0.01),
                               kernel_regularizer=keras.regularizers.l1(0.02),
                               )(inputs)
    cross = MyLayer(feature_dim)(inputs)
    add = keras.layers.Add()([liner, cross])
    predictions = keras.layers.Activation('sigmoid')(add)
    model = keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.BinaryCrossentropy()])
    return model


def main01():
    model = FM(30)
    model.summary()
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2,
        random_state=27, stratify=data.target)
    model.fit(x_train, y_train, epochs=3, batch_size=16, validation_data=(x_test, y_test))


def get_first_order(feat_index, args):
    embedding = tf.nn.embedding_lookup(args, feat_index)[:, :, -1]
    print('embedding', embedding)
    linear = tf.reduce_sum(embedding, axis=-1)
    sum_embedding = K.expand_dims(linear, axis=1)
    return sum_embedding


def get_second_order(feat_index, args):
    embedding = tf.nn.embedding_lookup(args, feat_index)[:, :, :args.shape[-1] - 1]
    # 加权求和
    sum_embedding = tf.reduce_sum(embedding, axis=1)
    sum_square = K.square(sum_embedding)
    # 先平方在求和
    squared = K.square(embedding)
    square_sum = tf.reduce_sum(squared, axis=1)
    # 二阶交叉项
    second_order = 0.5 * tf.subtract(sum_square, square_sum)
    return second_order


class FmLayer(keras.layers.Layer):

    def __init__(self, feature_num, output_dim, **kwargs):
        self.kernel = None
        self.feature_num = feature_num
        self.output_dim = output_dim

        super().__init__(**kwargs)

    # 定义模型初始化 根据特征数目
    def build(self, input_shape):
        # create a trainable weight variable for this layer
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.feature_num, self.output_dim + 1),
                                      initializer='glorot_normal',
                                      trainable=True)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # input 为多个样本的稀疏特征表示
        first_order = get_first_order(inputs, self.kernel)
        seconder_order = get_second_order(inputs, self.kernel)
        concat_order = tf.concat([first_order, seconder_order], axis=-1)
        return concat_order

    def compute_output_shape(self, input_shape):
        return input_shape(0), self.output_dim + 1


def load_data():
    import numpy as np
    # 原始特征输入
    num_samples = 60000
    cate_a = np.random.randint(0, 100, (num_samples, 1))
    cate_b = np.random.randint(100, 200, (num_samples, 1))
    cate_c = np.random.randint(200, 300, (num_samples, 1))
    cate_d = np.random.randint(300, 400, (num_samples, 1))

    features = np.concatenate([cate_a, cate_b, cate_c, cate_d], axis=-1).astype('int32')
    print("训练数据样例与Size:")
    print(features[0:5])
    print(features.shape)

    labels = np.random.randint(0, 2, size=num_samples)
    labels = np.asarray(labels)
    print("样本labels:")
    print(labels[0:10])

    return features, labels


def main02():
    feature_input = keras.layers.Input(shape=4, name='deep', dtype='int32')
    fm_layer = FmLayer(400, 8)(feature_input)
    pred = keras.layers.Dense(1, activation='sigmoid')(fm_layer)
    model = keras.Model(feature_input, pred)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics='accuracy')
    model.summary()

    features, labels = load_data()
    model.fit(features, labels, epochs=10, batch_size=128)


def debug():
    import numpy as np
    a = np.arange(1, 17).reshape(4, 4)
    au = np.triu(a)  # np.triu(a,1) 不包括对角元
    al = np.tril(a)  # np.tril(a,-1) 不包括对角元
    adv = np.diag(a)
    ad = np.diag(np.diag(a))

    b = tf.constant(a, dtype=tf.float32)
    bu = tf.linalg.band_part(b, 0, -1)  # tf.linalg.band_part(b, -1, 0)-tf.linalg.band_part(b, 0, 0) 不包括对角元


if __name__ == '__main__':
    main02()
