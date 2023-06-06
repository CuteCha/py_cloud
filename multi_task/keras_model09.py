# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow._api.v2.feature_column import *


def bebug01():
    age_bucket = bucketized_column(
        numeric_column(key='age', shape=(1,),
                       default_value=0,
                       dtype=tf.dtypes.float32),
        boundaries=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    features = dict()
    features['age'] = [[23.0], [35.0], [68.0]]
    feature_layer = keras.layers.DenseFeatures(age_bucket)
    inputs = feature_layer(features)
    print(inputs)


def debug02():
    a = numeric_column('a')
    b = numeric_column('b')
    a_buckets = bucketized_column(a, boundaries=[10, 15, 20, 25, 30])
    feature_layer = keras.layers.DenseFeatures([a_buckets, b])

    data = {'a': [15, 9, 17, 19, 21, 18, 25, 30],
            'b': [5.0, 6.4, 10.5, 13.6, 15.7, 19.9, 20.3, 0.0]}
    inputs = feature_layer(data)
    print(inputs)


def debug03():
    sex = categorical_column_with_vocabulary_list('sex', ['male', 'female'])
    sex_ind = indicator_column(sex)
    sex_emb = embedding_column(sex, dimension=8)

    z_buckets = bucketized_column(numeric_column('z'), boundaries=[10, 15, 20, 25, 30])

    uid = categorical_column_with_hash_bucket('uid', hash_bucket_size=10)
    uid_emb = embedding_column(uid, dimension=8)

    feature_columns = [z_buckets, sex_ind, sex_emb, uid_emb]
    # z_buckets.name = "a"  # can't set attribute

    # sorted(feature_columns, key=lambda x: x.name) 输入在内部被排序了
    feature_layer = keras.layers.DenseFeatures(feature_columns)

    data = {'sex': ["male", "male", "female", "male", "female", "female", "female", "male"],
            'z': [15, 9, 17, 19, 21, 18, 25, 30],
            'uid': ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]}
    inputs = feature_layer(data)
    print(inputs)

    # 验证输出顺序
    print("=" * 72)
    ori_name = [c.name for c in feature_columns]
    out_order = sorted(ori_name)
    print(ori_name)
    print(out_order)


def debug04():
    color = categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    color_ind = indicator_column(color)
    color_emb = embedding_column(color, dimension=8)
    data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}
    feature_layer = keras.layers.DenseFeatures([color_ind, color_emb])
    inputs = feature_layer(data)
    print(inputs)


def debug04b():
    good_sets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    fea1 = categorical_column_with_vocabulary_list('fea1', good_sets, dtype=tf.string)

    emb_feat1 = embedding_column(fea1, 3, combiner='mean')
    data = {'fea1': [['a', 'b', 'c'], ['a', 'e', 'f']]}
    feature_layer = keras.layers.DenseFeatures([emb_feat1])
    inputs = feature_layer(data)
    print(inputs)


def debug05():
    f_input = keras.Input(shape=(100,), dtype='int32', name='input')
    x = keras.layers.Embedding(
        output_dim=512, input_dim=10000, input_length=100)(f_input)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[f_input], outputs=[output])
    dot_img_file = './logs/model.png'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    model.summary()


def debug06():
    movie_id = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=10, default_value=0)
    movie_id_emb = tf.feature_column.embedding_column(movie_id, 3)
    movie_id_ind = tf.feature_column.indicator_column(movie_id)

    data = {'movieId': [1, 7, 9]}
    feature_layer = keras.layers.DenseFeatures([movie_id_emb, movie_id_ind])
    features = feature_layer(data)
    print(features)


if __name__ == '__main__':
    debug06()
