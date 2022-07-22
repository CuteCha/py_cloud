# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow._api.v2.feature_column import *

_HEADERS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket']

_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
             [0], [0], [0], [''], ['']]

train_file = "./data/adult/train.csv"
valid_file = "./data/adult/valid.csv"
model_dir = "./logs/adult/checkpoints/"
server_model_dir = "./logs/adult/save_model/"
model_h5 = "./logs/adult/h5/model.h5"
model_pb = "./logs/adult/pb/"


def build_model_columns():
    # 连续型特征
    age = numeric_column('age')
    education_num = numeric_column('education_num')
    capital_gain = numeric_column('capital_gain')
    capital_loss = numeric_column('capital_loss')
    hours_per_week = numeric_column('hours_per_week')

    # 离散型特征
    education = categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100)

    age_buckets = bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # 用于宽度模型与深度模型
    base_columns = [
        indicator_column(education), indicator_column(marital_status), indicator_column(relationship),
        indicator_column(workclass), indicator_column(occupation), age_buckets]

    # 构建交叉特征
    crossed_columns = [
        indicator_column(crossed_column(['education', 'occupation'], hash_bucket_size=100)),
        indicator_column(crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=100)),
        indicator_column(crossed_column([age_buckets, "gender"], hash_bucket_size=100)),
        indicator_column(crossed_column(["native_country", "gender"], hash_bucket_size=100))
    ]

    # 宽度模型特征列
    wide_columns = base_columns + crossed_columns
    # 深度模型特征列
    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        # embedding特征列
        embedding_column(occupation, dimension=8),
        embedding_column(workclass, dimension=8)
    ]

    return wide_columns, deep_columns


def build_feature_columns():
    # age = numeric_column("age")
    # age_buckets = bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    # thal = categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
    # thal_one_hot = indicator_column(thal)
    # thal_embedding = embedding_column(thal, dimension=8)
    # thal_hashed = categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
    # crossed_feature = crossed_column([age_buckets, thal], hash_bucket_size=1000)

    feature_columns = []

    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(numeric_column(header))

    # bucketized cols
    age = numeric_column("age")
    age_buckets = bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    thal = categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = indicator_column(thal)
    feature_columns.append(thal_one_hot)

    occupation = categorical_column_with_hash_bucket('occupation', hash_bucket_size=100)
    occupation_one_hot = indicator_column(occupation)
    feature_columns.append(occupation_one_hot)

    # embedding cols
    thal_embedding = embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    # crossed cols
    crossed_feature = crossed_column([age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    feature_layer = keras.layers.DenseFeatures(feature_columns)
    inputs = keras.layers.Input(tensor=feature_layer, name='features')
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)


def input_fn(data_file, num_epochs, batch_size, shuffle):
    tf.print(f'Parsing {data_file}')

    # 解析csv文件中一行数据
    def parse_csv(value):
        columns = tf.io.decode_csv(value, record_defaults=_DEFAULTS, field_delim=",")
        features = dict(zip(_HEADERS, columns))
        print(features)
        labels = features.pop('income_bracket')
        classes = tf.cast(tf.equal(labels, '>50K'), tf.int32)  # binary classification
        return features, classes

    # 构建训练数据集
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def main01():
    wide_columns, deep_columns = build_model_columns()
    feature_columns = wide_columns + deep_columns

    model = keras.Sequential([
        keras.layers.DenseFeatures(feature_columns),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.SparseCategoricalCrossentropy()
                  )

    ds_train = input_fn(train_file, 1, 64, True)
    ds_valid = input_fn(valid_file, 1, 3, False)
    history = model.fit(ds_train, validation_data=ds_valid, epochs=10)
    print(history.history)


def main02():
    wide_columns, deep_columns = build_model_columns()
    feature_columns = wide_columns + deep_columns

    model = keras.Sequential([
        keras.layers.DenseFeatures(feature_columns),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', name="pred"),
    ])

    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.BinaryCrossentropy(name="loss"),
                  # loss=keras.losses.binary_crossentropy,  # loss='binary_crossentropy',
                  metrics=[
                      keras.metrics.BinaryAccuracy(name="acc"),
                      # keras.metrics.Accuracy().binary_accuracy,  # 'accuracy'
                      # keras.metrics.Precision(name="precision"),
                      # keras.metrics.Recall(name="recall"),
                      # keras.metrics.AUC(curve='PR', num_thresholds=512, name='pr'),
                      keras.metrics.AUC(num_thresholds=512, name='auc')]
                  )

    ds_train = input_fn(train_file, 10, 64, True)
    ds_valid = input_fn(valid_file, 10, 64, False)
    history = model.fit(ds_train, validation_data=ds_valid, epochs=1)
    print(history.history)

    model.save(model_h5)
    model.save(model_pb, save_format="tf")


def build_model_inputs():
    # 1.连续型特征
    num_headers = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    # 1.1 连续特征直接取值
    num_columns = []
    buc_columns = []
    num_inputs = {}
    for header in num_headers:
        num_inputs[header] = keras.Input(shape=(1,), name=header)
        num_columns.append(numeric_column(header))

    # 1.2 连续特征分桶
    age = numeric_column('age')
    age_buckets = bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    buc_columns.append(age_buckets)

    # 2.离散型特征
    education = categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=100)

    # 2.1 one-hot
    cate_one_hot_columns = [indicator_column(education), indicator_column(marital_status),
                            indicator_column(relationship),
                            indicator_column(workclass), indicator_column(occupation)]
    # 2.2 embedding
    cate_emb_columns = [embedding_column(occupation, dimension=8), embedding_column(workclass, dimension=8)]
    cate_inputs = {"education": keras.Input(shape=(1,), name='education', dtype=tf.string),
                   "marital_status": keras.Input(shape=(1,), name='marital_status', dtype=tf.string),
                   "relationship": keras.Input(shape=(1,), name='relationship', dtype=tf.string),
                   "workclass": keras.Input(shape=(1,), name='workclass', dtype=tf.string),
                   "occupation": keras.Input(shape=(1,), name='occupation', dtype=tf.string),
                   "gender": keras.Input(shape=(1,), name='gender', dtype=tf.string),
                   "native_country": keras.Input(shape=(1,), name='native_country', dtype=tf.string),
                   }

    # 构建交叉特征
    crossed_columns = [
        indicator_column(crossed_column(['education', 'occupation'], hash_bucket_size=100)),
        indicator_column(crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=100)),
        indicator_column(crossed_column([age_buckets, "gender"], hash_bucket_size=100)),
        indicator_column(crossed_column(["native_country", "gender"], hash_bucket_size=100))
    ]

    one_hot_columns = cate_one_hot_columns + buc_columns + crossed_columns
    dense_columns = num_columns + cate_emb_columns
    feature_columns = one_hot_columns + dense_columns
    inputs = dict(num_inputs, **cate_inputs)

    return feature_columns, inputs


def main():
    feature_columns, inputs = build_model_inputs()
    print(f'inputs:\n{inputs}\n')
    feature_layer = keras.layers.DenseFeatures(feature_columns)
    input_tensor = feature_layer(inputs)

    dense_tensor = keras.layers.Dense(100, activation='relu')(input_tensor)
    dense_tensor = keras.layers.Dense(100, activation='relu')(dense_tensor)
    output_tensor = keras.layers.Dense(1, activation='sigmoid', name="pred")(dense_tensor)

    model = keras.Model(inputs=[v for v in inputs.values()], outputs=[output_tensor])
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.BinaryCrossentropy(name="loss"),
                  metrics=[
                      keras.metrics.BinaryAccuracy(name="acc"),
                      keras.metrics.AUC(num_thresholds=512, name='auc')]
                  )

    model.summary()

    ds_train = input_fn(train_file, 10, 64, True)
    ds_valid = input_fn(valid_file, 10, 64, False)
    history = model.fit(ds_train, validation_data=ds_valid, epochs=1)
    print(history.history)

    # model.save(model_h5)
    # model.save(model_pb, save_format="tf")


if __name__ == '__main__':
    main()
