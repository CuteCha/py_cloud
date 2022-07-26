# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.feature_column import *

tf.disable_v2_behavior()

_HEADERS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket']

_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
             [0], [0], [0], [''], ['']]

train_file = "./data/adult/adult.data"
eval_file = "./data/adult/adult.test"
model_dir = "./logs/adult/checkpoints/"
server_model_dir = "./logs/adult/save_model/"


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
        education, marital_status, relationship, workclass, occupation,
        age_buckets]
    # 构建交叉特征
    crossed_columns = [
        crossed_column(['education', 'occupation'], hash_bucket_size=100),
        crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=100),
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
        # onehot 特征列
        indicator_column(workclass),
        indicator_column(education),
        indicator_column(marital_status),
        indicator_column(relationship),
        # embedding特征列
        embedding_column(occupation, dimension=8),
        embedding_column(workclass, dimension=8)
    ]

    return wide_columns, deep_columns


def input_fn(data_file, num_epochs, batch_size, shuffle):
    # 解析csv文件中一行数据
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.io.decode_csv(value, record_defaults=_DEFAULTS)
        features = dict(zip(_HEADERS, columns))
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


def get_serving_input_receiver_fn(wide_columns, deep_columns):
    feature_spec = tf.feature_column.make_parse_example_spec(wide_columns + deep_columns)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)


def main(_):
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 25]

    gpu_options = tf.GPUOptions(allow_growth=True)  # allow_growth=False
    session_conf = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True,
        gpu_options=gpu_options)  # log_device_placement=False
    run_config = tf.estimator.RunConfig(
        tf_random_seed=123,
        save_summary_steps=500,
        save_checkpoints_steps=10000,
        keep_checkpoint_max=1,
        log_step_count_steps=100
    ).replace(session_config=session_conf)

    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)

    model.train(
        input_fn=lambda: input_fn(train_file, 1, 64, True),
        hooks=None)

    results = model.evaluate(
        input_fn=lambda: input_fn(eval_file, 1, 64, False))
    print(results)

    serving_input_receiver_fn = get_serving_input_receiver_fn(wide_columns, deep_columns)
    model.export_savedmodel(server_model_dir, serving_input_receiver_fn)


def main02(_):
    tf.estimator.EstimatorSpec()
    tf.estimator.Head()
    tf.estimator.MultiHead()
    tf.estimator.RegressionHead()
    tf.estimator.MultiClassHead()
    tf.estimator.Head.create_estimator_spec()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
