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


def get_serving_input_receiver_fn(wide_columns, deep_columns):
    feature_spec = tf.feature_column.make_parse_example_spec(wide_columns + deep_columns)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)


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


def debug():
    a = numeric_column('a')
    b = numeric_column('b')
    a_buckets = bucketized_column(a, boundaries=[10, 15, 20, 25, 30])

    data = {'a': [15, 9, 17, 19, 21, 18, 25, 30],
            'b': [5.0, 6.4, 10.5, 13.6, 15.7, 19.9, 20.3, 0.0]}

    print("=" * 72)
    inputs = input_layer(data, [a_buckets, b])
    a_inputs = input_layer(data, [a])
    y = tf.layers.dense(inputs, 3)
    print(inputs)
    print(a_inputs)
    print(y)
    print("=" * 72)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(inputs))
        print(sess.run(a_inputs))
        print(sess.run(y))


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


def model_fn(features, labels, mode, params):
    wide_columns = params["wide_columns"]
    deep_columns = params["deep_columns"]

    wide_inputs = input_layer(features=features, feature_columns=wide_columns)
    deep_inputs = input_layer(features=features, feature_columns=deep_columns)
    nn_outputs = tf.layers.dense(deep_inputs, 128, activation=tf.nn.relu)
    nn_outputs = tf.layers.dense(nn_outputs, 64, activation=tf.nn.relu)

    last_inputs = tf.concat([wide_inputs, nn_outputs], axis=1)
    logits = tf.layers.dense(last_inputs, params['n_classes'], activation=None, name="ctr")
    pred_cls = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'cls_ids': pred_cls[:, tf.newaxis],
                'probs': tf.nn.softmax(logits),
                'logits': logits,
            })

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'acc': tf.metrics.accuracy(labels=labels, predictions=pred_cls),
                "auc": tf.metrics.auc(labels=labels, predictions=tf.sigmoid(logits[:, 1]))
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)


def main():
    wide_columns, deep_columns = build_model_columns()

    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)),
        log_step_count_steps=100, save_summary_steps=500)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={
            "wide_columns": wide_columns,
            "deep_columns": deep_columns,
            'n_classes': 2
        },
        config=config)

    model.train(
        input_fn=lambda: input_fn(train_file, 1, 64, True))
    print("train success")

    results = model.evaluate(
        input_fn=lambda: input_fn(eval_file, 1, 64, False))
    print(results)
    print("evaluate success")

    results = model.predict(
        input_fn=lambda: input_fn(eval_file, 1, 64, False))
    print(results)
    print("predict success")

    serving_input_receiver_fn = get_serving_input_receiver_fn(wide_columns, deep_columns)
    model.export_saved_model(server_model_dir, serving_input_receiver_fn)
    print("save model success")


if __name__ == '__main__':
    main()
