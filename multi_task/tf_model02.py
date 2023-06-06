# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.feature_column import *

tf.disable_v2_behavior()

_FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
_SPECIES = ['Setosa', 'Versicolor', 'Virginica']
_HEADERS = _FEATURES + ['Species']

_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0]]


def build_model_columns():
    columns = [numeric_column(key) for key in _FEATURES]

    return columns


def input_fn(data_file, num_epochs, batch_size, shuffle):
    # 解析csv文件中一行数据
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.io.decode_csv(value, record_defaults=_DEFAULTS)
        features = dict(zip(_HEADERS, columns))
        labels = features.pop('Species')
        classes = tf.cast(labels, tf.int32)  # binary classification
        return features, classes

    # 构建训练数据集
    dataset = tf.data.TextLineDataset(data_file).skip(1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def get_serving_input_receiver_fn():
    columns = build_model_columns()
    feature_spec = tf.feature_column.make_parse_example_spec(columns)

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)


def app(_):
    columns = build_model_columns()
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

    model_dir = "./logs/iris/checkpoints/"

    model = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=columns,
        hidden_units=hidden_units,
        n_classes=3,
        config=run_config)

    train_file = "./data/iris/train.csv"
    eval_file = "./data/iris/test.csv"
    model.train(
        input_fn=lambda: input_fn(train_file, 1, 64, True),
        hooks=None)

    results = model.evaluate(
        input_fn=lambda: input_fn(eval_file, 1, 64, False))
    print(results)

    server_model_dir = "./logs/iris/save_model/"
    serving_input_receiver_fn = get_serving_input_receiver_fn()
    model.export_saved_model(server_model_dir, serving_input_receiver_fn)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=app)


if __name__ == '__main__':
    main()
