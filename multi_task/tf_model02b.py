# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.feature_column import *

tf.disable_v2_behavior()

train_file = "./data/iris/train.csv"
eval_file = "./data/iris/test.csv"
model_dir = "./logs/iris/checkpoints/"
server_model_dir = "./logs/iris/save_model/"

_FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
_SPECIES = ['Setosa', 'Versicolor', 'Virginica']
_HEADERS = _FEATURES + ['Species']

_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0]]


def build_model_columns():
    feature_columns = [numeric_column(key) for key in _FEATURES]

    return feature_columns


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


def eval_input_fn(features, labels, batch_size=1):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)

    return dataset


def model_fn(features, labels, mode, params):
    net = input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    pred_cls = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'cls_ids': pred_cls[:, tf.newaxis],
            'probs': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_cls,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def get_serving_input_receiver_fn():
    feature_columns = build_model_columns()
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)


def main():
    feature_columns = build_model_columns()

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
        log_step_count_steps=10
    ).replace(session_config=session_conf)

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [100, 25],
            'n_classes': 3,
        },
        config=run_config)

    model.train(
        input_fn=lambda: input_fn(train_file, 100, 64, True),
        hooks=None)
    print("train done")
    print("=" * 72)

    results = model.evaluate(
        input_fn=lambda: input_fn(eval_file, 1, 64, False))
    print(results)
    print("=" * 72)

    predict_feature = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predict_y = ['Setosa', 'Versicolor', 'Virginica']
    predictions = model.predict(
        input_fn=lambda: eval_input_fn(predict_feature, None)
    )
    for pred_dict, y in zip(predictions, predict_y):
        cls_id = pred_dict['cls_ids'][0]
        prob = pred_dict['probs'][cls_id]
        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(_SPECIES[cls_id], 100 * prob, y))
    print("evaluate done")
    print("=" * 72)
    serving_input_receiver_fn = get_serving_input_receiver_fn()
    model.export_saved_model(server_model_dir, serving_input_receiver_fn)
    print("save model done")


if __name__ == '__main__':
    main()
