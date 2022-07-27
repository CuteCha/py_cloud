# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow._api.v2.feature_column import *

_HEADERS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket']

_DEFAULTS = [
    [0.0], [''], [0.0], [''], [0.0], [''], [''], [''], [''], [''],
    [0.0], [0.0], [0.0], [''], ['']]

train_file = "./data/adult/train.csv"
valid_file = "./data/adult/valid.csv"
model_dir = "./logs/adult/checkpoints/"
server_model_dir = "./logs/adult/save_model/"
model_h5 = "./logs/adult/h5/model.h5"
model_pb = "./logs/adult/pb/"


class MMoE:
    @staticmethod
    def expert(feature_inputs, name=0):
        x = keras.layers.Dense(128, activation=keras.activations.relu, name=f"e{name}_h1")(feature_inputs)
        x = keras.layers.Dense(126, activation=keras.activations.relu, name=f"e{name}_h2")(x)
        x = keras.layers.Dense(128, activation=keras.activations.relu, name=f"e{name}_h3")(x)

        return tf.expand_dims(x, axis=1)

    @staticmethod
    def task(feature_inputs, expert_outs, expert_nums, name="cls"):
        gate = keras.layers.Dense(expert_nums, activation=keras.activations.softmax, name=f"t{name}_g")(feature_inputs)
        gate = tf.expand_dims(gate, -1)
        weighted_expert_output = expert_outs * gate
        weighted_expert_output = tf.reduce_sum(weighted_expert_output, 1)

        last_inputs = weighted_expert_output
        last_inputs = keras.layers.BatchNormalization()(last_inputs)
        out_act = None
        # if name == "cls":
        #     out_act = keras.activations.sigmoid
        logit = keras.layers.Dense(1, activation=out_act, name=name)(last_inputs)

        return logit

    @staticmethod
    def build(expert_nums, task_names, feature_columns, feature_inputs):
        feature_layer = keras.layers.DenseFeatures(feature_columns)
        input_tensor = feature_layer(feature_inputs)

        expert_out_list = [MMoE.expert(input_tensor, i) for i in range(expert_nums)]
        expert_outs = keras.layers.concatenate(expert_out_list, axis=1)

        logit_list = [MMoE.task(input_tensor, expert_outs, expert_nums, i) for i in task_names]

        model = keras.Model(inputs=[v for v in feature_inputs.values()], outputs=logit_list, name="MMoE")
        return model


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


def input_fn01(data_file, num_epochs, batch_size, shuffle):
    tf.print(f'Parsing {data_file}')

    # 解析csv文件中一行数据
    def parse_csv(value):
        columns = tf.io.decode_csv(value, record_defaults=_DEFAULTS, field_delim=",")
        features = dict(zip(_HEADERS, columns))
        print(features)
        cls_label = features.pop('income_bracket')
        reg_label = features.pop('fnlwgt')
        classes = tf.cast(tf.equal(cls_label, '>50K'), tf.int32)  # binary classification
        return features, classes, reg_label

    # 构建训练数据集
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def main01():
    feature_columns, feature_inputs = build_model_inputs()
    model = MMoE.build(3, ["cls", "mse"], feature_columns, feature_inputs)
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss={
                      # "cls": tf.nn.sigmoid_cross_entropy_with_logits,
                      "cls": keras.losses.BinaryCrossentropy(),
                      "mse": keras.losses.MSE
                  },
                  # metrics=[
                  #     keras.metrics.BinaryAccuracy(name="acc"),
                  #     keras.metrics.AUC(num_thresholds=512, name='auc')]
                  )

    model.summary()

    ds_train = input_fn01(train_file, 10, 64, True)
    ds_valid = input_fn01(valid_file, 1, 64, False)
    history = model.fit(ds_train, validation_data=ds_valid, epochs=1)
    print(history.history)


def input_fn02(data_file, num_epochs, batch_size, shuffle):
    tf.print(f'Parsing {data_file}')

    # 解析csv文件中一行数据
    def parse_csv(value):
        columns = tf.io.decode_csv(value, record_defaults=_DEFAULTS, field_delim=",")
        features = dict(zip(_HEADERS, columns))
        print(features)
        cls_label = features.pop('income_bracket')
        reg_label = features.pop('fnlwgt')
        classes = tf.cast(tf.equal(cls_label, '>50K'), tf.int32)  # binary classification
        return features, classes, reg_label

    # 构建训练数据集
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def main02():
    feature_columns, feature_inputs = build_model_inputs()
    model = MMoE.build(3, ["cls", "mse"], feature_columns, feature_inputs)
    optimizer = keras.optimizers.Nadam()

    ds_train = input_fn02(train_file, 2, 1024, True)
    for step, (feature, cls, reg) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            pred_cls, pred_reg = model(feature, training=True)
            loss_cls = keras.losses.BinaryCrossentropy()(cls, tf.nn.sigmoid(pred_cls))
            # print(cls)  # (64, )
            # print(pred_cls)  # (64, 1)
            # print(reg)  # (64, )
            # print(pred_reg)  # (64, 1)
            # print(tf.reshape(pred_reg, [-1]))
            loss_reg = keras.losses.MSE(reg, tf.reshape(pred_reg, [-1]))
            loss = loss_cls + loss_reg
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"step: {(step + 1):0>5d}, loss: {loss:.2f}, loss_cls: {loss_cls:.2f}, loss_reg: {loss_reg:.2f}")


if __name__ == '__main__':
    main02()
