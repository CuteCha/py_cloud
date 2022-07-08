import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from fashionnet import FashionNet
from imutils import paths
import matplotlib
import numpy as np
import argparse
import random
import cv2
import os

matplotlib.use("Agg")

'''
python train.py -d data/dataset -m logs/h5/fashionnet.h5 -p logs/pb -l logs/label_map/category.pickle -c logs/label_map/color.pickle -f logs/fig
'''

EPOCHS = 5
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)


def load_train_data(file_name):
    image_paths = sorted(list(paths.list_images(file_name)))
    random.seed(101)
    random.shuffle(image_paths)

    data = []
    category_labels = []
    color_labels = []

    for imagePath in image_paths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        data.append(image)
        (color, cat) = imagePath.split(os.path.sep)[-2].split("_")
        category_labels.append(cat)
        color_labels.append(color)

    data = np.array(data, dtype="float") / 255.0
    print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(image_paths), data.nbytes / (1024 * 1000.0)))

    category_labels = np.array(category_labels)
    color_labels = np.array(color_labels)

    category_lb = LabelBinarizer()
    color_lb = LabelBinarizer()
    category_labels = category_lb.fit_transform(category_labels)
    color_labels = color_lb.fit_transform(color_labels)
    print("cate_num: {}, color_num: {}".format(len(category_lb.classes_), len(color_lb.classes_)))

    (train_x, test_x, train_category_y, test_category_y, train_color_y, test_color_y) = \
        train_test_split(data, category_labels, color_labels, test_size=0.2, random_state=101)

    return train_x, test_x, train_category_y, test_category_y, train_color_y, test_color_y, category_lb, color_lb


def input_fn(x, y, c):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, c)) \
        .shuffle(buffer_size=1000).batch(32) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    return dataset


def load_dataset(file_name):
    train_x, test_x, train_category_y, test_category_y, train_color_y, test_color_y, category_lb, color_lb = \
        load_train_data(file_name)

    ds_train = input_fn(train_x, train_category_y, train_color_y)
    ds_valid = input_fn(test_x, test_category_y, test_color_y)

    return ds_train, ds_valid, category_lb, color_lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="/data/code/py_work/multi_task/data/dataset",
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", default="logs/h5/fashionnet.h5", help="path to output h5 model")
    ap.add_argument("-p", "--pbmodel", default="logs/pb", help="path to output pb model")
    ap.add_argument("-l", "--cate", default="logs/label_map/category.pickle",
                    help="path to output category label binarizer")
    ap.add_argument("-c", "--color", default="logs/label_map/color.pickle", help="path to output color label binarizer")
    ap.add_argument("-f", "--fig", default="logs/fig", help="base filename for generated plots")
    args = vars(ap.parse_args())

    train_x, test_x, train_category_y, test_category_y, train_color_y, test_color_y, category_lb, color_lb = \
        load_train_data(args["dataset"])

    model = FashionNet.build(96, 96, len(category_lb.classes_), len(color_lb.classes_))

    losses = {"category_output": "categorical_crossentropy", "color_output": "categorical_crossentropy"}
    loss_weights = {"category_output": 1.0, "color_output": 1.0}
    opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

    h = model.fit(train_x, {"category_output": train_category_y, "color_output": train_color_y},
                  validation_data=(test_x, {"category_output": test_category_y, "color_output": test_color_y}),
                  epochs=EPOCHS, verbose=1)


@tf.function
def train_step(model, train_cate_loss, train_cate_metric, train_color_loss, train_color_metric,
               features, cates, colors, loss_func, optimizer):
    p = np.random.uniform()
    print("p={}".format(p))
    if p < 0.4:
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            cate_loss = loss_func(cates, predictions[0])
            color_loss = loss_func(colors, predictions[1])
        gradients = tape.gradient(cate_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    elif p < 0.5:
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(features, training=True)
            cate_loss = loss_func(cates, predictions[0])
            color_loss = loss_func(colors, predictions[1])
        gradients_cate = tape.gradient(cate_loss, model.trainable_variables)
        gradients_color = tape.gradient(color_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients_cate, model.trainable_variables))
        optimizer.apply_gradients(zip(gradients_color, model.trainable_variables))
        del tape

    elif p < 0.6:
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            cate_loss = loss_func(cates, predictions[0])
            color_loss = loss_func(colors, predictions[1])
            loss = cate_loss + color_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    else:
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            cate_loss = loss_func(cates, predictions[0])
            color_loss = loss_func(colors, predictions[1])
        gradients = tape.gradient(color_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_cate_loss.update_state(cate_loss)
    train_cate_metric.update_state(cates, predictions[0])
    train_color_loss.update_state(color_loss)
    train_color_metric.update_state(colors, predictions[1])


@tf.function
def valid_step(model, valid_cate_loss, valid_cate_metric, valid_color_loss, valid_color_metric,
               features, cates, colors, loss_func):
    predictions = model(features)
    cate_loss = loss_func(cates, predictions[0])
    color_loss = loss_func(colors, predictions[1])

    valid_cate_loss.update_state(cate_loss)
    valid_cate_metric.update_state(cates, predictions[0])
    valid_color_loss.update_state(color_loss)
    valid_color_metric.update_state(colors, predictions[1])


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="/data/code/py_work/multi_task/data/dataset",
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", default="logs/h5/fashionnet.h5", help="path to output h5 model")
    ap.add_argument("-p", "--pbmodel", default="logs/pb", help="path to output pb model")
    ap.add_argument("-l", "--cate", default="logs/label_map/category.pickle",
                    help="path to output category label binarizer")
    ap.add_argument("-c", "--color", default="logs/label_map/color.pickle", help="path to output color label binarizer")
    ap.add_argument("-f", "--fig", default="logs/fig", help="base filename for generated plots")
    args = vars(ap.parse_args())

    ds_train, ds_valid, category_lb, color_lb = load_dataset(args["dataset"])
    model = FashionNet.build(96, 96, len(category_lb.classes_), len(color_lb.classes_))

    optimizer = keras.optimizers.Nadam()
    loss_func = keras.losses.CategoricalCrossentropy()

    train_cate_loss = keras.metrics.Mean(name='train_cate_loss')
    train_cate_metric = keras.metrics.CategoricalCrossentropy(name='train_cate_metric')
    train_color_loss = keras.metrics.Mean(name='train_color_loss')
    train_color_metric = keras.metrics.CategoricalCrossentropy(name='train_color_metric')

    valid_cate_loss = keras.metrics.Mean(name='valid_cate_loss')
    valid_cate_metric = keras.metrics.CategoricalCrossentropy(name='valid_cate_metric')
    valid_color_loss = keras.metrics.Mean(name='valid_color_loss')
    valid_color_metric = keras.metrics.CategoricalCrossentropy(name='valid_color_metric')

    for epoch in range(5):
        for features, cates, colors in ds_train:
            train_step(model, train_cate_loss, train_cate_metric, train_color_loss, train_color_metric,
                       features, cates, colors, loss_func, optimizer)

        for features, cates, colors in ds_valid:
            valid_step(model, valid_cate_loss, valid_cate_metric, valid_color_loss, valid_color_metric,
                       features, cates, colors, loss_func)

        print("epoch:{}".format(epoch + 1))
        print("train: cate_loss:{}, cate_metric:{}, color_loss:{}, color_metric:{}".format(
            train_cate_loss.result(), train_cate_metric.result(), train_color_loss.result(), train_color_metric.result()
        ))
        print("valid: cate_loss:{}, cate_metric:{}, color_loss:{}, color_metric:{}".format(
            valid_cate_loss.result(), valid_cate_metric.result(), valid_color_loss.result(), valid_color_metric.result()
        ))
        print("=" * 72)

        train_cate_loss.reset_states()
        train_cate_metric.reset_states()
        train_color_loss.reset_states()
        train_color_metric.reset_states()
        valid_cate_loss.reset_states()
        valid_cate_metric.reset_states()
        valid_color_loss.reset_states()
        valid_color_metric.reset_states()


if __name__ == '__main__':
    run()
