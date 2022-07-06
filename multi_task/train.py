from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from fashionnet import FashionNet
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

matplotlib.use("Agg")

'''
python train.py -d data/dataset -m logs/h5/fashionnet.h5 -p logs/pb -l logs/label_map/category.pickle -c logs/label_map/color.pickle -f logs/fig
'''

EPOCHS = 50
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


def save_model(model, category_lb, color_lb, args):
    model.save(args["pbmodel"])
    model.save(args["model"])
    save_label(args["cate"], category_lb)
    save_label(args["color"], color_lb)


def save_label(file_name, label_dict):
    with open(file_name, "wb") as fr:
        fr.write(pickle.dumps(label_dict))


def show_metric(H, metrics, fig_name):
    num = len(metrics)
    y_label = fig_name.strip().split("/")[-1]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(num, 1, figsize=(13, 13))
    for (i, name) in enumerate(metrics):
        ax[i].set_title(fig_name)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel(y_label)
        ax[i].plot(np.arange(0, EPOCHS), H.history[name], label=name)
        ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + name], label="val_" + name)
        ax[i].legend()

    plt.tight_layout()
    plt.savefig("{}.png".format(fig_name))
    plt.close()


def plt_metric(h, args):
    loss_names = ["loss", "category_output_loss", "color_output_loss"]
    show_metric(h, loss_names, "{}/loss".format(args["fig"]))

    accuracy_names = ["category_output_accuracy", "color_output_accuracy"]
    show_metric(h, accuracy_names, "{}/acc".format(args["fig"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="data/dataset", help="path to input dataset (i.e., directory of images)")
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
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

    h = model.fit(train_x, {"category_output": train_category_y, "color_output": train_color_y},
                  validation_data=(test_x, {"category_output": test_category_y, "color_output": test_color_y}),
                  epochs=EPOCHS, verbose=1)

    save_model(model, category_lb, color_lb, args)
    plt_metric(h, args)


if __name__ == '__main__':
    main()
