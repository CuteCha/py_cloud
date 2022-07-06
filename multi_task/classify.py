from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2

'''
python classify.py -i data/examples/black_dress.jpg -m logs/h5/fashionnet.h5 -l logs/label_map/category.pickle -c logs/label_map/color.pickle
'''


def load_image(file_name):
    image = cv2.imread(file_name)
    image_show = imutils.resize(image, width=400)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image, image_show


def inference(file_name, trained_model, trained_category_lb, trained_color_lb):
    image, image_show = load_image(file_name)
    (category_proba, color_proba) = trained_model.predict(image)
    category_idx = category_proba[0].argmax()
    color_idx = color_proba[0].argmax()
    category_label = trained_category_lb.classes_[category_idx]
    color_label = trained_color_lb.classes_[color_idx]

    category_text = "category: {} ({:.2f}%)".format(category_label, category_proba[0][category_idx] * 100)
    color_text = "color: {} ({:.2f}%)".format(color_label, color_proba[0][color_idx] * 100)
    cv2.putText(image_show, color_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image_show, category_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("[INFO] {}".format(color_text))
    print("[INFO] {}".format(category_text))

    infer_file_name = "logs/pred/{}".format(file_name.strip().split("/")[-1])
    cv2.imwrite(infer_file_name, image_show)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="logs/h5/fashionnet.h5", help="path to trained model model")
    ap.add_argument("-l", "--cate", type=str, default="logs/label_map/category.pickle",
                    help="path to output category label binarizer")
    ap.add_argument("-c", "--color", type=str, default="logs/label_map/color.pickle",
                    help="path to output color label binarizer")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    trained_model = load_model(args["model"], custom_objects={"tf": tf})
    trained_category_lb = pickle.loads(open(args["cate"], "rb").read())
    trained_color_lb = pickle.loads(open(args["color"], "rb").read())

    inference(args["image"], trained_model, trained_category_lb, trained_color_lb)


if __name__ == '__main__':
    main()
