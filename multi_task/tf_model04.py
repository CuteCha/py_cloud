# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.feature_column import *

tf.disable_v2_behavior()


class MMOEClassifier(tf.estimator.Estimator):
    def __init__(self):
        pass


def debug01():
    x = tf.ones((2, 20))
    y = tf.layers.dense(x, 60)
    z = tf.keras.layers.Dense(60)(x)

    print(y.get_shape())
    print(z.get_shape())


if __name__ == '__main__':
    debug01()
