# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.feature_column import *
import numpy as np

tf.disable_v2_behavior()  # tf.logging.set_verbosity(tf.logging.ERROR)


def test01():
    x = tf.placeholder("float", [None, 1], name="x")
    y = tf.placeholder("float", [None, 1], name="y")

    z = x + y

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run(
            z,
            feed_dict={x: np.random.rand(10, 1),
                       y: np.random.rand(10, 1)}
        )

        print(r)


def test02():
    x = tf.placeholder(dtype=tf.float32, shape=[10, 10], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[10, 20], name="y")
    z = tf.placeholder(dtype=tf.float32, shape=[10, 20], name="z")

    w0 = tf.Variable(initial_value=np.random.rand(10, 20), name="w0", dtype=tf.float32)
    w1 = tf.Variable(initial_value=np.random.rand(20, 20), name="w1", dtype=tf.float32)
    w2 = tf.Variable(initial_value=np.random.rand(20, 20), name="w2", dtype=tf.float32)

    shared_layer = tf.nn.relu(tf.matmul(x, w0))
    y_pred = tf.nn.relu(tf.matmul(shared_layer, w1))
    z_pred = tf.nn.relu(tf.matmul(shared_layer, w2))

    y_loss = tf.nn.l2_loss(y - y_pred)
    z_loss = tf.nn.l2_loss(z - z_pred)

    y_op = tf.train.AdamOptimizer().minimize(y_loss)
    z_op = tf.train.AdamOptimizer().minimize(z_loss)

    loss = [y_loss, z_loss]
    op = [y_op, z_op]
    num = len(op)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(30):
            k = i % num
            _, r_loss = session.run(
                [op[k], loss[k]],
                {
                    x: np.random.rand(10, 10) * 10,
                    y: np.random.rand(10, 20) * 10,
                    z: np.random.rand(10, 20) * 10
                })
            print("step: {}, task[{}]: ".format(i, k), r_loss)


def test03():
    x = tf.placeholder(dtype=tf.float32, shape=[10, 10], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[10, 20], name="y")
    z = tf.placeholder(dtype=tf.float32, shape=[10, 20], name="z")

    w0 = tf.Variable(initial_value=np.random.rand(10, 20), name="w0", dtype=tf.float32)
    w1 = tf.Variable(initial_value=np.random.rand(20, 20), name="w1", dtype=tf.float32)
    w2 = tf.Variable(initial_value=np.random.rand(20, 20), name="w2", dtype=tf.float32)

    shared_layer = tf.nn.relu(tf.matmul(x, w0))
    y_pred = tf.nn.relu(tf.matmul(shared_layer, w1))
    z_pred = tf.nn.relu(tf.matmul(shared_layer, w2))

    y_loss = tf.nn.l2_loss(y - y_pred)
    z_loss = tf.nn.l2_loss(z - z_pred)

    y_op = tf.train.AdamOptimizer().minimize(y_loss)
    z_op = tf.train.AdamOptimizer().minimize(z_loss)

    loss = [y_loss, z_loss]
    op = [y_op, z_op]

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(30):
            _, r_loss = session.run(
                [op, loss],
                {
                    x: np.random.rand(10, 10) * 10,
                    y: np.random.rand(10, 20) * 10,
                    z: np.random.rand(10, 20) * 10
                })
            print("step: {} ".format(i), r_loss)


if __name__ == '__main__':
    test03()
