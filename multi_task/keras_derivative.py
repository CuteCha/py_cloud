import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def f(x):
    return 3. * x ** 2 + 2. * x - 1


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_derivative(func, x, eps=1e-6):
    return (func(x + eps) - func(x - eps)) / (2. * eps)


def approximate_gradient(func, x1, x2, eps=1e-6):
    dg_x1 = approximate_derivative(lambda x: func(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: func(x1, x), x2, eps)
    return dg_x1, dg_x2


def app_grad_test():
    print(approximate_derivative(f, 1.))
    print(approximate_gradient(g, 2., 3.))


def tape_grad_test01():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        z = g(x1, x2)

    grad_z = tape.gradient(z, [x1, x2])
    print(grad_z)


def tape_grad_test02():
    x1 = tf.constant(2.0)
    x2 = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x1)
        tape.watch(x2)
        z = g(x1, x2)

    grad_z = tape.gradient(z, [x1, x2])
    print(grad_z)


def tape_grad_test02b():
    x1 = tf.constant(2.0)
    x2 = tf.constant(3.0)
    with tf.GradientTape() as tape:
        z = g(x1, x2)

    grad_z = tape.gradient(z, [x1, x2])
    print(grad_z)  # [None, None]


def tape_grad_test03():
    """
    对多个方程的同一参数求导
    注意：此时会将每个方程的求导结果相加
    """
    x = tf.Variable(5.0)
    with tf.GradientTape() as tape:
        z1 = 3 * x
        z2 = x ** 2

    grad_z = tape.gradient([z1, z2], x)
    print(grad_z)


def tape_grad_test04():
    x = tf.Variable(5.0)
    with tf.GradientTape(persistent=True) as tape:
        z1 = 3 * x
        z2 = x ** 2

    grad_z1 = tape.gradient(z1, x)
    grad_z2 = tape.gradient(z2, x)
    print(grad_z1)
    print(grad_z2)

    del tape


def tape_grad_test05():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)
    with tf.GradientTape(persistent=True) as tape:
        z1 = f(x1)
        z2 = g(x1, x2)

    grad_z1 = tape.gradient(z1, [x1, x2])
    grad_z2 = tape.gradient(z2, [x1, x2])
    print(grad_z1)
    print(grad_z2)

    del tape


def tape_grad_test05a():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        z1 = f(x1)
        z2 = g(x1, x2)

    grad_z = tape.gradient(z1 + z2, [x1, x2])
    print(grad_z)


def tape_grad_test05b():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        z1 = f(x1)
        z2 = g(x1, x2)
        z = z1 + z2

    grad_z = tape.gradient(z, [x1, x2])
    print(grad_z)


def tape_grad_test06():
    x1 = tf.Variable(2.0)
    x2 = tf.Variable(3.0)
    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape(persistent=True) as inner_tape:
            z = g(x1, x2)
        grad_z = inner_tape.gradient(z, [x1, x2])
    grad2_z = [outer_tape.gradient(inner_grad, [x1, x2]) for inner_grad in grad_z]
    print(grad_z)
    print(grad2_z)

    del inner_tape
    del outer_tape


def tape_sgd_test01():
    x = tf.Variable(0.0)
    lr = 0.001
    for _ in range(1000):
        with tf.GradientTape() as tape:
            z = f(x)
        grad_z = tape.gradient(z, x)
        x.assign_sub(lr * grad_z)

    print(x, f(x))  # (1/3, -4/3)


def tape_sgd_test02():
    x = tf.Variable(0.0)
    lr = 0.001
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    for _ in range(1000):
        with tf.GradientTape() as tape:
            z = f(x)
        grad_z = tape.gradient(z, x)
        optimizer.apply_gradients([(grad_z, x)])

    print(x, f(x))


def tape_sgd_test03():
    x = tf.Variable(0.0)
    y = tf.Variable(1.0)
    lr = 0.001
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    grad_z = []
    for _ in range(1000):
        with tf.GradientTape() as tape:
            t = x + y
            z = f(x)
        grad_z = tape.gradient(z, [x, y])
        optimizer.apply_gradients(zip(grad_z, [x, y]))

    print(grad_z)
    print(x, f(x))


def tape_sgd_test03b():
    x = tf.Variable(0.0)
    y = tf.Variable(1.0)
    lr = 0.001

    optimizer = keras.optimizers.SGD(learning_rate=lr)
    grad_z = []
    for _ in range(1000):
        with tf.GradientTape() as tape:
            t = x + y
            z = f(x)
        grad_z = tape.gradient(z, [x, y])
        optimizer.apply_gradients(zip(grad_z, [x, y]))

    print(grad_z)
    print(x, f(x))


def main():
    tape_sgd_test03b()


if __name__ == '__main__':
    main()
