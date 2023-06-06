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


def func00():
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y


@tf.function
def func01():
    """
    ValueError: tf.function-decorated function tried to create variables on non-first call.
    因为 tf.function可能会对一段Python函数进行多次执行来构图，在多次执行的过程中，同样的Variable被创建了多次，产生错误．
    这其实也是一个很容易混乱的概念，在eager mode下一个Variable是一个Python object，所以会在执行范围外被销毁．但是在 tf.function的装饰下，Variable变成了 tf.Variable，是在Graph中持续存在的．
    把一个在eager mode下正常执行的函数转换到Tensorflow图形式，需要一边思考着计算图一边构建程序．

    因为tf.function可能会对一段python代码进行多次执行来进行graph的构建，在多次的执行过程中，Variable被创建了多次，
    而tensorflow 2.x文档中明确指出*State (like tf.Variable objects) are only created the first time the function f is called.*。所以就报错了。
    因此我们在构建被tf.function修饰的函数时，一定要记得保证每一个tf.Variable变量只被创建一次，否则就有可能报错。

    以下两种情况不适合使用 tf.function 进行修饰：
    1.函数本身的计算非常简单，那么构建计算图本身的时间就会相对非常浪费；
    2.当我们需要在函数之中定义 tf.Variable 的时候，因为 tf.function 可能会被调用多次，因此定义 tf.Variable 会产生重复定义的情况。

    """
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y


class Func02:
    def __init__(self):
        self._b = None

    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)  # 保证tf.Variable变量只被创建一次
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y


def func03(a, b):
    print('this runs at trace time; a is', a, 'and b is', b)
    return b


def func_test01():
    print(func03(1, tf.constant(1)))
    print("-" * 36)
    print(func03(1, tf.constant(2)))
    print("-" * 36)
    print(func03(2, tf.constant(1)))
    print("-" * 36)
    print(func03(2, tf.constant(2)))


@tf.function
def func04(x):
    return tf.abs(x)


def func_test02():
    f1 = func04.get_concrete_function(1)
    f2 = func04.get_concrete_function(2)
    print(f1 is f2)


def main():
    func_test02()
    # func02 = Func02()
    # func02()


if __name__ == '__main__':
    main()
