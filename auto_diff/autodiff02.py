# -* encoding:utf-8 *-
import math

import numpy as np


class Node(object):
    def __init__(self, op, input_nodes, name):
        self.op = op
        self.input_nodes = input_nodes
        self.name = name
        self.grad = None
        self.value = None
        self.input_values = self.node2values()

        self.evaluate()

    def node2values(self):
        new_inputs = []
        for i in self.input_nodes:
            if isinstance(i, Node):
                i = i.value
            new_inputs.append(i)
        return new_inputs

    def evaluate(self):
        self.value = self.op.compute(self.node2values())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Node {}, input_value: {}, op:{}, value: {},grad: {}". \
            format(self.name, self.input_values, self.op.name, self.value, self.grad)


class Variable:
    def __init__(self, value=None):
        self.value = value
        self.grad = None
        self.next = None
        self.root = None

    def func(self, *args):
        return

    def func_grad(self, *args):
        return


class Placeholder(Variable):
    def __init__(self, size):
        super().__init__(self)  # 运行父类的构造函数
        self.size = size
        self.root = self  # root就是自己
        self.grad = 1


class Exp(Variable):
    def __init__(self, X):
        super().__init__()  # 继承父类
        X.next = self  # 作为上一个节点的下一步运算
        self.root = X.root  # 声明自变量，复合函数自变量都是前一函数的自变量

    def func(self, X):
        return np.exp(X)

    def func_grad(self, X):
        return np.exp(X)


class Sin(Variable):
    def __init__(self, X):
        super().__init__()
        X.next = self
        self.root = X.root

    def func(self, X):
        return np.sin(X)

    def func_grad(self, X):
        return np.cos(X)


class Cos(Variable):
    def __init__(self, X):
        super().__init__()
        X.next = self
        self.root = X.root

    def func(self, X):
        return np.cos(X)

    def func_grad(self, X):
        return -np.sin(X)


class Log(Variable):
    def __init__(self, X):
        super().__init__()
        X.next = self
        self.root = X.root

    def func(self, X):
        return np.log(X)

    def func_grad(self, X):
        return 1 / X


class Square(Variable):
    def __init__(self, X):
        super().__init__()
        X.next = self
        self.root = X.root

    def func(self, X):
        return np.square(X)

    def func_grad(self, X):
        return 2 * X


class Session:
    def run(self, operator, feed_dict):
        root = operator.root  # 计算起始点
        root.value = feed_dict[root]  # 传入自变量的数据
        while root.next is not operator.next:  # 计算到operator便停止计算
            root.next.value = root.next.func(root.value)  # 计算节点的值
            root.next.grad = root.grad * root.next.func_grad(root.value)  # 计算梯度
            root = root.next  # 去往下一个节点
        return root.value


def debug01():
    xs = Placeholder((2, 2))
    h1 = Square(xs)
    h2 = Log(h1)
    h3 = Sin(h2)
    h4 = Exp(h3)

    # 具体数据
    x = np.array([[1, 2], [3, 4]])
    # 建立Session计算
    sess = Session()
    out = sess.run(h4, feed_dict={xs: x})
    # 自动求导
    grad = h4.grad
    # 手动求导
    grad0 = np.exp(np.sin(np.log(np.square(x)))) * np.cos(np.log(np.square(x))) * 1 / np.square(x) * 2 * x
    print()
    print("自动求导:\n", grad)
    print("手动求导:\n", grad0)
    print("out:\n {}".format(out))


if __name__ == '__main__':
    debug01()
