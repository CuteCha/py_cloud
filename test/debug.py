import numpy as np
import matplotlib.pyplot as plt

import sympy


def debug02():
    xjm1, xj, xjp1 = sympy.symbols(['x_{j-1}', 'x_{j}', 'x_{j+1}'])
    x, h, eps = sympy.symbols(['x', 'h', '\epsilon'])
    rho = lambda x: sympy.sin(sympy.pi * x)

    A = - sympy.integrate(rho(x) * (x - xjm1) / h, (x, xjm1, xj))
    B = - sympy.integrate(rho(x) * (xjp1 - x) / h, (x, xj, xjp1))

    C = sympy.simplify(A + B)
    print(C)
    print(type(C))


def debug01():
    t = np.linspace(0, 0.06, 1000)
    f = 50
    u_max = 220 * np.sqrt(2)
    phi_list = np.array([0, -2 * np.pi / 3, -4 * np.pi / 3])
    u_list = [u_max * np.cos(2 * np.pi * f * t + phi) for phi in phi_list]
    plt.figure(figsize=(10, 5))
    for u in u_list:
        plt.plot(t, u)
    plt.xlabel('t(s)')
    plt.ylabel('U(v)')
    plt.grid()
    plt.show()

    print("done")


def debug03():
    x = np.linspace(0, 1, 1000)
    u1 = -x ** 2 / 2 + x
    u2 = 16 / (np.pi ** 3) * np.sin(np.pi / 2 * x)
    u3 = 16 / (np.pi ** 3) * np.sin(np.pi / 2 * x) + 16 / (27 * np.pi ** 3) * np.sin(np.pi / 2 * x)
    plt.figure(figsize=(10, 5))
    plt.plot(x, u1, 'k--')
    plt.plot(x, u2, 'r-')
    plt.plot(x, u3, 'b-')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid()
    plt.show()


def debug04():
    x = input("content: ")
    print(type(x))
    print(x)
    y = x.encode()
    print(type(y))
    print(y)


class Car(object):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def add_gas(v):
        print(f"add gas {v}L")

    def get_name(self):
        print(f"name: {self.name}")

    @classmethod
    def fun(cls):
        print(f"class method: {cls}")


def debug05():
    c = Car("benz")
    c.add_gas(2)
    c.get_name()
    Car.fun()
    Car.get_name(c)
    c.fun()


if __name__ == '__main__':
    debug05()
