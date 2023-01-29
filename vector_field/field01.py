import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def debug01():
    n = 8

    # 二维网格坐标
    X, Y = np.mgrid[0:n, 0:n]

    # U,V 定义方向
    U = X + 1
    V = Y + 1

    # C 定义颜色
    C = X + Y

    plt.quiver(X, Y, U, V, C)
    plt.show()


def debug02():
    n = 8

    # 二维网格坐标
    X, Y = np.mgrid[-n:n, -n:n]

    # U,V 定义方向
    U = -Y
    V = X

    # C 定义颜色
    C = X + Y

    plt.quiver(X, Y, U, V, C)
    plt.show()


def debug03():
    n = 8

    # 二维网格坐标
    X, Y = np.mgrid[-n:n, -n:n]

    # U,V 定义方向
    U = -Y / (X ** 2 + Y ** 2)
    V = X / (X ** 2 + Y ** 2)

    # C 定义颜色
    C = X + Y

    plt.quiver(X, Y, U, V, C)
    plt.show()


def debug04():
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

    plt.show()


if __name__ == '__main__':
    debug04()
