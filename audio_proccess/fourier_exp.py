import numpy as np
import matplotlib.pyplot as plt


def basic_sin(k, x):
    return np.sin((2 * k - 1) * x)


def main():
    N = 1000
    n = 10
    x = np.linspace(-2 * np.pi, 2 * np.pi, num=N)
    # y = np.array([basic_sin(k, x) for k in range(1, 7)])
    # s = np.cumsum(y, axis=1)
    s = np.zeros(N)
    plt.figure(figsize=(8, 8 * n))
    for k in range(1, 1 + n):
        y = basic_sin(k, x)
        s += y
        plt.subplot(n, 1, k)
        plt.plot(x, s)
        plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
