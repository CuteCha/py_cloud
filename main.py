# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def test6():
    def rect(t, T):
        Th = T / 2.0
        return np.where((t > -Th) & (t < Th), 1.0, 0.0)

    def rect2(t, tao, T):
        N = len(t)
        tao_h = tao / 2.0
        res = np.where((t > -tao_h) & (t < tao_h), 1.0, 0.0)
        k = 1
        while k * T < t[-1]:
            res += np.where((t > k * T - tao_h) & (t < k * T + tao_h), 1.0, 0.0)
            res += np.where((t > -k * T - tao_h) & (t < -k * T + tao_h), 1.0, 0.0)
            k += 1

        return res

    tc = np.arange(-2 * np.pi, 2 * np.pi, 1e-4)
    xc = np.cos(tc) + 1.0

    sc = rect2(tc, 0.5, np.pi / 4.0)
    xs = sc * xc

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(tc, xc, 'b--', label='Origin')
    plt.plot(tc, xs, 'r-', label='RectSam')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.legend(loc='lower left')

    plt.subplot(212)
    # plt.plot(tc, xc, 'r--', label='Original')
    plt.plot(tc, xs, 'b-', label='Reconstruct')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal(Original vs Reconstruct)")
    plt.legend()

    plt.show()


def print_hi(name):
    print('Hi, {}'.format(name))


if __name__ == '__main__':
    print_hi('PyCharm')
    test6()
