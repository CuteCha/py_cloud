# -*- encoding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_orbit():
    t = np.linspace(0, 2 * np.pi, 10000)
    x = 5 * np.cos(t) + np.cos(12 * t)  # 38.4*10^4km 1.496×10^8km 390
    y = 5 * np.sin(t) + np.sin(12 * t)  # 6400km
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, "b")
    plt.grid(True)
    plt.show()


def main():
    plot_2d_orbit()


if __name__ == '__main__':
    main()
