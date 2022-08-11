import numpy as np
import matplotlib.pyplot as plt


def main():
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


if __name__ == '__main__':
    main()
