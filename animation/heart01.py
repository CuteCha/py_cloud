import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def example01():
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def line_space(b_lst):
        x = np.linspace(0, 10, 100)
        return x, x + b_lst

    def update(B):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        x, y = line_space(B)
        line.set_data(x, y)
        return line

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 20, 100), interval=50)
    ani.save("move.gif", writer='imgemagick', fps=10)
    plt.show()


def example02():
    t = np.arange(-10, 10.01, 0.01)
    x = 16 * (np.sin(t)) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'r-')
    plt.grid(True)
    plt.show()


def example03():
    t = np.arange(0, 1 + 0.01, 0.01)
    omega0 = 2 * np.pi
    x0 = np.cos(omega0 * t)
    y0 = np.sin(omega0 * t)
    omega1 = 500 * np.pi
    r1 = 0.2
    x1 = []
    y1 = []
    for s in t:
        cx = np.cos(omega0 * s)
        cy = np.sin(omega0 * s)
        # x1.append(cx + r1 * np.cos(omega1 * t))
        # y1.append(cy + r1 * np.sin(omega1 * t))
        x1.append(cx + r1 * np.cos(omega1 * s))
        y1.append(cy + r1 * np.sin(omega1 * s))
    plt.figure(figsize=(8, 8))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x0, y0, 'r-')
    plt.plot(x1, y1, 'g-')
    plt.grid(True)
    plt.show()


def example04():
    import matplotlib.animation as animation

    # 初始化画布
    fig = plt.figure()
    plt.grid(ls='--')

    # 绘制一条正弦函数曲线
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    crave_ani = plt.plot(x, y, 'red', alpha=0.5)[0]

    # 绘制曲线上的切点
    point_ani = plt.plot(0, 0, 'r', alpha=0.4, marker='o')[0]

    # 绘制x、y的坐标标识
    xtext_ani = plt.text(5, 0.8, '', fontsize=12)
    ytext_ani = plt.text(5, 0.7, '', fontsize=12)
    ktext_ani = plt.text(5, 0.6, '', fontsize=12)

    # 计算切线的函数
    def tangent_line(x0, y0, k):
        xs = np.linspace(x0 - 0.5, x0 + 0.5, 100)
        ys = y0 + k * (xs - x0)
        return xs, ys

    # 计算斜率的函数
    def slope(x0):
        num_min = np.sin(x0 - 0.05)
        num_max = np.sin(x0 + 0.05)
        k = (num_max - num_min) / 0.1
        return k

    # 绘制切线
    k = slope(x[0])
    xs, ys = tangent_line(x[0], y[0], k)
    tangent_ani = plt.plot(xs, ys, c='blue', alpha=0.8)[0]

    # 更新函数
    def updata(num):
        k = slope(x[num])
        xs, ys = tangent_line(x[num], y[num], k)
        tangent_ani.set_data(xs, ys)
        point_ani.set_data(x[num], y[num])
        xtext_ani.set_text('x=%.3f' % x[num])
        ytext_ani.set_text('y=%.3f' % y[num])
        ktext_ani.set_text('k=%.3f' % k)
        return [point_ani, xtext_ani, ytext_ani, tangent_ani, k]

    ani = animation.FuncAnimation(fig=fig, func=updata, frames=np.arange(0, 100), interval=100)
    ani.save('./logs/sin_x.gif')
    plt.show()


def example05():
    fig, ax = plt.subplots(figsize=(8, 8))
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    frames = np.linspace(0, 2 * np.pi, 128)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save('./logs/sin_o.gif')
    plt.show()


def example06():
    import turtle

    pen = turtle.Turtle()

    def curve():
        for i in range(200):
            pen.right(1)
            pen.forward(1)

    def heart():
        pen.fillcolor('red')
        pen.begin_fill()
        pen.left(140)
        pen.forward(113)
        curve()
        pen.left(120)
        curve()
        pen.forward(112)
        pen.end_fill()

    def txt():
        pen.up()
        pen.setpos(-68, 95)
        pen.down()
        pen.color('lightgreen')
        pen.write("GeeksForGeeks", font=("Verdana", 12, "bold"))

    heart()
    txt()
    pen.ht()


def example07():
    alpha = 100
    t = np.arange(0, 3, 0.001)
    s = np.power(t, 2 / 3) + 0.9 * np.sqrt(3.3 - np.power(t, 2)) * np.sin(alpha * np.pi * t)
    x = np.concatenate((-t[::-1], t[1:]))
    y = np.concatenate((s[::-1], s[1:]))
    plt.figure(figsize=(8, 8))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-3, 3, -2, 4])
    plt.plot(x, y, 'r-')
    plt.grid(True)
    plt.show()


def example08():
    fig, ax = plt.subplots(figsize=(8, 8))
    ln, = plt.plot([], [], 'r-')

    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 4)
        return ln,

    def update(frame):
        alpha = frame
        t = np.arange(0, 3, 0.001)
        s = np.power(t, 2 / 3) + 0.9 * np.sqrt(3.3 - np.power(t, 2)) * np.sin(alpha * np.pi * t)
        x = np.concatenate((-t[::-1], t[1:]))
        y = np.concatenate((s[::-1], s[1:]))
        ln.set_data(x, y)
        return ln,

    frames = np.linspace(0, 100, 128)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save('./logs/heart_08.gif')
    plt.show()


def example09():
    from math import pi, sin, cos
    X = []
    Y = []
    n = 17
    alpha = 2 * pi / n
    X.append(0.0)
    Y.append(0.0)

    for i in range(0, n + 1):
        theta = alpha * i
        X.append(sin(theta))
        Y.append(cos(theta))
        X.append(0.0)
        Y.append(0.0)
        X.append(sin(theta))
        Y.append(cos(theta))

    plt.plot(X, Y)
    plt.axis('equal')
    plt.show()


# Bezout's Lemma
def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)


# Extended Euclidean algorithm
def ext_euclid(a, b):
    old_s, s = 1, 0
    old_t, t = 0, 1
    old_r, r = a, b
    if b == 0:
        return 1, 0, a
    else:
        while r != 0:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
            old_t, t = t, old_t - q * t
    print("a*s+b*t=d")
    print(f"({a})*({old_s})+({b})*({old_t})={old_r}")
    return old_s, old_t, old_r


def example10():
    print(gcd(3, 10))
    print(ext_euclid(3, 10))


if __name__ == '__main__':
    example10()
