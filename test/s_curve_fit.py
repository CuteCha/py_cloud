import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate


def s_curve(x, a, c, w, b):
    y = a / (1 + np.exp(-w * (x - c))) + b
    return y


def fit(xdata, ydata, name):
    x = np.arange(min(xdata), max(xdata), 0.1)

    # s曲线
    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(f"[a, c, w, b]= {popt}")

    ys = s_curve(x, *popt)

    # 分段线性
    f = interpolate.interp1d(xdata, ydata, kind='linear')
    yi = f(x)

    plt.figure()
    plt.plot(xdata, ydata, "ob", label="ori")  # 数据散点图
    plt.plot(x, ys, "-r", label="s_curve")  # s曲线图
    plt.plot(x, yi, "-g", label="seg_linear")  # 分度线性图
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./{name}.png")


def main():
    xdata1 = np.array([1, 2, 5, 10, 11, 12])
    ydata1 = np.array([75, 80, 90, 100, 100, 100])
    name1 = "data1"
    fit(xdata1, ydata1, name1)
    print("-" * 72)
    xdata2 = np.array([1, 3, 5, 6, 7])
    ydata2 = np.array([75, 90, 100, 100, 100])
    name2 = "data2"
    fit(xdata2, ydata2, name2)


if __name__ == '__main__':
    main()
