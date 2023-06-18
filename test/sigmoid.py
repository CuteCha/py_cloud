import numpy as np
import matplotlib.pyplot as plt


def sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = 1.0 / (1 + np.exp(-x))

    plt.plot(x, y, label="sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def fit_sigmoid():
    x = np.array([1, 6, 100])
    y = np.array([75, 90, 100])
    plt.plot(x, y, label="eval")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def fit00():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    x = np.linspace(1, 50, 50)
    f = np.poly1d([2, 5, 10])
    y = f(x)
    x = np.reshape(x, (-1, 1))

    poly_reg = PolynomialFeatures(degree=2)
    x_ploy = poly_reg.fit_transform(x)
    lin_reg = LinearRegression()
    lin_reg.fit(x_ploy, y)

    print(lin_reg.coef_)

    y_pred = lin_reg.predict(x_ploy)

    plt.plot(x, y, "x-r", label="ori")
    plt.plot(x, y_pred, c='blue', label="reg")
    plt.legend()
    plt.show()


def fit01():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    x = np.array([1, 6, 100])
    y = np.array([75, 90, 100])
    x = np.reshape(x, (-1, 1))

    poly_reg = PolynomialFeatures(degree=1)
    x_ploy = poly_reg.fit_transform(x)
    lin_reg = LinearRegression()
    lin_reg.fit(x_ploy, y)

    print(lin_reg.coef_)

    x_pred = np.linspace(1, 100, 100)
    x_pred_ploy = poly_reg.fit_transform(np.reshape(x_pred, (-1, 1)))
    y_pred = lin_reg.predict(x_pred_ploy)

    plt.plot(x, y, "x-r", label="ori")
    plt.plot(x_pred, y_pred, c='blue', label="reg")
    plt.legend()
    plt.show()


def fit02():
    import scipy.interpolate as spi

    X = np.array([1, 6, 100])
    Y = np.array([75, 90, 100])

    x = np.arange(1, 101)
    ipo1 = spi.splrep(X, Y, k=1)
    iy1 = spi.splev(x, ipo1)

    # ipo3 = spi.splrep(X, Y, k=3)
    # iy3 = spi.splev(x, ipo3)

    plt.plot(X, Y, label="eval")
    plt.plot(x, iy1, label="iy1")
    # plt.plot(x, iy3, label="iy3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def fit03():
    from tensorflow import keras

    x = np.array([1, 6, 100])
    y = np.array([75, 90, 100])

    model = keras.models.Sequential([
        keras.layers.Dense(1, input_dim=1, kernel_initializer='normal', activation='sigmoid'),
        keras.layers.Dense(1, input_dim=1, kernel_initializer='normal'),
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x, y, epochs=100)

    print(model.weights)

    u = np.arange(1, 101, 1)
    v = model.predict(u)

    plt.plot(x, y, "x-r", label="ori")
    plt.plot(u, v, c='blue', label="reg")
    plt.legend()
    plt.show()


def fit04():
    from sympy import symbols, Eq, solve, exp, nsolve
    x, y, z = symbols('x,y,z')
    eqs = [Eq(x / 1 + exp(-(y + z)), 75),
           Eq(x / 1 + exp(-(6 * y + z)), 90),
           Eq(x / 1 + exp(-(100 * y + z)), 100)]
    print(solve(eqs, [x, y, z]))

    # x, y = symbols('x y')
    # eqs = [Eq(x ** 2 / 4 + y ** 2, 1),
    #        Eq((x - 0.2) ** 2 - y, 3)]
    # print(solve(eqs, [x, y]))


def s_curve(x, a, c, w, b):
    y = a / (1 + np.exp(-w * (x - c))) + b
    return y


def fit05():
    from scipy.optimize import curve_fit
    xdata = np.array([1, 2, 5, 10, 11, 12])
    ydata = np.array([75, 80, 90, 100, 100, 100])
    x = np.arange(1, 12, 0.1)

    # xdata = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
    # ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])
    # x = np.linspace(1, 15, 50)
    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(popt)

    y = s_curve(x, *popt)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.grid(True)
    plt.legend()
    plt.show()


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def fit06():
    from scipy.optimize import curve_fit
    # xdata = np.array([1, 2, 5, 6, 7, 8, 10])
    # ydata = np.array([75, 80, 90, 97, 98, 99.8, 100])
    xdata = np.array([1, 2, 5, 10])
    ydata = np.array([75, 80, 90, 100])
    x = np.arange(1, 15, 0.1)

    # xdata = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
    # ydata = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])
    # x = np.linspace(1, 15, 50)

    p, e = curve_fit(piecewise_linear, xdata, ydata)
    print(p)

    y = piecewise_linear(x, *p)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit07():
    from scipy.interpolate import interp1d
    xdata = np.array([1, 2, 5, 10, 11, 12])
    ydata = np.array([75, 80, 90, 100, 100, 100])
    x = np.arange(1, 12, 0.1)

    f = interp1d(xdata, ydata, kind='linear')
    y = f(x)
    # print(f(np.array([6, 7, 8, 9])))

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit08():
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    xdata = np.array([1, 2, 5, 10, 11, 12])
    ydata = np.array([75, 80, 90, 100, 100, 100])
    x = np.arange(1, 12, 0.1)
    #[54.13901852  0.94792998  0.31875893 47.96351715]

    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(popt)

    y = s_curve(x, *popt)
    f = interp1d(xdata, ydata, kind='linear')
    yi = f(x)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.plot(x, yi, "-g", label="linear")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit09():
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    xdata = np.array([1, 3, 5, 6, 7])
    ydata = np.array([75, 90, 100, 100, 100])
    x = np.arange(1, 7, 0.1)

    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(popt)

    y = s_curve(x, *popt)
    f = interp1d(xdata, ydata, kind='linear')
    yi = f(x)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.plot(x, yi, "-g", label="linear")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit10():
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    xdata = np.array([1, 3, 5, 6, 7])
    ydata = np.array([75, 90, 100, 100, 100])
    x = np.arange(1, 7, 0.1)

    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(popt)

    y = s_curve(x, *popt)
    f = interp1d(xdata, ydata, kind='quadratic')
    yi = f(x)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="reg")
    plt.plot(x, yi, "-g", label="linear")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit11():
    from scipy.optimize import curve_fit
    from scipy import interpolate
    xdata = np.array([1, 3, 5, 6, 7, 8, 9, 10])
    ydata = np.array([75, 90, 100, 100, 100, 100, 100, 100])
    x = np.arange(1, 10, 0.1)

    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    popt, pcov = curve_fit(s_curve, xdata, ydata, p0, method='dogbox', maxfev=5000)
    print(popt)

    tck = interpolate.splrep(xdata, ydata, s=0, k=3)
    y = interpolate.BSpline(*tck)(x)

    f = interpolate.interp1d(xdata, ydata, kind='linear')
    yi = f(x)

    plt.plot(xdata, ydata, "ob", label="ori")
    plt.plot(x, y, "-r", label="b_spline")
    plt.plot(x, yi, "-g", label="linear")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # sigmoid()
    # fit_sigmoid()
    fit08()
    # fit11()
