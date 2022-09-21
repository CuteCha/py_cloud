import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt


def exact_u(x):
    '''
    -u_{xx}+0.25*pi^2*u=0.5*pi^2*sin(0.5*pi*x)  x in (0,1)
    u(0)=0
    u'(0)=0
    '''
    return np.sin(0.5 * np.pi * x) + 0.5 * np.exp(-0.5 * np.pi * x) - 0.5 * np.exp(0.5 * np.pi * x)


def force(x):
    return 0.5 * np.pi ** 2 * np.sin(0.5 * np.pi * x)


def gen_matrix(x, func, h, N):
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    f = np.zeros(N)
    r = 0.5 * np.pi
    q = r ** 2
    for i in range(1, N):
        # ea = (np.array([[1, -1], [-1, 1]]) / h + 0.25 * np.pi ** 2 * np.array([[2, 1], [1, 2]]) * h / 6)
        # ef = (np.array([func(x[i - 1]), func(x[i])]) * h / 2)
        ek = (np.array([[1, -1], [-1, 1]]) / h)
        em = (q * np.array([[2, 1], [1, 2]]) * h / 6)
        ef = np.array([np.pi * np.cos(r * x[i - 1]) - 2 * (np.sin(r * x[i]) - np.sin(r * x[i - 1])) / h,
                       -np.pi * np.cos(r * x[i]) + 2 * (np.sin(r * x[i]) - np.sin(r * x[i - 1])) / h])
        s = i - 1
        t = s + 2
        K[s:t, s:t] += ek
        M[s:t, s:t] += em
        f[s:t] += ef

    return K + M, f


def fem_u(x, func, h, N, u0):
    a, f = gen_matrix(x, func, h, N)
    # from pprint import pprint
    # pprint(a[:6, :6])
    # pprint(a[-6:, -6:])
    # pprint(f[:6])
    # pprint(f[-6:])
    A = a[1:, 1:]
    b = f[1:] - a[:, 0][1:] * u0
    u = np.linalg.solve(A, b)

    return np.concatenate((np.array([u0]), u))


def derivative():
    import sympy as sp
    x = sp.symbols('x')
    y = sp.sin(0.5 * sp.pi * x) + 0.5 * sp.exp(-0.5 * sp.pi * x) - 0.5 * sp.exp(0.5 * sp.pi * x)
    d2y = sp.diff(y, x, 2)
    print(d2y)
    print(y.evalf(subs={'x': 0.0}))
    print(d2y.evalf(subs={'x': 0.0}))
    print(sp.simplify(-d2y + 0.25 * sp.pi ** 2 * y))


def debug():
    print(exact_u(0))
    derivative()


def main():
    u0 = 0.0
    L = 1
    h = 0.01
    x = np.arange(0, L + h, h)
    N = len(x)
    print(N)
    print(x[:10])
    print(x[-10:])
    u_exact = exact_u(x)
    u_fem = fem_u(x, force, h, N, u0)

    plt.figure()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, u_fem, 'bx')
    plt.plot(x, u_exact, 'r-')
    plt.grid(True)
    plt.show()
    print("done")


if __name__ == '__main__':
    # debug()
    main()
