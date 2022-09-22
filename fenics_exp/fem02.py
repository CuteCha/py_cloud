import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt


def exact_u(x):
    '''
    u_{xx}+u=-x  x in (0,1)
    u(0)=0
    u(1)=0
    '''
    return np.sin(x) / np.sin(1) - x


def force(x):
    return x


def gen_matrix(x, func, h, N):
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    f = np.zeros(N)
    for i in range(1, N):
        ek = (np.array([[1, -1], [-1, 1]]) / h)
        em = (np.array([[2, 1], [1, 2]]) * h / 6)
        ef = (h * np.array([h / 6 + x[i - 1] / 2, h / 3 + x[i - 1] / 2]))
        s = i - 1
        t = s + 2
        K[s:t, s:t] += ek
        M[s:t, s:t] += em
        f[s:t] += ef

    return K - M, f


def fem_u(x, func, h, N, u0, un):
    a, f = gen_matrix(x, func, h, N)
    # from pprint import pprint
    # pprint(a[:6, :6])
    # pprint(a[-6:, -6:])
    # pprint(f[:6])
    # pprint(f[-6:])
    A = a[1:-1, 1:-1]
    b = f[1:-1] - a[:, 0][1:-1] * u0 - a[:, -1][1:-1] * un
    # u = np.linalg.solve(A, b)
    u = np.linalg.lstsq(A, b, rcond=None)

    return np.concatenate((np.array([u0]), u[0], np.array([un])))


def fam_u():
    import scipy.sparse

    h = 0.01
    N = 99
    b = -h ** 2 * np.arange(h, 1, h)
    data = np.array([[1] * N, [h ** 2 - 2] * N, [1] * N])
    diags = np.array([-1, 0, 1])
    A = scipy.sparse.spdiags(data, diags, N, N).toarray()
    u = np.linalg.lstsq(A, b, rcond=None)

    return np.concatenate((np.array([0]), u[0], np.array([0])))


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
    un = 0.0
    L = 1
    h = 0.01
    x = np.arange(0, L + h, h)
    N = len(x)
    print(N)
    print(x[:10])
    print(x[-10:])
    u_exact = exact_u(x)
    u_fem = fem_u(x, force, h, N, u0, un)
    u_fam = fam_u()

    plt.figure(figsize=(8, 8))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, u_fem, 'bo', label='u_fem')
    plt.plot(x, u_fam, color="yellow", marker="x", label='u_fam')
    plt.plot(x, u_exact, 'r-', label='u_exact')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    print("done")


if __name__ == '__main__':
    # debug()
    main()
