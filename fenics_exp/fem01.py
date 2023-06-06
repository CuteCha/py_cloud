import numpy as np
from matplotlib import pyplot as plt


def exact_u(x):
    '''
    -u_{xx}+u=sin(x)  x in (0,1)
    u(0)=0
    u'(0)=0
    '''
    return np.sin(x) / 2 + np.exp(-x) / 4 - np.exp(x) / 4


def force(x):
    return np.sin(x)


def gen_matrix(x, func, h, N):
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    f = np.zeros(N)
    for i in range(1, N):
        ek = np.array([[1, -1], [-1, 1]]) / h
        em = np.array([[2, 1], [1, 2]]) * h / 6
        tmp = (np.sin(x[i]) - np.sin(x[i - 1])) / (h ** 2)
        ef = np.array([np.cos(x[i - 1]) / h - tmp, -np.cos(x[i]) / h + tmp]) * h
        # ef = np.array([func(x[i - 1]), func(x[i])]) * h / 2  # approximate
        s = i - 1
        t = s + 2
        K[s:t, s:t] += ek
        M[s:t, s:t] += em
        f[s:t] += ef

    return K + M, f


def fem_u(x, func, h, N, u0):
    a, f = gen_matrix(x, func, h, N)
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


def fdm_u(h, N, x_lst, func, u0, du0):
    B = np.array([[1, 0], [0, 1]])
    phi = np.array([u0, du0])
    res = list()
    res.append(phi[0])
    for i in range(N - 1):
        F = np.array([func(x_lst[i]), 0])
        phi = h * np.matmul(B, phi - F) + phi
        res.append(phi[0])

    return res


def fdm_r_u(h, N, x_lst, func, u0, du0):
    B = np.array([[1, 0], [0, 1]])
    t = h / 2
    A = np.array([[1, t], [t, 1]]) / (1 - t ** 2)
    phi = np.array([u0, du0])
    res = list()
    res.append(phi[0])
    for i in range(N - 1):
        F_c = np.array([func(x_lst[i]), 0])
        F_b = np.array([func(x_lst[i + 1]), 0])
        phi = np.matmul(A, t * np.matmul(B, phi - F_c - F_b) + phi)
        res.append(phi[0])

    return res


def main():
    u0 = 0.0
    du0 = 0
    L = 1
    h = 0.01
    x = np.arange(0, L + h, h)
    N = len(x)
    print(N)
    print(x[:10])
    print(x[-10:])
    u_exact = exact_u(x)
    # u_fem = fem_u(x, force, h, N, u0)
    u_fdm = fdm_u(h, N, x, force, u0, du0)
    u_fdm_r = fdm_r_u(h, N, x, force, u0, du0)

    plt.figure()
    plt.xlabel('x')
    plt.ylabel('u')
    # plt.plot(x, u_fem, 'bx', label="fem")
    plt.plot(x, u_exact, 'r-', label="exact")
    plt.plot(x, u_fdm, 'gx', label="fdm")
    plt.plot(x, u_fdm_r, 'bo', label="fdm_r")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("done")


if __name__ == '__main__':
    # debug()
    main()
