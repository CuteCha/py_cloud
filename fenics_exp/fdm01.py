import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt


def exact_u(x, t):
    return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)


def debug01():
    L = 1
    T = 1
    N = 500
    J = 10
    dt = T / N
    dx = L / J
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    r = dt / (dx ** 2)

    mx, mt = np.meshgrid(x, t)
    u_exact = exact_u(mx, mt)

    plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlabel('x')
    plt.ylabel('t')
    ax.plot_surface(mx, mt, u_exact, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    print("done")


def forward_diff(r, J, N, x):
    u_pre = exact_u(x[1:-1], 0)
    data = np.array([[r] * (J - 1), [1 - 2 * r] * (J - 1), [r] * (J - 1)])
    diags = np.array([-1, 0, 1])
    A = scipy.sparse.spdiags(data, diags, J - 1, J - 1).toarray()

    u = list()
    u.append(u_pre)
    for _ in range(N):
        u_cur = np.matmul(A, u_pre)
        # print(u_cur)
        u.append(u_cur)
        u_pre = u_cur

    v = np.array([[0] * (N + 1)]).T
    # u.reverse()
    u = np.array(u)
    u = np.column_stack((v, u))
    u = np.column_stack((u, v))

    return u


def debug02():
    L = 1
    T = 1
    N = 500
    J = 10
    dt = T / N
    dx = L / J
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    r = dt / (dx ** 2)

    mx, mt = np.meshgrid(x, t)
    u_exact = exact_u(mx, mt)
    u_forward = forward_diff(r, J, N, x)
    print(u_exact.shape, u_forward.shape)

    plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlabel('x')
    plt.ylabel('t')
    ax.plot_surface(mx, mt, u_forward, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    print("done")


def debug03():
    L = 1
    T = 1
    N = 500
    J = 10
    dt = T / N
    dx = L / J
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, T + dt, dt)
    r = dt / (dx ** 2)

    mx, mt = np.meshgrid(x, t)
    u_exact = exact_u(mx, mt)
    u_forward = forward_diff(r, J, N, x)
    print(u_exact.shape, u_forward.shape)

    plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlabel('x')
    plt.ylabel('t')
    ax.plot_surface(mx, mt, u_forward, rstride=1, cstride=1, cmap='rainbow', label="forward")
    ax.plot_surface(mx, mt, u_exact, rstride=1, cstride=1, cmap='Blues', label="exact")
    plt.show()
    print("done")


if __name__ == '__main__':
    debug03()
