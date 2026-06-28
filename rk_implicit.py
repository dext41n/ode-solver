import numpy as np
from newton import newton


def butcher_radau():
    """koeficenty butcherovy tabulk, vychází z literatur, odvození přes kvadraturu"""
    c = np.array([
        (4 - np.sqrt(6)) / 10,
        (4 + np.sqrt(6)) / 10,
        1.0,
    ])

    A = np.array([
        [(88 - 7 * np.sqrt(6)) / 360,       (296 - 169 * np.sqrt(6)) / 1800,    (-2 + 3 * np.sqrt(6)) / 225],
        [(296 + 169 * np.sqrt(6)) / 1800,   (88 + 7 * np.sqrt(6)) / 360,        (-2 - 3 * np.sqrt(6)) / 225],
        [(16 - np.sqrt(6)) / 36,            (16 + np.sqrt(6)) / 36,             1 / 9],
    ])

    b = np.array([
        (16 - np.sqrt(6)) / 36,
        (16 + np.sqrt(6)) / 36,
        1 / 9,
    ])

    return A, c, b


def count_coefs(K, f, x, t, h, A, c):
    n = len(x)
    k_num = len(c)
    k = K.reshape(k_num,n)              #abych to zapsal elegantně jak v explicitním
    G = np.zeros((k_num,n))

    for i in range(k_num):
        x_stage = x + h*(A[i]@k)
        G[i] = k[i] - f(t+c[i]*h, x_stage)
    return G.flatten()


def radau_step(f, x, t, h, A, c, b, K_prev=None):
    n = len(x)

    def g(K):
        return count_coefs(K, f, x, t, h, A, c)

    if K_prev is None:
        f0 = f(t, x)
        K_prev = np.array([f0,f0,f0])

    convergence, iters, K_new = newton(g,K_prev)
    k = K_new.reshape(3,n)
    x_new = x + h*(b@k)

    return x_new, K_new




