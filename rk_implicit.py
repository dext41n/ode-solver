import numpy as np
from newton import newton
from results import Result


def butcher_radau():
    """
    Butcherova tabulka pro implicitní Runge-Kuttovu metodu Radau (3 stage, řád 5).
    :return: (A, c, b) -- matice A, uzly c a váhy b metody
    """
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
    """
    Sestaví reziduální soustavu G(K) = 0 pro implicitní stage koeficienty K
    (3n neznámých), kterou pak řeší newton().
    :param K: zploštělý vektor stage koeficientů k1..k3, délka 3n
    :param f: funkce pravé strany, callable f(t, x)
    :param x: hodnota x na začátku kroku
    :param t: čas na začátku kroku
    :param h: délka kroku
    :param A: matice z Butcherovy tabulky
    :param c: uzly z Butcherovy tabulky
    :return: zploštělý vektor rezidua G, délka 3n
    """
    n = len(x)
    k_num = len(c)
    k = K.reshape(k_num,n)              #abych to zapsal elegantně jak v explicitním
    G = np.zeros((k_num,n))

    for i in range(k_num):
        x_stage = x + h*(A[i]@k)
        G[i] = k[i] - f(t+c[i]*h, x_stage)
    return G.flatten()


def radau_step(f, x, t, h, A, c, b, K_prev=None):
    """
    Spočítá jeden krok radau metody.
    :param f: callable funkce
    :param x: současný bod
    :param t: současný čas
    :param h: délka kroku
    :param A: matice A z Butcherovy tabulky
    :param c: vektor c z Butcherovy tabulky
    :param b: vektor b z Butcherovy tabulky
    :param K_prev: vektor K z minulého kroku, použitý jako odhad pro newtona
    :return: nový x, nový K, použitý krok
    """
    n = len(x)
    h_try = h
    if K_prev is None:
        f0 = f(t, x)
        K_prev = np.tile(f0,3)

    while h_try > 1e-9:
        # g je už opravdu ta funkci vektoru K, 3n proměnných
        def g(K):
            return count_coefs(K, f, x, t, h_try, A, c)

        convergence, iters, K_new = newton(g,K_prev)
        if convergence:
            k = K_new.reshape(3,n)
            x_new = x + h_try*(b@k)
            return x_new, K_new, h_try
        else:
            h_try /= 2
    raise RuntimeError(f"Newton nezkonvergoval ani při h={h_try}")


def radau(f, x0, t0, t_end, h):
    """
    Řeší rovnici implicitní runge kutta metodou, Radau IIA
    :param f: callable
    :param x0: počáteční podmínka
    :param t0: počátenčí čas
    :param t_end: koncový čas
    :param h: délka kroku, tu je povinná
    :return: objekt Results
    """
    time_eps = 1e-9
    x0 = np.atleast_1d(x0)
    x = x0
    t = t0
    sol = Result(x0,t,f(t,x0))
    A, c, b = butcher_radau()
    K_new = None

    while t < t_end:
        step = min(h,t_end-t)
        if step < time_eps:
            #udělá poslední super malý krok eulerem
            x_new = x + step * f(t, x)
            t = t + step

        else:
            x_new, K_new, h_try = radau_step(f, x, t, step, A, c, b, K_prev=K_new)
            t = t + h_try
        x = x_new
        sol.add(x, t, f(t,x))

    return sol










