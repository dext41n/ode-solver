import matplotlib.pyplot as plt
from euler_methods import euler
from rk_implicit import radau
from rk_explicit import rk45_explicit


def davinci(t, x):
    fig, ax = plt.subplots()
    ax.plot(t,x)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.show()


def solve_ivp(f, x0, t0, t_end, method='RK45', graph = True ,h=None, max_step=None, atol=1e-6, rtol=1e-3, adaptive = True, min_step = 1e-12):
    """
    Funkce pro řešení rovnice x' = f(t,x) spolu s počáteční podmínkou x(t0) = x0.
    :param f: funkce callable
    :param x0: počáteční podmínka
    :param t0: počáteční čas
    :param t_end: koncový čas
    :param method: "Euler", "ImplicitEuler", "RK45", "Radau", default je RK45
    :param graph: True/False, jestli má být výstupem i graf
    :param h: krok metody -- povinné pro Eulery a Radau
    :param max_step: maximální krok -- jen pro RK45
    :param atol: absolutní tolerance -- pouze pro RK45
    :param rtol: relativní tolerance -- pouze pro RK45
    :param adaptive: True/False, jestli se má použít adaptivní krok -- poouze RK45
    :param min_step: min. krok -- pouze RK45
    :return: Result objekt
    """
    if method == "Euler":
        sol =  euler(f, x0, t0, t_end, h, implicit=False)
    elif method == "ImplicitEuler":
        sol = euler(f, x0, t0, t_end, h, implicit=True)
    elif method == "RK45":
        sol = rk45_explicit(f, x0, t0, t_end, max_step, adaptive, atol, rtol, min_step)
    elif method == 'Radau':
        sol = radau(f, x0, t0, t_end, h)
    else:
        raise ValueError(f"Neznámá metoda '{method}'")

    if graph:
        x, t = sol.as_arrays()
        davinci(t,x)

    return sol

