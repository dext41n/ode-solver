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


def solve_ivp(f, x0, t0, t_end, method = "RK45", graph = True, **kwargs):
    """
    Funkce pro řešení rovnice x' = f(t,x) spolu s počáteční podmínkou x(t0) = x0.
    :param f: funkce callable
    :param x0: počáteční podmínka
    :param t0: počáteční čas
    :param t_end: koncový čas
    :param method: "Euler", "ImplicitEuler", "RK45", "Radau", default je RK45
    :param graph: True/False, jestli má být výstupem i graf
    :param kwargs: Doplňující parametry pro spefické metody, pro Eulery a Radou krok, a pro RK45 lze dodat max_step, atol, rtol, min_step a
    adpative, kde lze zvolit pomcí True/False jestlí má být použit adaptivní krok, pokud false, musí být uveden max_step.
    :return: Result objekt
    """
    if method == "Euler":
        sol =  euler(f, x0, t0, t_end, kwargs['h'], implicit=False)
    elif method == "ImplicitEuler":
        sol = euler(f, x0, t0, t_end, kwargs['h'], implicit=True)
    elif method == "RK45":
        sol = rk45_explicit(f, x0, t0, t_end,
                             max_step=kwargs.get("max_step"),
                             atol=kwargs.get("atol", 1e-6),
                             rtol=kwargs.get("rtol", 1e-3),
                             min_step=kwargs.get("min_step", 1e-12),
                             adaptive=kwargs.get("adaptive", True))
    elif method == 'Radau':
        sol = radau(f, x0, t0, t_end, kwargs['h'])
    else:
        raise ValueError(f"Neznámá metoda '{method}'")

    if graph:
        x, t = sol.as_arrays()
        davinci(t,x)

    return sol

