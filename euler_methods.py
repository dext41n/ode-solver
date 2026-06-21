import numpy as np
from newton import newton


class Result:

    def __init__(self, x = None, t = None):
        self.x = [x]
        self.t = [t]

    def add(self, x_new, t_new):
        if self.x is None:
            self.x = [x_new]
        else:
            self.x.append(x_new)
        if self.t is None:
            self.t = [t_new]
        else:
            self.t.append(t_new)

    def as_arrays(self):
        return np.array(self.x), np.array(self.t)


def implicit_step(f, x_prev, t, h):
    t = t + h
    g = lambda x: x_prev + h * f(t, x) - x
    initial_guess = x_prev + h * f(t, x_prev)
    convergence, iter, x_new = newton(g, initial_guess)

    return x_new, t


def euler(f, x0, t0, t_end, h, implicit = False):
    """
    Řeší differenciální rovnici eulerovou explicitní/implicitní metodou
    :param f: callable funkce
    :param x0: počáteční vektor
    :param t0: počáteční čas
    :param t_end: konečný čas
    :param h: délka kroku
    :param implicit: True použije implicitní, False explicitního eulera
    :return: objekt výsledků
    """
    n_steps = int(round((t_end - t0)//h))
    x_prev = x0
    t = t0
    sol = Result(x_prev, t)

    if implicit:
        for i in range(n_steps + 1):
            x_new, t = implicit_step(f, x_prev, t, h)
            x_prev = x_new
            sol.add(x_new, t)

    else:
        for i in range(n_steps+1):
            x = x_prev + h*f(t,x_prev)
            x_prev, t = x, t+h
            sol.add(x,t)
    return sol

