import numpy as np
from newton import newton
from results import Result


def implicit_step(f, x_prev, t, h):
    """
    Udělá jeden krok implicitního (zpětného) Eulera: řeší x_new = x_prev + h*f(t+h, x_new)
    Newtonovou metodou. Pokud Newton nezkonverguje, zkusí to s poloviční délkou kroku.
    :param f: funkce pravé strany, callable f(t, x)
    :param x_prev: hodnota x na začátku kroku
    :param t: čas na začátku kroku
    :param h: požadovaná délka kroku
    :return: (nová hodnota x, nový čas t + použitý krok)
    :raises RuntimeError: pokud Newton nezkonverguje ani po opakovaném půlení kroku
    """
    h_try = h
    while h_try > 1e-9:
        t_new = t + h_try

        def g(x, h_try=h_try, t_new=t_new):
            return x_prev + h_try * f(t_new, x) - x
        #x0 pro newtona odhadnu klasicky pomocí explicitního eulera
        guess = x_prev + h_try * f(t_new, x_prev)
        convergence, iters, x_new = newton(g, guess)

        if convergence:
            return x_new, t_new
        h_try /= 2

    raise RuntimeError(f"Newton nezkonvergoval ani při h={h_try}")


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
    if isinstance(x0,(int,float)):
        x_prev = np.array([x0])
    else:
        x_prev = x0
    n_steps = int(round((t_end - t0)/h))
    t = t0
    sol = Result(x_prev, t, f(t,x_prev))

    if implicit:
        time_eps = 1e-8 #nějaký práh relativní k typickému h
        while t < t_end:
            h_step = min(h, t_end - t)
            if h_step < time_eps:
                # poslední krok je tak malý, že nemá smysl řešit nelineárně
                x_new = x_prev + h_step * f(t, x_prev)
                t = t + h_step
            else:
                x_new, t = implicit_step(f, x_prev, t, h_step)
            x_prev = x_new
            sol.add(x_new, t, f(t,x_new))

    else:
        for i in range(n_steps):
            x = x_prev + h*f(t,x_prev)
            x_prev, t = x, t+h
            sol.add(x,t,f(t,x))
    return sol

