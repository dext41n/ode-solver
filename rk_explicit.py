import numpy as np
from results import Result


def rk45_params():
    """jen pro zisk koeficientu, poradil pán dormand a prince"""
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0,           0,            0,           0,          0,            0,     0],
        [1/5,         0,            0,           0,          0,            0,     0],
        [3/40,        9/40,         0,           0,          0,            0,     0],
        [44/45,      -56/15,        32/9,        0,          0,            0,     0],
        [19372/6561, -25360/2187,   64448/6561, -212/729,    0,            0,     0],
        [9017/3168,  -355/33,       46732/5247,  49/176,    -5103/18656,   0,     0],
        [35/384,      0,            500/1113,    125/192,   -2187/6784,    11/84, 0],
    ])
    b_5 = np.array([35/384,     0, 500/1113,  125/192,  -2187/6784,    11/84,    0])
    b_4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    return A, c, b_4, b_5


def count_coeficients(f, x, t, h, c, A, k1):
    """spočítá koeficienty k2 až k7 pro jeden krok rk45, k1 = k7 předešlý
    k je matice co má v řádcích prvky k1 až k7
    """
    n = len(x)
    k_num = len(c)
    k = np.zeros((k_num,n))
    k[0] = k1

    for i in range(1,k_num):
        #elegantně přes matice
        xi = x + h*A[i,:i]@k[:i]
        k[i] = f(t + h*c[i], xi)
    return k


def rk45_step(f, x, t, h, k1, c, A, b_4, b_5, adaptive):
    """
    c je levý sloupec koeficentů z butcherovy tabulky, A je hlavní část tabulky a b_4 resp. b_5
    jsou vektory pro finální lin, kombinaci. k1 je první koeficient
    """
    k = count_coeficients(f,x,t,h,c,A,k1)
    x_new = x + h*(b_5@k)
    k7 = k[-1]
    if adaptive:
        x_hat = x + h*(b_4@k)
        error = x_new - x_hat
        return x_new, error, k7
    else:
        return x_new, k7


def adaptive_step(error, atol, rtol, x, x_new, h, safety, max_step, fmin=0.1, fmax=10, p=4):
    """
    funkce počítá adaptivní krok pro rk45
    :param max_step: maximální povolený krok
    :param error: chybový vektor počítaný ve funkci step
    :param atol: absolutní tolerance
    :param rtol: relativní tolerance
    :param x: současný x ve kterým jsme
    :param x_new: nově vznikklý x minulým krokem
    :param h: minulý použitý krok
    :param safety: pojistka aby to moc nerostlo
    :param fmin: kolikrát se nejvýše může krok zmenšit
    :param fmax: kolikrát se nejvýše může krok zvětšit
    :param p: řád té méně řádové metody
    :return: nový krok, normovanou chybu vůči atol a rtol
    """
    n = len(x)
    e_sum = 0
    for i in range(n):
        e_sum += (error[i]/(atol + rtol*max(np.abs(x[i]), np.abs(x_new[i]))))**2
    e = (e_sum/n)**(1/2)
    if e == 0:
        factor = fmax
    else:
        factor = safety * e ** (-1 / (p + 1))
    factor = min(fmax, max(fmin,factor))

    h_new = h*factor
    if max_step is not None:
        if h_new > max_step:
            h_new = max_step
    return h_new, e


def first_step(f, x0, t0, atol, rtol, p=4):
    """hrubý odhad prvního kroku na základě počáteční hodnoty"""
    scale = atol + rtol*np.abs(x0)
    n = len(x0)
    d0 = 0
    d1 = 0
    for i in range(n):
        d0 += (x0[i]/scale[i])**2
        d1 += (np.atleast_1d(f(t0,x0))[i]/scale[i])**2

    d0 = (d0/n)**0.5
    d1 = (d1/n)**0.5

    if d0 < 1e-5 or d1 < 1e-5:
        return 1e-6
    return 0.01 * d0 / d1



def rk45_explicit(f, x0, t0, t_end, max_step = None,  adaptive = True, atol = 1e-6, rtol = 1e-3, min_step = 1e-12):
    """
    Řeší diferenciílní rovnici x' = f(t,x) numericky metodou Runge-Kutta 45, s adaptivním krokem.
    :param f: funkce pravé strany, klidně soustava
    :param x0: počáteční podmínka
    :param t0: počáteční čas
    :param t_end: konec intervalu času řešení
    :param max_step: maximální krok metody, není nutný vyplňovat
    :param rtol: relativní tolerance
    :param atol: absolutní tolerance
    :param adaptive: True/False jestli chceš použít adaptivní krok
    :param min_step: min. povolený step pro adaptivní krok, asi není nutný skoro nikdy měnit
    :return: objekt Result
    """
    safety = 0.8
    time_eps = 1e-9

    if isinstance(x0,(int,float)):
        x = np.array([x0])
    else:
        x = x0
    if adaptive:
        step_1 = first_step(f, x0, t0, atol, rtol, p=4)
        step = min(max_step, step_1) if max_step is not None else step_1
    else:
        if isinstance(max_step,(int,float)):
            step = max_step
        else:
            raise ValueError("Není správně, nebo vůbec zadán max_step")
    t = t0
    A, c, b_4, b_5 = rk45_params()
    k1 = f(t,x)
    sol = Result(x,t,k1)

    while t < t_end:

        if adaptive:
            while True:
                h = min(step, t_end - t)
                x_new, error, k7 = rk45_step(f, x, t, h, k1, c, A, b_4, b_5, True)
                h_new, e = adaptive_step(error, atol, rtol, x, x_new, h, safety, max_step)

                if e <= 1:
                    break  # normální přijetí, žádná zpráva
                if h < min_step:
                    print(f"Použit min. krok na t = {t}, chyba e: {e}")
                    break
                step = h_new
        else:
            h = step
            x_new, k7 = rk45_step(f, x, t, h, k1, c, A, b_4, b_5, False)

        #krok přijat
        t = t + h
        x = x_new
        k1 = k7
        sol.add(x, t, k1)
        if adaptive:
            step = h_new

    return sol
        


