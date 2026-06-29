import numpy as np


def search_num(array, x):
    """na principu binar search"""
    start, end = 0, len(array)
    while start < end:
        mid = (start + end) // 2
        if array[mid] < x:
            start = mid + 1
        else:
            end = mid
    return start


def search(array, t_eval):
    """hledá interval kam patří jaký prvek"""
    indices = np.zeros(len(t_eval), dtype=int)
    for i,a in enumerate(t_eval):
        index_a = search_num(array, a)
        indices[i] = index_a
    return indices - 1


def clip(a, start, end):
    return np.array([max(start, min(x, end)) for x in a])


class Result:
    """
    Třída pro výsledky, pro lepší manipulaci a pak vykreslování.

    Uchovává posloupnost (t, x, dx) z numerické integrace a umožňuje
    dense output přes Hermitovu interpolaci voláním Result(t_eval).

    :ivar x: list hodnot stavu v jednotlivých uzlech
    :ivar t: list časů uzlů
    :ivar dx: list derivací f(t,x) v jednotlivých uzlech
    """
    def __init__(self, x = None, t = None, dx = None):
        if x is None:
            self.x = []
        else:
            self.x = [x]
        if t is None:
            self.t = []
        else:
            self.t = [t]
        if dx is None:
            self.dx = []
        else:
            self.dx = [dx]

    def add(self, x_new, t_new, dx):
        self.x.append(x_new)
        self.t.append(t_new)
        self.dx.append(dx)

    def as_arrays(self):
        """
        Převede Result na array
        :return: vrací x, t
        """
        return np.array(self.x), np.array(self.t)

    def __call__(self, t_eval):
        """
        dense output pomocí hermitovy interpolace mezi uzly
        """
        eps = 1e-8              #nutný kvůli floating point arithmetics
        t0, t_end = self.t[0], self.t[-1]
        t_eval = np.array([x for x in t_eval if t0 <= x <= t_end + eps])          #oseknutí extrapolace

        t_arr = np.asarray(self.t)
        x_arr = np.asarray(self.x)  # shape (n_kroku, n_promennych)
        dx_arr = np.asarray(self.dx)
        #najít index intervalu, indexy jsou pole, pro numpy to lze dělat vektorově
        i = search(t_arr, t_eval)
        i = clip(i, 0, len(t_arr)-2)
        #normování
        h = t_arr[i+1] - t_arr[i]
        s = (t_eval - t_arr[i]) / h
        #sestrojení bázového polynomu
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        value = (h00[:, None]*x_arr[i] + h10[:, None]*h[:, None]*dx_arr[i]
                + h01[:, None]*x_arr[i+1] + h11[:, None]*h[:, None]*dx_arr[i+1])
        scalar = np.isscalar(t_eval) or np.ndim(t_eval) == 0
        if scalar:
            return value[0]
        return value


