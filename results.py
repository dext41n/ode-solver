import numpy as np


def search_num(array, x):
    """
    Najde index prvního prvku v seřazeném poli, který je >= x (na principu binary search).
    :param array: seřazené 1D pole
    :param x: hledaná hodnota
    :return: index prvního prvku >= x
    """
    start, end = 0, len(array)
    while start < end:
        mid = (start + end) // 2
        if array[mid] < x:
            start = mid + 1
        else:
            end = mid
    return start


def search(array, t_eval):
    """
    Pro každý bod v t_eval najde index intervalu (uzlu vlevo), kam bod patří.
    :param array: seřazené pole uzlů (rostoucí časy)
    :param t_eval: pole časů, pro které hledáme příslušný interval
    :return: pole indexů levých krajních bodů intervalů, stejné délky jako t_eval
    """
    indices = np.zeros(len(t_eval), dtype=int)
    for i,a in enumerate(t_eval):
        index_a = search_num(array, a)
        indices[i] = index_a
    return indices - 1


def clip(a, start, end):
    """
    Ořízne hodnoty pole do intervalu [start, end].
    :param a: pole hodnot
    :param start: dolní mez
    :param end: horní mez
    :return: nové pole se všemi hodnotami oříznutými do [start, end]
    """
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
        """
        Přidá jeden krok řešení, trojici (x, t, dx)
        :param x_new: nové x
        :param t_new: nové t
        :param dx: nová derivace
        """
        self.x.append(x_new)
        self.t.append(t_new)
        self.dx.append(dx)

    def as_arrays(self):
        """
        Převede uložené seznamy uzlů na numpy pole.
        :return: dvojice (x, t) jako numpy pole
        """
        return np.array(self.x), np.array(self.t)

    def __call__(self, t_eval):
        """
        Dense output pomocí Hermitovy interpolace mezi uzly.

        Funguje jak pro jeden skalární čas, tak pro pole časů. Body mimo
        [t0, t_end] jsou oříznuty (žádná extrapolace) -- pokud
        nějaký bod z t_eval padne mimo interval integrace, vrácené pole
        bude kratší než vstupní t_eval.
        :param t_eval: skalár nebo pole časů, ve kterých chceme hodnotu řešení
        :return: interpolovaná hodnota x(t_eval); skalár/1D pole podle vstupu
        """

        eps = 1e-8                                                      #kvůli fpa
        t0, t_end = self.t[0], self.t[-1]
        scalar = np.isscalar(t_eval) or np.ndim(t_eval) == 0                #ověření skaláru
        t_eval = np.atleast_1d(t_eval)
        t_eval = t_eval[(t_eval >= t0) & (t_eval <= t_end + eps)]        #oseknutí extrapolace

        if t_eval.size == 0:
            if scalar:
                raise ValueError(
                    f"Zadaný čas je mimo interval integrace [{t0}, {t_end}]"
                )
            return np.array([])

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
        #samotná interpolace dle vzorce
        value = (h00[:, None]*x_arr[i] + h10[:, None]*h[:, None]*dx_arr[i]
                + h01[:, None]*x_arr[i+1] + h11[:, None]*h[:, None]*dx_arr[i+1])
        if scalar:
            return value[0]
        return value


