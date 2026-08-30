import numpy as np


def jacobi(f,x,n):
    """
    Aproximuje Jacobiho matici funkce f v bodě x centrálními diferencemi (lepší přesnost než jen na jednu stranu).
    :param f: callable, funkce Rn -> Rn
    :param x: bod, ve kterém se Jacobiho matice počítá
    :param n: dimenze Rn
    :return: aproximovaná Jacobiho matice, tvar (n, n)
    """
    J = np.zeros((n, n))
    h = 1e-8
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        J[:, i] = (f(x + ei) - f(x - ei)) / (2 * h)
    return J


def regularization(J, fn, n, reg0 = 1e-10, max_tries = 10):
    """
    Zkusí vyřešit soustavu J@delta = -fn pro regularizovanou matici (J + reg*I),
    pokud je J singulární. Regularizační faktor se při neúspěchu vždy desetkrát zvětší.
    :param J: (singulární) Jacobiho matice
    :param fn: hodnota f(x), pravá strana soustavy
    :param n: rozměr soustavy
    :param reg0: počáteční regularizační faktor
    :param max_tries: maximální počet pokusů se zvětšujícím se reg
    :return: řešení delta regularizované soustavy
    :raises numpy.linalg.LinAlgError: pokud soustava zůstane singulární i po max_tries pokusech
    """
    reg = reg0
    for _ in range(max_tries):
        try:
            return np.linalg.solve(J + reg*np.eye(n), -fn)
        except np.linalg.LinAlgError:
            reg *= 10
    raise np.linalg.LinAlgError("Stále po regularizaci neřešitelná")


def damping(f, x, delta, norm_old, max_tries = 20):
    """
    Zkracuje krok (line search s půlením), dokud se nezlepší reziduum. Nemusí pomoct.
    :param f: callable, funkce soustavy f(x) = 0
    :param x: současný bod
    :param delta: navržený Newtonův krok
    :param norm_old: norma rezidua v současném bodě, se kterou se porovnává zlepšení
    :param max_tries: maximální počet půlení kroku
    :return: (nový bod x, f(nový bod) nebo None při neúspěchu, bool jestli se podařilo zlepšit)
    """
    alpha = 1
    for _ in range(max_tries):
        x_new = x + alpha * delta
        fn_new = f(x_new)
        norm_new = np.max(np.abs(fn_new))
        if norm_new < norm_old:
            return x_new, fn_new, True
        alpha *= 0.5

    return x, None, False


def newton(f, x0, jac = None, maxiter = 200, tol = 1e-9, reg = 1e-10):
    """
    Řeší soustavu f(x) = 0
    :param f: funkce callable
    :param x0: počáteční odhad
    :param jac: pokud ho máme spočtěný na papíře, formát jako matice
    :param maxiter: maximální počet itercí, default 200
    :param tol: tolerance na splnění rovnosti
    :param reg: počáteční regularizační faktor
    :return: Zpráva o konvergenci, počet potřebných iterací, kořen
    """
    if isinstance(x0,(int,float)):              #testuje že to je číslo, aby fungoval skalární případ
        n = 1
        x0 = np.array([x0])
    else:
        n = len(x0)
    x = [x0]
    fn = f(x[-1])

    for i in range(maxiter):
        #test jestli už mám požadovanou přesnost
        norm_fn = np.max(np.abs(fn))
        if norm_fn < tol:
            convergence = True
            return convergence, i, np.array(x)[-1]
        #výpočet jacobiánu
        if jac is None:
            J = jacobi(f, x[-1], n)
        else:
            J = jac(x[-1])
        #může nastat chyba se singulární maticí
        try:
            delta = np.linalg.solve(J, -fn)
        except np.linalg.LinAlgError:
            #zkusíme ji zregularizovat
            try:
                delta = regularization(J,fn,n,reg)
            except np.linalg.LinAlgError:
                return False, i, np.array(x)[-1]

        x_new, fn_new, improved = damping(f, x[-1], delta, norm_fn)
        if not improved:
            return False, i, x[-1]
        #jen test pro možnost bez dampingu
        #x_new = x[-1] + delta
        #fn_new = f(x_new)

        x.append(x_new)
        fn = fn_new

    return False, maxiter, np.array(x)[-1]


def f(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]])

def g(x):
    #funkce pro kterou by měl být využit damping
    #bez dampingu diverguje
    return np.arctan(x)

def test():
    x0 = 3
    info, iter, x = newton(g, x0)
    print(info, iter, x)

if __name__ == "__main__":
    test()