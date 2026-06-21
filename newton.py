import numpy as np


def jacobi(f,x,n):
    """aproximuje jacobian konečnýma diferencema"""
    J = np.zeros((n, n))
    h = 1e-8
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        J[:, i] = (f(x + ei) - f(x - ei)) / (2 * h)
    return J


def newton(f, x0, jac = None, maxiter = 200, tol = 1e-9):
    """
    Řeší soustavu f(x) = 0
    :param f: funkce callable
    :param x0: počáteční odhad
    :param jac: pokud ho máme spočtěný na papíře, formát jako matice
    :param tol: tolerance na splnění rovnosti
    :return: Zpráva o konvergenci, počet potřebných iterací, kořen
    """
    if isinstance(x0,(int,float)):
        n = 1
        x0 = np.array([x0])
    else:
        n = len(x0)
    x = [x0]

    for i in range(maxiter):
        fn = f(x[-1])

        if np.max(np.abs(fn)) < tol:
            convergence = True
            return convergence, i, np.array(x)[-1]
        if jac is None:
            J = jacobi(f, x[-1], n)
        else:
            J = jac(x[-1])
        delta = np.linalg.solve(J, -fn)
        x.append(x[-1] + delta)

    convergence = False
    num_iter = 100
    return convergence, num_iter, np.array(x)[-1]


def f(x):
    return np.array([x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]])

def g(x):
    return np.sin(x)

def test():
    x0 = [1,1]
    info, iter, x = newton(f, x0)
    print(info, iter, x)

if __name__ == "__main__":
    test()