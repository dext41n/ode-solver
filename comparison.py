import numpy as np
import matplotlib.pyplot as plt
from solve_ode import solve_ivp

def oscilator_v1(t, x, omega = 5):
    return np.array([x[1],-omega*x[0]])

def oscilator_v2(t,x,omega=4, ksi=0.1):
    A = np.array([[0, 1], [-omega**2, -2*ksi*omega]])
    return A@x

def f(t, x, k=1):
    return k*x


def stiff(t,x):
    return -15*x


def count_error(exact,x):
    return np.abs(exact-x)

def error_vs_step():
    """Na známé rovnici vyštříme konvergenci závislou na počtu kroků"""

    def exact_sol(t, k=1):
        return np.exp(t * k)

    cases = [
        ("Euler", 1),
        ("Euler", 0.1),
        ("Euler", 0.01),
        ("Euler", 0.001),
        ("RK45", None),
    ]

    reseni = []
    for method, h in cases:
        if h is None:
            sol = solve_ivp(f, 1, 0, 5, method=method, adaptive=False, max_step=1)
        else:
            sol = solve_ivp(f, 1, 0, 5, method=method, h=h, max_step=1)
        reseni.append(sol)

    fig, ax = plt.subplots()
    for (method, h), sol in zip(cases, reseni):
        x, t = sol.as_arrays()
        label = method +" h=1" if h is None else f"{method} h={h}"
        ax.plot(t, x, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Závislost na kroku")
    ax.legend()
    plt.show()

    t_eval = np.linspace(0, 5, 6)
    exact = exact_sol(t_eval)

    fig, ax = plt.subplots()
    for (method, h), sol in zip(cases, reseni):
        label = method + " h=1" if h is None else f"{method} h={h}"
        err = np.abs(exact - sol(t_eval).ravel())
        ax.plot(t_eval, err, label=label)
    ax.set_title("Chyba v logaritmickém měřítku")
    ax.set_yscale("log")
    ax.set_ylabel("e")
    ax.set_xlabel("t")
    ax.legend()
    plt.show()



def energy():
    """mělo by se ukázat, že exp euler zvyšuje energii, imp euler snižuje, RK45 a Radau zachovává"""
    x0 = np.array([0,1])

    solulu_euler = solve_ivp(oscilator_v1, x0, 0, 10, method="Euler", h=0.01, graph=False)
    solulu_rk45 = solve_ivp(oscilator_v1, x0, 0, 10, method="RK45", max_step=0.01, graph=False)
    solulu_radau = solve_ivp(oscilator_v1, x0, 0, 10, method="Radau", h = 0.01, graph=False)
    solulu_ie = solve_ivp(oscilator_v1, x0, 0, 10, method="ImplicitEuler", h = 0.01, graph=False)

    x_euler, t_euler = solulu_euler.as_arrays()
    x_rk45, t_rk45 = solulu_rk45.as_arrays()
    x_radau, t_radau = solulu_radau.as_arrays()
    x_ie, t_ie = solulu_ie.as_arrays()

    fig, ax = plt.subplots(figsize = (9,5))
    ax.plot(t_euler, x_euler[:,0], label = "exp euler")
    ax.plot(t_rk45,x_rk45[:,0], label = "RK45")
    ax.plot(t_radau,x_radau[:,0], label = "Radau")
    ax.plot(t_ie,x_ie[:,0], label = "imp euler")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Preservation of energy")
    ax.legend()
    plt.show()


def stiff_problem():
    """mělo by se ukázat, že implicitní metody jsou lepší na stiff problémy"""
    def exact(t):
        return np.exp(-15*t)

    exp_sol = solve_ivp(stiff,1, 0,1, method="Euler", h = 0.1)
    imp_sol = solve_ivp(stiff, 1, 0,1, method="ImplicitEuler", h=0.1)
    radau_sol = solve_ivp(stiff, 1,0,1, method="Radau", h=0.1)
    x1, t1 = exp_sol.as_arrays()
    x2, t2 = imp_sol.as_arrays()
    x3, t3 = radau_sol.as_arrays()
    t_eval = np.linspace(0,1,11)
    exact_x = exact(t_eval)
    fig, ax = plt.subplots()
    ax.plot(t1, x1, label = "exp euler")
    ax.plot(t2, x2, label = "imp euler")
    ax.plot(t3, x3, label = "radau")
    ax.plot(t_eval, exact_x, label = "exact solulu")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()
    plt.show()


def A_stability():
    """Při moc velkém kroku Euler úplně exploduje, řešení je naprosto špatně"""
    exp_sol = solve_ivp(oscilator_v2, np.array([1,1]), 0, 10,method= "Euler", h =0.1)
    imp_sol = solve_ivp(oscilator_v2, np.array([1,1]), 0, 10,method= "ImplicitEuler", h =0.1)
    x1, t1 = exp_sol.as_arrays()
    x2, t2 = imp_sol.as_arrays()
    fig, ax = plt.subplots()
    ax.plot(t1, x1, label="exp euler")
    ax.plot(t2, x2, label="imp euler")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()
    plt.show()


error_vs_step()
energy()
stiff_problem()
A_stability()