import matplotlib.pyplot as plt
import numpy as np
from euler_methods import euler
from rk_explicit import rk45_explicit
from rk_implicit import radau
from solve_ode import solve_ivp

def oscilator(t,x,omega=4, ksi=0.1):
    A = np.array([[0, 1], [-omega**2, -2*ksi*omega]])
    return A@x


def test_rce(t,x):
    return x + np.sin(t)


def test_euler():
    reseni = euler(oscilator,np.array([1,1]), 0, 10, 0.01, implicit=True)
    x,t = reseni.as_arrays()

    fig, ax = plt.subplots()
    ax.plot(t,x)
    plt.show()
    print("len(reseni.t):", len(reseni.t))


def test_rk_expl():
    reseni = rk45_explicit(oscilator,np.array([1,1]),0,10, max_step=0.1, adaptive=False)
    x, t = reseni.as_arrays()
    t_array = np.linspace(0,10,1000)
    x_plot = reseni(t_array)

    fig, ax = plt.subplots()
    ax.plot(t,x)

    fig, ax = plt.subplots()
    ax.plot(t_array, x_plot)
    plt.show()
    print("len(reseni.t):", len(reseni.t))


def test_rk_implicit():
    reseni = radau(oscilator,np.array([1,1]),0,10, h = 0.1)
    x, t = reseni.as_arrays()

    fig, ax = plt.subplots()
    ax.plot(t, x)

    t_array = np.linspace(0, 10, 1000)
    x_plot = reseni(t_array)

    fig, ax = plt.subplots()
    ax.plot(t_array, x_plot)
    plt.show()
    print("len(reseni.t):", len(reseni.t))


def test_solve_ode():
    reseni = solve_ivp(test_rce, 0, 0, 10, method="RK45", graph=True)

#test_solve_ode()

test_euler()
test_rk_expl()
test_rk_implicit()
