import matplotlib.pyplot as plt
import numpy as np
from euler_methods import euler

def oscilator(t,x,omega=4, ksi=2):
    A = np.array([[0, 1], [-omega**2, -2*ksi*omega]])
    return A@x

def test_rce(t,x):
    return -15*x + np.sin(t)

reseni = euler(test_rce,0, 0, 10, 0.01, implicit=True)
x,t = reseni.as_arrays()

fig, ax = plt.subplots()
ax.plot(x)
plt.show()


