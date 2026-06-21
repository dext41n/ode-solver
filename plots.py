import matplotlib.pyplot as plt
import numpy as np
from euler_methods import explicit_euler

def oscilator(t,x,omega=4, ksi=2):
    A = np.array([[0, 1], [-omega**2, -2*ksi*omega]])
    return A@x

reseni = explicit_euler(oscilator,np.array([1,1]), 0, 10, 0.01)
x,t = reseni.as_arrays()

fig, ax = plt.subplots()
ax.plot(x)
plt.show()


