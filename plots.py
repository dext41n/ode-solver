import matplotlib as plt
import numpy as np

def oscilator(t,x,omega=4, ksi=2):
    A = np.array([[0, 1], [-omega**2, -2*ksi*omega]])
    return A@x


