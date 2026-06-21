import numpy as np

class Result:

    def __init__(self, x = None, t = None):
        self.x = [x]
        self.t = [t]

    def add(self, x_new, t_new):
        if self.x is None:
            self.x = [x_new]
        else:
            self.x.append(x_new)
        if self.t is None:
            self.t = [t_new]
        else:
            self.t.append(t_new)

    def as_arrays(self):
        return np.array(self.x), np.array(self.t)


def explicit_euler(f, x0, t0, t_end, h):

    n_steps = int(round((t_end - t0)//h))
    x_prev = x0
    t_prev = t0
    sol = Result(x_prev, t_prev)

    for i in range(n_steps+1):
        x = x_prev + h*f(t_prev,x_prev)
        x_prev, t_prev = x, t_prev+h
        sol.add(x,t_prev)
    return sol
