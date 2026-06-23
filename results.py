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