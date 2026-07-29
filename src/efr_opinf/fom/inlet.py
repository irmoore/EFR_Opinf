import numpy as np
import petsc4py.PETSc as PETSc

gdim = 2


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        if self.t <= 4:
            values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        else:
            values[0] = 4 * 1.5 * np.sin(4 * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        return values
