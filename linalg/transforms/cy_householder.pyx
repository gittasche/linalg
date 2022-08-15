import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
np.import_array()

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef double cy_house(double[::1] x, int i):
    cdef Py_ssize_t k
    cdef double x_i, sigma, beta

    x_i = x[i]
    for k in prange(x.shape[0], nogil=True):
        if k != i:
            sigma += x[k]**2
    x[i] = 1.0

    if sigma == 0.0:
        beta = 0.0
    else:
        nu = sqrt(x_i**2 + sigma)
        if x_i < 0.0:
            x[i] = x_i - nu
        else:
            x[i] = -sigma / (x_i + nu)
        beta = 2 * x[i]**2 / (sigma + x[i]**2)
        x_i = x[i]
        for k in prange(x.shape[0], nogil=True):
            x[k] = x[k] / x_i
    return beta
