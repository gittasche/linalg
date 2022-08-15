import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from cython.parallel import prange
np.import_array()

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef struct Pair:
    double c
    double s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef Pair cy_givens(double a, double b):
    cdef double tau
    cdef Pair giv

    if b == 0.0:
        giv.c = 1.0
        giv.s = 0.0
    else:
        if fabs(b) > fabs(a):
            tau = -<double>a / b
            giv.s = 1.0 / sqrt(1.0 + tau**2)
            giv.c = giv.s * tau
        else:
            tau = -<double>b / a
            giv.c = 1.0 / sqrt(1.0 + tau**2)
            giv.s = giv.c * tau

    return giv