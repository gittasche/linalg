from numpy.testing import assert_allclose

ATOL = 1e-12
RTOL = 1e-5

def assert_allclose_custom(x, y):
    return assert_allclose(x, y, atol=ATOL, rtol=RTOL)