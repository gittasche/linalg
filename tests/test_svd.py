import numpy as np
from numpy.testing import assert_allclose

from linalg.svd_decomp import bidiag

def test_svd():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    a = a.T
    u, b, v = bidiag(a)
    assert_allclose(a, u @ b @ v.T, atol=1e-12)
    a = np.random.rand(100, 50)
    u, b, v = bidiag(a)
    assert_allclose(b, u.T @ a @ v, atol=1e-12)
