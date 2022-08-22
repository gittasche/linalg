import numpy as np
from linalg.svd_decomp import bidiag, svd

def test_svd():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    a = a.T
    u, b, v = bidiag(a)
    assert np.allclose(a, u @ b @ v.T)
    a = np.random.rand(100, 50)
    u, b, v = bidiag(a)
    assert np.allclose(b, u.T @ a @ v)
