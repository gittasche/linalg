import numpy as np
from numpy.testing import assert_allclose

from linalg.eig_unsym import house_hess

def test_eigunsym():
    a = np.array([
        [1, 5, 7],
        [3, 0, 6],
        [4, 3, 1]
    ])
    u, h = house_hess(a)
    assert_allclose(a, u @ h @ u.T, atol=1e-12)
    a = np.random.rand(50, 50)
    u, h = house_hess(a)
    assert_allclose(a, u @ h @ u.T, atol=1e-12)
