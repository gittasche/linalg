import numpy as np

from linalg.eig_unsym import house_hess
from precision_cfg import assert_allclose_custom

def test_eigunsym():
    a = np.array([
        [1, 5, 7],
        [3, 0, 6],
        [4, 3, 1]
    ])
    u, h = house_hess(a)
    assert_allclose_custom(a, u @ h @ u.T)
    a = np.random.rand(50, 50)
    u, h = house_hess(a)
    assert_allclose_custom(a, u @ h @ u.T)
