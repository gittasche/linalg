import numpy as np

from linalg.sympos_decomp.cholesky import cholesky, sympos_solve
from precision_cfg import assert_allclose_custom

def test_cholesky():
    # check decomposition
    a = np.array([
        [4, -1, 1],
        [-1, 4.25, 2.75],
        [1, 2.75, 3.5]
    ])
    l = cholesky(a)
    assert_allclose_custom(a, l @ l.T)

    # test solve
    a = np.array([
        [4, 1, -1, 0],
        [1, 3, -1, 0],
        [-1, -1, 5, 2],
        [0, 0, 2, 4]
    ])
    b = np.array([7, 8, -4, 6])
    x = sympos_solve(a, b)
    assert_allclose_custom(b, a @ x)
