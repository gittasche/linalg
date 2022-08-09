import numpy as np

from linalg.lu.lu_band import lu_band, lu_band_solve
from linalg.sym_decomp.ldlt_band import ldlt_band, sym_band_solve
from linalg.sympos_decomp.cholesky_band import cholesky_band, sympos_band_solve

def test_banded():
    # test lu
    a = np.array([
        [5, 2, -1, 0, 0, 0],
        [2, 4, 2, -2, 0, 0],
        [0, 3, 3, 2, -3, 0],
        [0, 0, 1, 2, 2, -3],
        [0, 0, 0, 4, 1, 2],
        [0, 0, 0, 0, 2, -1]
    ])
    l, u, p = lu_band(a, 1, 2)
    assert np.allclose(a, p @ l @ u)

    # test solve
    b = np.array([0, 1, 2, 2, 3, 3])
    x = lu_band_solve(a, 1, 2, b)
    assert np.allclose(b, a @ x)

    # test cholesky
    a = np.array([
        [4, 2, -1, 0, 0, 0],
        [2, 5, 2, -1, 0, 0],
        [-1, 2, 6, 2, -1, 0],
        [0, -1, 2, 7, 2, -1],
        [0, 0, -1, 2, 8, 2],
        [0, 0, 0, -1, 2, 9]
    ])

    l = cholesky_band(a, d=2)
    assert np.allclose(a, l @ l.T)

    # test solve
    b = np.array([1, 2, 2, 3, 3, 3])
    x = sympos_band_solve(a, 2, b)
    assert np.allclose(b, a @ x)

    # test ldlt
    l, d = ldlt_band(a, d=2)
    assert np.allclose(a, l @ np.diag(d) @ l.T)

    x = sym_band_solve(a, 2, b)
    assert np.allclose(b, a @ x)

if __name__ == "__main__":
    test_banded()