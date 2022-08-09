import numpy as np

from linalg.sympos_decomp.cholesky import cholesky, sympos_solve

def test_cholesky():
    # check decomposition
    a = np.array([
        [4, -1, 1],
        [-1, 4.25, 2.75],
        [1, 2.75, 3.5]
    ])
    l = cholesky(np.copy(a))
    assert np.allclose(a, l @ l.T)

    # test solve
    a = np.array([
        [4, 1, -1, 0],
        [1, 3, -1, 0],
        [-1, -1, 5, 2],
        [0, 0, 2, 4]
    ])
    b = np.array([7, 8, -4, 6])
    x = sympos_solve(a, b)
    assert np.allclose(b, a @ x)

if __name__ == "__main__":
    test_cholesky()