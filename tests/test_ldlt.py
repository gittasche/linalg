import numpy as np

from linalg.sym_decomp.ldlt import ldlt, sym_solve

def test_ldlt():
    # test well-conditioned
    a = np.array([
        [10, 20, 30],
        [20, 45, 80],
        [30, 80, 171]
    ])
    l, d, p = ldlt(a, piv_option="sym")
    assert np.allclose(a, p @ l @ np.diag(d) @ l.T @ p.T)

    # test poor-conditioned
    # None pivoting got divide by zero
    a = np.array([
        [10, 0, 30],
        [0, 0, 80],
        [30, 80, 0]
    ])
    l, d, p = ldlt(a, piv_option="sym")
    assert np.allclose(a, p @ l @ np.diag(d) @ l.T @ p.T)

    # test symmetric solver
    a = np.array([
        [4, 1, -1, 0],
        [1, 3, -1, 0],
        [-1, -1, 5, 2],
        [0, 0, 2, 4]
    ])
    b = np.array([7, 8, -4, 6])
    x = sym_solve(a, b, piv_option="sym")
    assert np.allclose(b, a @ x)

    a = np.array([
        [3.3330, 15920, 10.333],
        [2.2220, 16.710, 9.6120],
        [-1.5611, 5.1792, -1.6855]
    ])
    a = np.dot(a, a.T) # get symmetric
    b = np.array([7953, 0.965, 2.714])
    x = sym_solve(a, b)
    assert np.allclose(b, a @ x)

if __name__ == "__main__":
    test_ldlt()