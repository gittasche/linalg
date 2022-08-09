import numpy as np

from linalg.lu import lu_solve, lu

def test_lu():
    # test LU
    a = np.array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 10]
    ])
    l, u, p, _ = lu(a, piv_option="row")
    assert np.allclose(a, p @ l @ u)

    a = np.array([
        [2, -2, -3, -4],
        [4, -1, -4, -5],
        [6, 0, -9, -4],
        [8, 1, -14, -2]
    ])
    l, u, p, q = lu(a, piv_option="full")
    assert np.allclose(a, p @ l @ u @ q)

    # test system
    a = np.array([
        [2, -2, -3, -4],
        [4, -1, -4, -5],
        [6, 0, -9, -4],
        [8, 1, -14, -2]
    ])
    b = np.array([
        [1, 4],
        [2, 3],
        [3, 12],
        [4, 6]
    ])

    x = lu_solve(a, b, piv_option="full")
    assert np.allclose(b, a @ x)

    a = np.array([
        [3.3330, 15920, 10.333],
        [2.2220, 16.710, 9.6120],
        [-1.5611, 5.1792, -1.6855]
    ])
    b = np.array([7953, 0.965, 2.714])
    x = lu_solve(a, b)
    assert np.allclose(b, a @ x)

if __name__ == "__main__":
    test_lu()