import numpy as np

from linalg.elim import solve_elim
from linalg.elim import inverse
from linalg.elim import determinant

def test_gauss():
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
    x = solve_elim(a, b, piv_option="full", strategy="gaussj")
    assert np.allclose(b, a @ x)

    a = np.array([
        [3.3330, 15920, 10.333],
        [2.2220, 16.710, 9.6120],
        [-1.5611, 5.1792, -1.6855]
    ])
    b = np.array([7953, 0.965, 2.714])
    x = solve_elim(a, b)
    assert np.allclose(b, a @ x)

    # test inverse
    a = np.array([
        [14, 2, 0, 1],
        [1, 2, 1, 0],
        [-2, 3, 5, 2],
        [0, 1, 2, 1]
    ])
    a_inv = inverse(a)
    identity = np.identity(a.shape[0])
    assert np.allclose(identity, a @ a_inv)

    # test determinant
    a = np.array([
        [4, 5, 2, 3],
        [3, 2, 4, 6],
        [1, 6, 5, 4],
        [2, 4, 6, 5]
    ])
    det = determinant(a)
    assert np.allclose(det, 141.0)

if __name__ == "__main__":
    test_gauss()