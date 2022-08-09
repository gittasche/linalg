import numpy as np

from linalg.solve_tridiag import solve_tridiag, solve_sympos_tridiag

def test_tridiag():
    a_check = np.array([
        [5, 2, 0, 0, 0, 0],
        [2, 4, 2, 0, 0, 0],
        [0, 3, 3, 2, 0, 0],
        [0, 0, 1, 5, 2, 0],
        [0, 0, 0, 4, 1, 2],
        [0, 0, 0, 0, 2, -1]
    ])
    a = np.array([2, 3, 1, 4, 2])
    b = np.array([5, 4, 3, 5, 1, -1])
    c = np.array([2, 2, 2, 2, 2])
    d = np.array([0, 1, 2, 2, 3, 3])
    x = solve_tridiag((a, b, c), d)
    assert np.allclose(d, a_check @ x)

    a_check = np.array([
        [4, 2, 0, 0, 0, 0],
        [2, 5, 2, 0, 0, 0],
        [0, 2, 6, 2, 0, 0],
        [0, 0, 2, 7, 2, 0],
        [0, 0, 0, 2, 8, 2],
        [0, 0, 0, 0, 2, 9]
    ])
    d = np.array([4, 5, 6, 7, 8, 9])
    e = np.array([2, 2, 2, 2, 2])
    b = np.array([1, 2, 2, 3, 3, 3])
    x = solve_sympos_tridiag((d, e), b)
    assert np.allclose(b, a_check @ x)

if __name__ == "__main__":
    test_tridiag()