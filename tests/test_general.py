import numpy as np

from linalg import solve
from linalg import solve_triangle
from linalg import det
from linalg import inv
from linalg import solve_band, solves_band
from linalg import qr

from precision_cfg import assert_allclose_custom

def test_general():
    # test solve
    a = np.array([
        [4, 1, -1, 0],
        [1, 3, -1, 0],
        [-1, -1, 5, 2],
        [0, 0, 2, 4]
    ])
    b = np.array([7, 8, -4, 6])
    x = solve(a, b, assume_a="pos")
    assert_allclose_custom(b, a @ x)

    # test det
    a = np.array([
        [4, 5, 2, 3],
        [3, 2, 4, 6],
        [1, 6, 5, 4],
        [2, 4, 6, 5]
    ])
    d = det(a)
    assert_allclose_custom(d, 141.0)

    # test inv
    a = np.array([
        [14, 2, 0, 1],
        [1, 2, 1, 0],
        [-2, 3, 5, 2],
        [0, 1, 2, 1]
    ])
    a_inv = inv(a)
    identity = np.identity(a.shape[0])
    assert_allclose_custom(identity, a @ a_inv)

    # test band
    a = np.array([
        [5, 2, -1, 0, 0, 0],
        [2, 4, 2, -2, 0, 0],
        [0, 3, 3, 2, -3, 0],
        [0, 0, 1, 2, 2, -3],
        [0, 0, 0, 4, 1, 2],
        [0, 0, 0, 0, 2, -1]
    ])
    b = np.array([0, 1, 2, 2, 3, 3])
    x = solve_band(a, 1, 2, b)
    assert_allclose_custom(b, a @ x)

    a = np.array([
        [4, 2, -1, 0, 0, 0],
        [2, 5, 2, -1, 0, 0],
        [-1, 2, 6, 2, -1, 0],
        [0, -1, 2, 7, 2, -1],
        [0, 0, -1, 2, 8, 2],
        [0, 0, 0, -1, 2, 9]
    ])
    b = np.array([1, 2, 2, 3, 3, 3])
    x = solves_band(a, 2, b, ensure_pos=True)
    assert_allclose_custom(b, a @ x)
    x = solves_band(a, 2, b, ensure_pos=False)
    assert_allclose_custom(b, a @ x)
    
    a = np.array([
        [1, 1, 2, 2],
        [0, 2, 4, 3],
        [0, 0, 1, 7],
        [0, 0, 0, 6]
    ], dtype="float64")
    b = np.arange(4, dtype="float64")

    # test upper triangle
    x = solve_triangle(a, b)
    assert_allclose_custom(b, a @ x)
    x = solve_triangle(a, b, transposed=True)
    assert_allclose_custom(b, a.T @ x)
    
    # test lower triangle
    a = a.T
    x = solve_triangle(a, b, lower=True)
    assert_allclose_custom(b, a @ x)
    x = solve_triangle(a, b, lower=True, transposed=True)
    assert_allclose_custom(b, a.T @ x)

    # test qr
    a = np.array([
        [1, 2, 3],
        [1, 5, 6],
        [1, 8, 9],
        [1, 11, 12]
    ])
    q, r, p = qr(a, mode="full", pivoting=True, decode_p=True)
    assert_allclose_custom(a, q @ r @ p)
