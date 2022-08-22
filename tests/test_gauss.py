import numpy as np
import pytest
from numpy.testing import assert_allclose

from linalg.elim import solve_elim
from linalg.elim import inverse
from linalg.elim import determinant

@pytest.mark.parametrize("a, b", 
    [
        (
            np.array([
                [2, -2, -3, -4],
                [4, -1, -4, -5],
                [6, 0, -9, -4],
                [8, 1, -14, -2]
            ]),
            np.array([
                [1, 4],
                [2, 3],
                [3, 12],
                [4, 6]
            ])
        ),
        (
            np.array([
                [3.3330, 15920, 10.333],
                [2.2220, 16.710, 9.6120],
                [-1.5611, 5.1792, -1.6855]
            ]),
            np.array([7953, 0.965, 2.714])
        )
    ]
)
@pytest.mark.parametrize("piv_option", ["row", "col", "full"])
@pytest.mark.parametrize("strategy", ["gauss", "gaussj"])
def test_gauss_solve(a, b, piv_option, strategy):
    x = solve_elim(a, b, piv_option=piv_option, strategy=strategy)
    assert_allclose(b, a @ x, atol=1e-12)

def test_gauss_inv():
    a = np.array([
        [14, 2, 0, 1],
        [1, 2, 1, 0],
        [-2, 3, 5, 2],
        [0, 1, 2, 1]
    ])
    a_inv = inverse(a)
    identity = np.identity(a.shape[0])
    assert_allclose(identity, a @ a_inv, atol=1e-12)

def test_gauss_det():
    a = np.array([
        [4, 5, 2, 3],
        [3, 2, 4, 6],
        [1, 6, 5, 4],
        [2, 4, 6, 5]
    ])
    det = determinant(a)
    assert_allclose(det, 141.0, atol=1e-12)
