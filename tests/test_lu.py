import numpy as np
import pytest

from linalg.lu import lu_solve, lu
from precision_cfg import assert_allclose_custom

@pytest.mark.parametrize("a",
    [
        np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 10]
        ]),
        np.array([
            [2, -2, -3, -4],
            [4, -1, -4, -5],
            [6, 0, -9, -4],
            [8, 1, -14, -2]
        ])
    ]
)
@pytest.mark.parametrize("piv_option", ["row", "col", "full"])
def test_lu(a, piv_option):
    l, u, p, q = lu(a, piv_option=piv_option)
    assert_allclose_custom(a, p @ l @ u @ q)

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
def test_lu_solve(a, b, piv_option):
    x = lu_solve(a, b, piv_option=piv_option)
    assert_allclose_custom(b, a @ x)