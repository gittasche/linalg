import numpy as np
import pytest
from numpy.testing import assert_allclose

from linalg.sym_decomp.ldlt import ldlt, sym_solve

@pytest.mark.parametrize("a",
    [
        np.array([
            [10, 20, 30],
            [20, 45, 80],
            [30, 80, 171]
        ]),
        np.array([
            [10, 0, 30],
            [0, 0, 80],
            [30, 80, 0]
        ])
    ]
)
def test_ldlt(a):
    l, d, p = ldlt(a, piv_option="sym")
    assert_allclose(a, p @ l @ np.diag(d) @ l.T @ p.T, atol=1e-12)

@pytest.mark.parametrize("a, b",
    [
        (
            np.array([
                [4, 1, -1, 0],
                [1, 3, -1, 0],
                [-1, -1, 5, 2],
                [0, 0, 2, 4]
            ]),
            np.array([7, 8, -4, 6])
        ),
        (
            np.dot(
                np.array([
                    [3.3330, 15920, 10.333],
                    [2.2220, 16.710, 9.6120],
                    [-1.5611, 5.1792, -1.6855]
                ]),
                np.array([
                    [3.3330, 15920, 10.333],
                    [2.2220, 16.710, 9.6120],
                    [-1.5611, 5.1792, -1.6855]
                ]).T
            ),
            np.array([7953, 0.965, 2.714])
        )
    ]
)
def test_ldlt_solve(a, b):
    x = sym_solve(a, b, piv_option="sym")
    assert_allclose(b, a @ x, atol=1e-12)