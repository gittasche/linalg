import numpy as np

from linalg.transforms import house, givens

def test_transforms():
    # test householder
    x = np.array([3, 1, 5, 1])
    beta, v = house(x, 2)
    p = np.identity(4) - beta * np.outer(v, v)
    check = np.zeros(4)
    check[2] = np.linalg.norm(x)
    assert np.allclose(check, p @ x)

    # test givens
    x = np.array([1, 2, 3, 4])
    c, s = givens(x[1], x[3])
    g = np.array([
        [c, -s],
        [s, c]
    ])
    assert np.allclose(0.0, (g @ x[[1, 3]])[1])
