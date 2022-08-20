import numpy as np

from linalg.eig_unsym import house_hess, schur

def test_eigunsym():
    a = np.array([
        [1, 5, 7],
        [3, 0, 6],
        [4, 3, 1]
    ])
    u, h = house_hess(a)
    assert np.allclose(a, u @ h @ u.T)
    a = np.random.rand(50, 50)
    u, h = house_hess(a)
    assert np.allclose(a, u @ h @ u.T)

if __name__ == "__main__":
    test_eigunsym()