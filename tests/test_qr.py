import numpy as np

from linalg.qr_decomp import qr_house, qr_givens, qr_gram

def test_qr():
    a = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ])
    q, r = qr_house(a, mode="full")
    assert np.allclose(a, q @ r)
    q_g, r_g = qr_givens(a, mode="full")
    assert np.allclose(a, q_g @ r_g)
    q_gr, r_gr = qr_gram(a)
    assert np.allclose(a, q_gr @ r_gr)

    rng = np.random.default_rng()
    a = rng.standard_normal((9, 6))
    q, r = qr_house(a, mode="full")
    assert np.allclose(a, q @ r)
    q_g, r_g = qr_givens(a, mode="full")
    assert np.allclose(a, q_g @ r_g)
    q_gr, r_gr = qr_gram(a)
    assert np.allclose(a, q_gr @ r_gr)
    q_e, r_e = qr_house(a, mode="economic")
    assert np.allclose(q[:, :6], q_e)
    assert np.allclose(r[:6], r_e)
    assert np.allclose(a, q_e @ r_e)
    q_g_e, r_g_e = qr_givens(a, mode="economic")
    assert np.allclose(q_g[:, :6], q_g_e)
    assert np.allclose(r_g[:6], r_g_e)
    assert np.allclose(a, q_g_e @ r_g_e)
    
    a = rng.standard_normal((6, 9))
    q, r = qr_house(a, mode="full")
    assert np.allclose(a, q @ r)
    q_g, r_g = qr_givens(a, mode="full")
    assert np.allclose(a, q_g @ r_g)
    q_gr, r_gr = qr_gram(a)
    assert np.allclose(a, q_gr @ r_gr)

if __name__ == "__main__":
    test_qr()