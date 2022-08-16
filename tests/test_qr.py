import numpy as np

from linalg.qr_decomp import qr_house, qr_givens, qr_gram, qr_house_piv

def test_qr():
    a = np.array([
        [1, 2, 3],
        [1, 5, 6],
        [1, 8, 9],
        [1, 11, 12]
    ])
    q_p, r_p, p = qr_house_piv(a)
    assert np.allclose(a, q_p @ r_p @ p)

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
    q_p, r_p, p = qr_house_piv(a)
    assert np.allclose(a, q_p @ r_p @ p)

    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((9, 6))
    q, r = qr_house(a, mode="full")
    assert np.allclose(a, q @ r)
    q_g, r_g = qr_givens(a, mode="full")
    assert np.allclose(a, q_g @ r_g)
    q_gr, r_gr = qr_gram(a)
    assert np.allclose(a, q_gr @ r_gr)
    q_p, r_p, p = qr_house_piv(a)
    assert np.allclose(a, q_p @ r_p @ p)

    q_e, r_e = qr_house(a, mode="economic")
    assert np.allclose(q[:, :6], q_e)
    assert np.allclose(r[:6], r_e)
    assert np.allclose(a, q_e @ r_e)
    q_g_e, r_g_e = qr_givens(a, mode="economic")
    assert np.allclose(q_g[:, :6], q_g_e)
    assert np.allclose(r_g[:6], r_g_e)
    assert np.allclose(a, q_g_e @ r_g_e)
    q_p_e, r_p_e, p_e = qr_house_piv(a, mode="economic", decode_p=False)
    assert np.allclose(q_p[:, :6], q_p_e)
    assert np.allclose(r_p[:6], r_p_e)
    assert np.allclose(a[:, p_e], q_p_e @ r_p_e)
    
    a = rng.standard_normal((6, 9))
    q, r = qr_house(a, mode="full")
    assert np.allclose(a, q @ r)
    q_g, r_g = qr_givens(a, mode="full")
    assert np.allclose(a, q_g @ r_g)
    q_gr, r_gr = qr_gram(a)
    assert np.allclose(a, q_gr @ r_gr)
    q_p, r_p, p = qr_house_piv(a)
    assert np.allclose(a, q_p @ r_p @ p)

if __name__ == "__main__":
    test_qr()