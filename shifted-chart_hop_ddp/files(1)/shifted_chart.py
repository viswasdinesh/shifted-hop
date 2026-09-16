"""
Shifted-chart horizon-optimal LQR.

Reference implementation of the result in THEOREM.md: HOP's O(N) composed-map
horizon search, extended to indefinite stage/terminal cost matrices by tracking

    Phat_k = (P_k + c I)^{-1}          instead of      Ptilde_k = P_k^{-1}

Drop-in replacement for `lqr.hop_horizon_search` from
rap-lab-org/public_HOP_horizon_optimal_tutorial. Same signature plus `c`.

No dependency on the HOP repo; `_sym` is reproduced locally.
"""

import numpy as np

__all__ = ["shifted_horizon_search", "choose_shift", "psi_params", "compose_maps"]


def _sym(A):
    return 0.5 * (A + A.T)


# --------------------------------------------------------------------------
# Shift selection
# --------------------------------------------------------------------------
def choose_shift(Q_aug, QT_list, T_max=None, margin=2.0, floor=1e-3):
    """
    c must exceed -lambda_min over every stage and terminal cost matrix so that
    Qbar = Q + cI is positive definite everywhere.  Computed, not tuned.

    `margin` multiplies the strict floor.  Empirically anything in
    [1.001x, 100x] the floor gives ~1e-14..1e-11 relative accuracy; accuracy
    degrades only past ~1e3x, where the P = Phat^{-1} - cI recovery starts
    losing digits to cancellation.
    """
    T = len(Q_aug) if T_max is None else T_max
    lo = 0.0
    for M in list(Q_aug[:T]) + list(QT_list[:T]):
        lo = min(lo, float(np.linalg.eigvalsh(_sym(M))[0]))
    return max(margin * (-lo), floor)


# --------------------------------------------------------------------------
# Single-step maps
# --------------------------------------------------------------------------
def psi_params(A, B, Q, R_inv, c):
    """
    Coefficients of psi_k = h_k o phi, the one-step map in shifted coordinates.

        Qbar = Q + cI
        Ehat = Qbar^-1,  Fhat = Qbar^-1 A^T,  Ghat = A Qbar^-1 A^T + B R^-1 B^T
        K    = (I - c Ghat)^-1
        E'   = Ehat + c Fhat K Fhat^T,   F' = Fhat K,   G' = Ghat K

    Returns (E', F', G').  Raises np.linalg.LinAlgError if Qbar is not PD or
    I - c*Ghat is singular.

    Note: I - c*Ghat is symmetric but INDEFINITE whenever c > 1/lambda_max(Ghat),
    which is the normal operating regime.  Use LU/LDL^T, never Cholesky.
    """
    n = Q.shape[0]
    I = np.eye(n)
    Qbar = _sym(Q) + c * I
    if np.linalg.eigvalsh(Qbar)[0] <= 0.0:
        raise np.linalg.LinAlgError("Q + cI is not positive definite; increase c")

    Qbi = np.linalg.inv(Qbar)
    Ehat = _sym(Qbi)
    Fhat = Qbi @ A.T
    Ghat = _sym(A @ Qbi @ A.T + B @ R_inv @ B.T)

    M = I - c * Ghat
    if np.linalg.cond(M) > 1e14:
        raise np.linalg.LinAlgError(
            f"I - c*Ghat is numerically singular (cond={np.linalg.cond(M):.2e}); "
            "c is near 1/lambda_i(Ghat)")

    # One factorization, two solves, instead of an explicit inverse.
    lu = np.linalg.solve            # placeholder for an LDL^T routine
    Fp = lu(M.T, Fhat.T).T
    Gp = _sym(lu(M.T, Ghat.T).T)
    Ep = _sym(Ehat + c * Fp @ Fhat.T)
    return Ep, Fp, Gp


# --------------------------------------------------------------------------
# Composition (HOP Eq. 14, unchanged, applied to the primed coefficients)
# --------------------------------------------------------------------------
def compose_maps(Ep, Fp, Gp):
    """
    Forward sweep building the composed maps psi_{0:k} for every k.

        W_k    = (E'_k + Gbar_{k-1})^-1
        Ebar_k = Ebar_{k-1} - Fbar_{k-1} W_k Fbar_{k-1}^T
        Fbar_k = Fbar_{k-1} W_k F'_k
        Gbar_k = G'_k - F'_k^T W_k F'_k

    Returns lists (Ebar, Fbar, Gbar), each of length len(Ep).
    """
    Ebar, Fbar, Gbar = [Ep[0].copy()], [Fp[0].copy()], [Gp[0].copy()]
    for k in range(1, len(Ep)):
        W = np.linalg.inv(Ep[k] + Gbar[-1])
        Ebar.append(_sym(Ebar[-1] - Fbar[-1] @ W @ Fbar[-1].T))
        Fbar.append(Fbar[-1] @ W @ Fp[k])
        Gbar.append(_sym(Gp[k] - Fp[k].T @ W @ Fp[k]))
    return Ebar, Fbar, Gbar


# --------------------------------------------------------------------------
# Full horizon search
# --------------------------------------------------------------------------
def shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, c=None):
    """
    Cost J_t for every candidate horizon t = 1..T_max, in O(T_max n^3).

    Same interface as lqr.hop_horizon_search, plus the shift `c`.  If c is None
    it is computed by choose_shift().  Returns (J, c).

    Unlike hop_horizon_search this does NOT require Q_aug[k] > 0.
    """
    n = A_aug[0].shape[0]
    I = np.eye(n)
    if c is None:
        c = choose_shift(Q_aug, QT_list, T_max)
    R_inv = np.linalg.inv(R_mat)

    Ep, Fp, Gp = [], [], []
    for k in range(T_max):
        e, f, g = psi_params(A_aug[k], B_aug[k], Q_aug[k], R_inv, c)
        Ep.append(e); Fp.append(f); Gp.append(g)

    Ebar, Fbar, Gbar = compose_maps(Ep, Fp, Gp)

    J = np.zeros(T_max)
    for t in range(1, T_max + 1):
        QT = _sym(QT_list[t - 1]) + c * I
        if np.linalg.eigvalsh(QT)[0] <= 0.0:
            raise np.linalg.LinAlgError(
                f"Q_T + cI not PD at t={t}; increase c")
        PhatT = np.linalg.inv(QT)
        Phat0 = _sym(Ebar[t - 1]
                     - Fbar[t - 1] @ np.linalg.inv(PhatT + Gbar[t - 1]) @ Fbar[t - 1].T)
        P0 = _sym(np.linalg.inv(Phat0)) - c * I          # undo the shift
        J[t - 1] = 0.5 * float(z0 @ P0 @ z0)
    return J, c
