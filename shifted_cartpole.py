"""
Shifted chart  Phat = (P + cI)^{-1}  on the cartpole augmented system.

psi_k = h_k o phi   with
    Ehat = Qbar^{-1},  Fhat = Qbar^{-1} A^T,  Ghat = A Qbar^{-1} A^T + B R^{-1} B^T,  Qbar = Q + cI
    K    = (I - c Ghat)^{-1}
    E'   = Ehat + c Fhat K Fhat^T,   F' = Fhat K,   G' = Ghat K
then HOP's Eq.14 recursion verbatim on (E',F',G').

Two cost settings:
  PD    -- the repo's own Q_aug (quadratic in wrapped error; Schur complement = 2w > 0)
  INDEF -- true Eq.30 Q_aug for the (1-cos th)^2 swing-up cost; indefinite near th=pi
"""

import numpy as np

from ddp import linearize_trajectory
from lqr import build_augmented_system, hop_horizon_search
from utils import _sym, _wrap, rollout
from pair_gate import riccati_reference
from systems import make_cartpole


# ----------------------------------------------------------------- shifted chart
def shifted_horizon_search(A_aug, B_aug, Q_aug, R, z0, QT_list, T_max, c):
    n = A_aug[0].shape[0]
    I = np.eye(n)
    R_inv = np.linalg.inv(R)

    Ep, Fp, Gp = [], [], []
    for k in range(T_max):
        Qbar = _sym(Q_aug[k]) + c * I
        if np.linalg.eigvalsh(Qbar)[0] <= 0:
            return None, "Qbar not PD"
        Qbi = np.linalg.inv(Qbar)
        Eh = _sym(Qbi)
        Fh = Qbi @ A_aug[k].T
        Gh = _sym(A_aug[k] @ Qbi @ A_aug[k].T + B_aug[k] @ R_inv @ B_aug[k].T)
        M = I - c * Gh
        if abs(np.linalg.det(M)) < 1e-300 or np.linalg.cond(M) > 1e14:
            return None, f"I-cG singular at k={k} (cond={np.linalg.cond(M):.1e})"
        K = np.linalg.inv(M)                      # symmetric, indefinite -> not Cholesky
        Ep.append(_sym(Eh + c * Fh @ K @ Fh.T))
        Fp.append(Fh @ K)
        Gp.append(_sym(Gh @ K))

    # Phase 1: compose (HOP Eq. 14, unchanged)
    Eb, Fb, Gb = [Ep[0].copy()], [Fp[0].copy()], [Gp[0].copy()]
    for k in range(1, T_max):
        W = np.linalg.inv(Ep[k] + Gb[-1])
        Eb.append(_sym(Eb[-1] - Fb[-1] @ W @ Fb[-1].T))
        Fb.append(Fb[-1] @ W @ Fp[k])
        Gb.append(_sym(Gp[k] - Fp[k].T @ W @ Fp[k]))

    # Phase 2: query
    J = np.zeros(T_max)
    for t in range(1, T_max + 1):
        QT = _sym(QT_list[t - 1]) + c * I
        if np.linalg.eigvalsh(QT)[0] <= 0:
            return None, f"QT+cI not PD at t={t}"
        PhatT = np.linalg.inv(QT)
        Phat0 = _sym(Eb[t - 1] - Fb[t - 1] @ np.linalg.inv(PhatT + Gb[t - 1]) @ Fb[t - 1].T)
        P0 = _sym(np.linalg.inv(Phat0)) - c * I
        J[t - 1] = 0.5 * float(z0 @ P0 @ z0)
    return J, None


# ------------------------------------------------- true Eq.30 Q_aug (nonconvex cost)
def true_cost_augmented(X, U, xg, qs, r, qfs, w, N, wrap_idx):
    """l = .5[qp p^2 + qth(1-cos th)^2 + qpd pd^2 + qthd thd^2] + .5 r u^2 + w"""
    qp, qth, qpd, qthd = qs
    n = 4
    Q_aug, QT_list = [], []
    for k in range(N):
        p, th, pd, thd = X[k]
        s, cth = np.sin(th), np.cos(th)
        lx = np.array([qp * p, qth * (1 - cth) * s, qpd * pd, qthd * thd])
        lxx = np.diag([qp, qth * (s * s + cth - cth * cth), qpd, qthd])   # -2*qth at th=pi
        lu = np.array([r * U[k, 0]]); luu = np.array([[r]])
        l0 = 0.5 * (qp * p**2 + qth * (1 - cth) ** 2 + qpd * pd**2 + qthd * thd**2) \
             + 0.5 * r * U[k, 0] ** 2
        Qk = np.zeros((n + 1, n + 1))
        Qk[:n, :n] = _sym(lxx)                      # l_xu = 0 -> no Schur correction
        Qk[:n, n] = lx; Qk[n, :n] = lx
        Qk[n, n] = 2.0 * (l0 + w - 0.5 * float(lu @ np.linalg.solve(luu, lu)))
        Q_aug.append(_sym(Qk))
    qfp, qfth, qfpd, qfthd = qfs
    for t in range(1, N + 1):
        p, th, pd, thd = X[t]
        s, cth = np.sin(th), np.cos(th)
        px = np.array([qfp * p, qfth * (1 - cth) * s, qfpd * pd, qfthd * thd])
        pxx = np.diag([qfp, qfth * (s * s + cth - cth * cth), qfpd, qfthd])
        p0 = 0.5 * (qfp * p**2 + qfth * (1 - cth) ** 2 + qfpd * pd**2 + qfthd * thd**2)
        Qt = np.zeros((n + 1, n + 1))
        Qt[:n, :n] = _sym(pxx); Qt[:n, n] = px; Qt[n, :n] = px
        Qt[n, n] = 2.0 * p0
        QT_list.append(_sym(Qt))
    return Q_aug, QT_list


def report(tag, Q_aug, QT_list, A_aug, B_aug, R_mat, z0, T_max, cs):
    mins = [np.linalg.eigvalsh(_sym(Q))[0] for Q in Q_aug[:T_max]]
    minsT = [np.linalg.eigvalsh(_sym(Q))[0] for Q in QT_list[:T_max]]
    print(f"\n{'='*80}\n{tag}\n{'='*80}")
    print(f"min eig over stages  Q_aug: {min(mins):+.4e}   (indefinite: {min(mins) < 0})")
    print(f"min eig over terminals QT  : {min(minsT):+.4e}")

    J_ref = riccati_reference(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
    try:
        J_hop = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
        e_hop = np.abs(J_hop - J_ref) / np.maximum(np.abs(J_ref), 1e-12)
        hop_txt = f"{e_hop.max():.3e}   T*={int(np.argmin(J_hop))+1}"
    except Exception as exc:
        hop_txt = f"FAILED ({type(exc).__name__})"
    print(f"HOP (unshifted chart) max relerr : {hop_txt}")
    print(f"reference T* = {int(np.argmin(J_ref))+1}")

    print(f"\n{'c':>10} {'max relerr':>13} {'relerr@t=40':>13} {'relerr@t=T':>13} {'T*':>5}  note")
    for c in cs:
        J, err = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, c)
        if J is None:
            print(f"{c:>10.3g} {'--':>13} {'--':>13} {'--':>13} {'--':>5}  {err}")
            continue
        e = np.abs(J - J_ref) / np.maximum(np.abs(J_ref), 1e-12)
        print(f"{c:>10.3g} {e.max():>13.3e} {e[39]:>13.3e} {e[-1]:>13.3e} "
              f"{int(np.argmin(J))+1:>5}")


if __name__ == "__main__":
    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = make_cartpole()
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    X = rollout(F, x0, U)
    A_list, B_list = linearize_trajectory(F, X, U)
    A_aug, B_aug, Q_aug_pd, R_mat, z0, QT_pd = build_augmented_system(
        F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx)

    cs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0, 1e2, 1e3, 1e4, 1e6]

    report("CARTPOLE -- repo Q_aug (PD by construction)",
           Q_aug_pd, QT_pd, A_aug, B_aug, R_mat, z0, T_max, cs)

    Q_aug_ind, QT_ind = true_cost_augmented(
        X, U, xg, qs=(1.0, 10.0, 0.1, 0.1), r=0.01,
        qfs=(alpha, alpha, alpha, alpha), w=w, N=N, wrap_idx=wrap_idx)
    report("CARTPOLE -- true Eq.30 Q_aug, (1-cos th)^2 cost (INDEFINITE)",
           Q_aug_ind, QT_ind, A_aug, B_aug, R_mat, z0, T_max, cs)