"""
Empirical validation against the HOP repo on the Cartpole system.

Requires rap-lab-org/public_HOP_horizon_optimal_tutorial on the import path
(ddp.py, lqr.py, systems.py, utils.py).  Run from inside that repo with this
directory on PYTHONPATH, or copy these files into it.

Two cost settings:
  PD    -- the repo's own Q_aug (quadratic in wrapped error).  Schur complement
           of the top-left block is exactly 2w > 0, so Q_aug is PD by
           construction and Assumption 2 holds.  HOP is valid here.
  INDEF -- the paper's general Eq. 30 Q_aug for a (1-cos th)^2 swing-up cost.
           l_xx has eigenvalue q_th*(sin^2 th + cos th - cos^2 th), which is
           negative for th in (120, 240) degrees -- i.e. wherever a swing-up
           starts.  Assumption 2 fails.

For each, compares against a per-horizon sequential backward Riccati sweep
(ground truth, O(N^2)) and sweeps the shift c.

    python3 verify_cartpole.py
"""

import numpy as np

from ddp import linearize_trajectory
from lqr import build_augmented_system, hop_horizon_search
from systems import make_cartpole
from utils import rollout

from shifted_chart import _sym, choose_shift, shifted_horizon_search


def riccati_reference(A_aug, B_aug, Q_aug, R, z0, QT_list, T_max):
    """Ground truth: one full backward Riccati sweep per candidate horizon."""
    J = np.zeros(T_max)
    for t in range(1, T_max + 1):
        P = _sym(QT_list[t - 1])
        for k in range(t - 1, -1, -1):
            A, B, Q = A_aug[k], B_aug[k], Q_aug[k]
            S = R + B.T @ P @ B
            P = _sym(Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, B.T @ P @ A))
        J[t - 1] = 0.5 * float(z0 @ P @ z0)
    return J


def true_cost_augmented(X, U, qs, r, qfs, w, N):
    """
    Paper Eq. 30 augmented cost for
        l = .5[qp p^2 + qth(1-cos th)^2 + qpd pd^2 + qthd thd^2] + .5 r u^2 + w
    l_xu = 0, so Qtilde_k = l_xx exactly (no Schur correction).
    """
    qp, qth, qpd, qthd = qs
    n = 4
    Q_aug, QT_list = [], []
    for k in range(N):
        p, th, pd, thd = X[k]
        s, ct = np.sin(th), np.cos(th)
        lx = np.array([qp * p, qth * (1 - ct) * s, qpd * pd, qthd * thd])
        lxx = np.diag([qp, qth * (s * s + ct - ct * ct), qpd, qthd])
        lu = np.array([r * U[k, 0]])
        luu = np.array([[r]])
        l0 = 0.5 * (qp * p**2 + qth * (1 - ct) ** 2 + qpd * pd**2 + qthd * thd**2) \
            + 0.5 * r * U[k, 0] ** 2
        Qk = np.zeros((n + 1, n + 1))
        Qk[:n, :n] = _sym(lxx)
        Qk[:n, n] = lx
        Qk[n, :n] = lx
        Qk[n, n] = 2.0 * (l0 + w - 0.5 * float(lu @ np.linalg.solve(luu, lu)))
        Q_aug.append(_sym(Qk))

    qfp, qfth, qfpd, qfthd = qfs
    for t in range(1, N + 1):
        p, th, pd, thd = X[t]
        s, ct = np.sin(th), np.cos(th)
        px = np.array([qfp * p, qfth * (1 - ct) * s, qfpd * pd, qfthd * thd])
        pxx = np.diag([qfp, qfth * (s * s + ct - ct * ct), qfpd, qfthd])
        p0 = 0.5 * (qfp * p**2 + qfth * (1 - ct) ** 2 + qfpd * pd**2 + qfthd * thd**2)
        Qt = np.zeros((n + 1, n + 1))
        Qt[:n, :n] = _sym(pxx)
        Qt[:n, n] = px
        Qt[n, :n] = px
        Qt[n, n] = 2.0 * p0
        QT_list.append(_sym(Qt))
    return Q_aug, QT_list


def report(tag, A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max):
    lo_stage = min(np.linalg.eigvalsh(_sym(M))[0] for M in Q_aug[:T_max])
    lo_term = min(np.linalg.eigvalsh(_sym(M))[0] for M in QT_list[:T_max])
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
    print(f"min eig  stages = {lo_stage:+.4e}   terminals = {lo_term:+.4e}")
    print(f"Assumption 2 (Q_k > 0) holds: {lo_stage > 0 and lo_term > 0}")

    J_ref = riccati_reference(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)

    try:
        J_hop = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
        e = np.abs(J_hop - J_ref) / np.maximum(np.abs(J_ref), 1e-12)
        print(f"HOP  (unshifted chart)  max rel err = {e.max():.3e}")
    except Exception as exc:
        print(f"HOP  (unshifted chart)  FAILED: {type(exc).__name__}: {exc}")

    c_auto = choose_shift(Q_aug, QT_list, T_max)
    floor = max(0.0, -lo_stage, -lo_term)
    print(f"\nc floor = {floor:.4g}   choose_shift() = {c_auto:.4g}")
    print(f"\n{'c':>12} {'c/floor':>9} {'max rel err':>13} {'T*':>5}  note")
    if floor > 1e-6:
        grid = [floor * m for m in (1.001, 1.01, 1.1, 2, 5, 10, 100, 1e3, 1e4)]
    else:
        # floor is numerically zero -- c is unconstrained from below, sweep absolutely
        grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0, 1e2, 1e3, 1e4, 1e6]
    for c in grid:
        try:
            J, _ = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0,
                                          QT_list, T_max, c=c)
        except np.linalg.LinAlgError as exc:
            print(f"{c:>12.4g} {c / max(floor, 1):>9.3g} {'--':>13} {'--':>5}  {exc}")
            continue
        e = np.abs(J - J_ref) / np.maximum(np.abs(J_ref), 1e-12)
        print(f"{c:>12.4g} {c / max(floor, 1):>9.3g} {e.max():>13.3e} "
              f"{int(np.argmin(J)) + 1:>5}")


if __name__ == "__main__":
    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = make_cartpole()
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    X = rollout(F, x0, U)
    A_list, B_list = linearize_trajectory(F, X, U)
    A_aug, B_aug, Q_aug_pd, R_mat, z0, QT_pd = build_augmented_system(
        F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx)

    report("Cartpole -- repo Q_aug (PD by construction; HOP is valid here)",
           A_aug, B_aug, Q_aug_pd, R_mat, z0, QT_pd, T_max)

    Q_aug_ind, QT_ind = true_cost_augmented(
        X, U, qs=(1.0, 10.0, 0.1, 0.1), r=0.01,
        qfs=(alpha, alpha, alpha, alpha), w=w, N=N)

    report("Cartpole -- true Eq.30 Q_aug, (1-cos th)^2 cost (INDEFINITE)",
           A_aug, B_aug, Q_aug_ind, R_mat, z0, QT_ind, T_max)
