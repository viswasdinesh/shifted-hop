"""THEOREM.md Sec. 5.5: does the shifted chart hold up at n_aug=13 (the
quadrotor's dimension) -- the scale where an earlier chart-free formulation
(symplectic [X;Y] pair propagation) failed catastrophically (cond ~1e5/step,
unusable by t=5)? Two tests, since they check different things:

(a) The quadrotor's own cost via build_augmented_system, all-zero-control
    nominal, T=160. Its Q_aug is already ~PD (min eig ~7.7e-3 at stages,
    ~2.4e-14 at terminals), so this tests CONDITIONING at scale, not
    indefiniteness at scale. Requires ddp.py/lqr.py/systems.py/utils.py on
    the import path.

(b) Synthetic indefinite Q_k at the same n=13, m=4, T=160 (same
    construction as ltv_sweep_validated.py's LTV sweep, just at quadrotor's
    dimension instead of n=6). Closes the gap (a) leaves open: does
    genuine indefiniteness, not just conditioning, hold up at this scale?
    Requires only lqr.py.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from lqr import hop_horizon_search
from shifted_chart import shifted_horizon_search, choose_shift
from verify_cartpole import riccati_reference
from ltv_sweep_validated import make_Q, build_instance


def part_a():
    from ddp import linearize_trajectory
    from lqr import build_augmented_system
    from systems import make_quadrotor
    from utils import rollout

    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = make_quadrotor()
    print(f"(a) Quadrotor's own cost: N={N} T_max={T_max} n_aug={x0.size + 1}")
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    X = rollout(F, x0, U)
    A_list, B_list = linearize_trajectory(F, X, U)
    A_aug, B_aug, Q_aug, R_mat, z0, QT_list = build_augmented_system(
        F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx)

    lo_stage = min(np.linalg.eigvalsh(0.5 * (M + M.T))[0] for M in Q_aug)
    lo_term = min(np.linalg.eigvalsh(0.5 * (M + M.T))[0] for M in QT_list)
    print(f"    min eig stages={lo_stage:.4e}  terminals={lo_term:.4e}  (essentially PD already)")

    Jr = riccati_reference(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
    with np.errstate(over="ignore", invalid="ignore"):
        J_hop = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
        J_shift10, _ = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, c=10.0)
        c_auto = choose_shift(Q_aug, QT_list, T_max)
        J_shift_auto, _ = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, c=c_auto)

    for name, J in [("HOP (unshifted)", J_hop), ("shifted c=10", J_shift10),
                    (f"shifted c=choose_shift()={c_auto:.4g}", J_shift_auto)]:
        e = np.abs(J - Jr) / np.maximum(np.abs(Jr), 1e-12)
        T_star = int(np.argmin(J[T_min - 1:T_max]) + T_min)
        print(f"    {name:<32} max relerr={e.max():.3e}  relerr@t=160={e[-1]:.3e}  T*={T_star}")
    T_ref = int(np.argmin(Jr[T_min - 1:T_max]) + T_min)
    print(f"    reference T* = {T_ref}")


def part_b():
    n, m, T = 13, 4, 160
    print(f"\n(b) Synthetic indefinite Q_k at n={n}, m={m}, T={T} (not the quadrotor's actual dynamics)")
    for lam, rd in [(1.0, 0), (0.0, 0), (-1.0, 0), (-10.0, 0)]:
        rng = np.random.default_rng(3)
        A, B, R, QT, z0 = build_instance(rng, n, m, T)
        Q = [make_Q(rng, n, lam, rd) for _ in range(T)]
        Jr = riccati_reference(A, B, Q, R, z0, QT, T)
        with np.errstate(over="ignore", invalid="ignore"):
            J_hop = hop_horizon_search(A, B, Q, R, z0, QT, T)
            c = choose_shift(Q, QT, T)
            J_shift, c_used = shifted_horizon_search(A, B, Q, R, z0, QT, T, c=c)
        e_hop = np.max(np.abs(J_hop - Jr) / np.maximum(np.abs(Jr), 1e-12))
        e_shift = np.max(np.abs(J_shift - Jr) / np.maximum(np.abs(Jr), 1e-12))
        T_ref = int(np.argmin(Jr)) + 1
        T_hop = int(np.argmin(J_hop)) + 1 if np.all(np.isfinite(J_hop)) else None
        T_shift = int(np.argmin(J_shift)) + 1
        print(f"    lam_min={lam:6.2g}  c={c_used:<6.4g}  "
              f"HOP relerr={e_hop:.2e} (T*={T_hop}, ref={T_ref})   "
              f"shift relerr={e_shift:.2e} (T*={T_shift})")


if __name__ == "__main__":
    part_a()
    part_b()
