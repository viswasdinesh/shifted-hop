#!/usr/bin/env python3
"""Does HOP's selected T* depend on the arbitrary regularization constants
buried in the published implementation -- the 1e-9/1e-12 jitter added inside
build_augmented_system, and chol_inv's default jitter=1e-9 -- rather than on
genuine problem structure?

If T* moves as these constants are swept over orders of magnitude, on the
*standard quadratic-cost path* (the actual repo code, unmodified except for
exposing these constants as keyword args with unchanged defaults), that is
evidence of Assumption-2 fragility that requires no cartpole and no cost
substitution at all -- it shows up on the double integrator, the easiest
possible case.

Run order: double integrator (linear, fastest) -> quadrotor -> cartpole,
each varying eps_Q, eps_scalar (build_augmented_system) and jitter (chol_inv,
used throughout hop_horizon_search) one at a time, holding the others at
their repo defaults (1e-9, 1e-12, 1e-9 respectively).
"""

import numpy as np

from ddp import linearize_trajectory
from lqr import build_augmented_system, hop_horizon_search
from systems import make_cartpole, make_double_integrator, make_quadrotor
from utils import rollout

SWEEP_VALUES = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1]
DEFAULTS = {"eps_Q": 1e-9, "eps_scalar": 1e-12, "jitter": 1e-9}


def build_di_system():
    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max = make_double_integrator()
    dt = F.dt
    A_di = np.array([[1.0, dt], [0.0, 1.0]])
    B_di = np.array([[0.0], [dt]])
    A_list = [A_di] * N
    B_list = [B_di] * N
    X = rollout(F, x0, np.tile(u_ref.reshape(1, -1), (N, 1)))
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, A_list, B_list, X, U, None


def build_nonlinear_system(make_fn):
    out = make_fn()
    if len(out) == 12:
        F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = out
    else:
        F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max = out
        wrap_idx = None
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    X = rollout(F, x0, U)
    A_list, B_list = linearize_trajectory(F, X, U)
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, A_list, B_list, X, U, wrap_idx


def T_star_for(F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx,
                T_min, T_max, eps_Q, eps_scalar, jitter):
    A_aug, B_aug, Q_aug, R_mat, z0, QT_list = build_augmented_system(
        F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx,
        eps_Q=eps_Q, eps_scalar=eps_scalar)
    J = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, jitter=jitter)
    J = np.where(np.isfinite(J), J, np.inf)
    T_star = int(np.argmin(J[T_min - 1:T_max]) + T_min)
    return T_star, J[T_star - 1]


def sweep_one_system(name, F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max,
                      A_list, B_list, X, U, wrap_idx):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    base_T, base_J = T_star_for(F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx,
                                 T_min, T_max, **DEFAULTS)
    print(f"Default constants {DEFAULTS} -> T*={base_T}, J*={base_J:.4f}")

    any_moved = False
    for varying in ["eps_Q", "eps_scalar", "jitter"]:
        print(f"\n  Sweeping {varying} (others fixed at repo defaults):")
        print(f"  {'value':>10} {'T*':>5} {'J*':>14} {'T* moved?':>10}")
        for val in SWEEP_VALUES:
            kwargs = dict(DEFAULTS); kwargs[varying] = val
            T_star, J_star = T_star_for(F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx,
                                         T_min, T_max, **kwargs)
            moved = T_star != base_T
            any_moved = any_moved or moved
            print(f"  {val:10.0e} {T_star:5d} {J_star:14.4f} {'<-- YES' if moved else '':>10}")

    print(f"\n  {name}: T* changed somewhere in the sweep: {any_moved}")
    return any_moved


def main():
    results = {}

    di = build_di_system()
    results["Double Integrator"] = sweep_one_system("Double Integrator", *di)

    quad = build_nonlinear_system(make_quadrotor)
    results["Quadrotor"] = sweep_one_system("Quadrotor", *quad)

    cart = build_nonlinear_system(make_cartpole)
    results["Cartpole"] = sweep_one_system("Cartpole", *cart)

    print(f"\n{'='*70}\nSummary: does T* depend on the arbitrary regularization constants?\n{'='*70}")
    for name, moved in results.items():
        print(f"  {name:<20} {'YES -- T* is sensitive to jitter/epsilon' if moved else 'no -- T* stable across the sweep'}")


if __name__ == "__main__":
    main()
