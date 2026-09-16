#!/usr/bin/env python3
"""Validate the shifted-chart horizon search (shifted_chart.py) inside an
actual *converging* DDP loop on cartpole swing-up -- the gap THEOREM.md's
Limitations #1 states explicitly:

    "Not validated inside a converging DDP loop. All cartpole numbers use
    the zero-control nominal, where the pole sits at th=pi for every stage.
    J(T) is therefore monotone and the reference T* sits at the search
    boundary. The accuracy comparison is sound; a claim about correct
    *horizon selection* is not yet supported and needs a converging
    swing-up."

This reuses audit_cartpole_indefiniteness.py's true-cost machinery (the
paper's actual Eq. 28/30/32 augmented system, built from the real
(1-cos theta)^2 swing-up cost via autodiff-free finite differences, plus
the Vxx-clipped brute-force Riccati sweep that serves as ground truth) and
the same one-shot "flick" warm start that broke the exact-fixed-point
degeneracy earlier in this investigation. At every outer DDP iteration it
computes T* three ways -- raw (published, unshifted) HOP-LQR, the shifted
chart, and brute-force ground truth -- and runs two parallel DDP loops
(one driven by each of raw/shifted T*) far enough to see whether either
actually converges to a swing-up, not just whether they agree on iteration
0.

FINDING (see THEOREM.md Sec. 6, item 1 for the full writeup): the flick
warm start does the job -- both loops now run many outer iterations instead
of stalling at iteration 0. But on that genuinely time-varying trajectory,
raw hop_horizon_search's known Assumption-2 failure is joined by a SECOND,
independent failure that the shift does not fix: J_shift(T) tracks the
brute-force ground truth for small T, then develops large spurious negative
spikes at specific horizons where Quu passes near-singular (e.g. -14274 at
T=47 vs. the true minimum of -569 at T=17) -- these match a fully
UNREGULARIZED backward Riccati sweep almost exactly, which makes sense:
Theorem 1 proves the shift is numerically exact, so it faithfully
reproduces the ordinary Riccati recursion's own control-side conditioning
issue, orthogonal to the Q_k-indefiniteness problem the shift targets. The
per-trial "success"/T*-agreement numbers this script prints should be read
with that caveat -- T_shift is not yet a safe drop-in inside argmin() on a
dynamic trajectory without also regularizing Quu (Algorithm 2's Step 3 does
this, but only after T* is chosen, i.e. too late for Step 2's search).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                 "shifted-chart_hop_ddp", "files(1)"))
from shifted_chart import choose_shift, shifted_horizon_search  # noqa: E402

from audit_cartpole_indefiniteness import (  # noqa: E402
    build_augmented_system_true_cost, cost_derivatives, cost_true_nonlinear,
    forward_pass_true_cost, ilqr_backward_pass_true_cost, make_swingup_cost,
    regularize_to_pd, terminal_derivatives,
)
from ddp import linearize_trajectory  # noqa: E402
from lqr import hop_horizon_search  # noqa: E402
from systems import make_cartpole  # noqa: E402
from utils import rollout  # noqa: E402


def safe_argmin_T(J, T_min, T_max):
    J = np.where(np.isfinite(J), J, np.inf)
    return int(np.argmin(J[T_min - 1:T_max]) + T_min)


def bruteforce_true_cost(A_list, B_list, costs_k, terms_t, w, T_max, vxx_min_eig=-50.0):
    """Same Vxx-clipped ground-truth Riccati sweep as the audit script
    (kept local so this file doesn't depend on that script's internal
    clip_vxx helper's exact import path)."""
    from audit_cartpole_indefiniteness import clip_vxx
    from utils import _sym, chol_solve
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]
    J = np.zeros(T_max)
    for T in range(1, T_max + 1):
        p0, px, pxx = terms_t[T]
        Vxx, Vx, V0 = clip_vxx(pxx, vxx_min_eig), px.copy(), p0
        for t in reversed(range(T)):
            l0, lx, lu, lxx, luu, lxu = costs_k[t]
            lux = lxu.T
            A, B = A_list[t], B_list[t]
            Qx = lx + A.T @ Vx; Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A; Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A
            Quu_reg, _ = regularize_to_pd(Quu, margin=1.0)
            invQu = chol_solve(Quu_reg, Qu); invQux = chol_solve(Quu_reg, Qux)
            Vxx = clip_vxx(_sym(Qxx - Qux.T @ invQux), vxx_min_eig)
            Vx = Qx - Qux.T @ invQu
            V0 = l0 + w + V0 - 0.5 * float(Qu @ invQu)
        J[T - 1] = V0
    return J


def run_ddp_loop(F, x0, xg, ell, phi, w, N, T_min, T_max, method,
                  max_iter=25, flick_amp=1.0, flick_steps=5, verbose=False):
    """method in {'raw', 'shifted'}: which horizon-selection result drives
    the Step 3/4 backward+forward pass at each outer iteration."""
    U = np.zeros((N, 1))
    U[:flick_steps, 0] += flick_amp
    X = rollout(F, x0, U)

    history = []
    for it in range(max_iter + 1):
        A_list, B_list = linearize_trajectory(F, X, U)
        costs_k = [cost_derivatives(ell, X[k], U[k]) for k in range(N)]
        terms_t = [terminal_derivatives(phi, X[t]) for t in range(N + 1)]

        A_aug, B_aug, Q_aug, R_mat, z0, QT_list, stage_info = \
            build_augmented_system_true_cost(F, A_list, B_list, X, U, costs_k, terms_t, w)
        any_indef = any(not s["is_pd"] for s in stage_info)

        with np.errstate(over="ignore", invalid="ignore"):
            J_raw = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
            T_raw = safe_argmin_T(J_raw, T_min, T_max)

            try:
                c = choose_shift(Q_aug, QT_list, T_max)
                J_shift, c_used = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, c=c)
                T_shift = safe_argmin_T(J_shift, T_min, T_max)
            except np.linalg.LinAlgError as exc:
                T_shift, c_used = T_raw, float("nan")
                if verbose:
                    print(f"    shifted_horizon_search failed: {exc}")

            J_bf = bruteforce_true_cost(A_list, B_list, costs_k, terms_t, w, T_max)
            T_bf = safe_argmin_T(J_bf, T_min, T_max)

        T_star = T_raw if method == "raw" else T_shift
        history.append({"iter": it, "any_indefinite": any_indef, "c": c_used,
                         "T_raw": T_raw, "T_shift": T_shift, "T_bf": T_bf,
                         "T_used": T_star})
        if verbose:
            print(f"    it={it:2d} indef={any_indef!s:5} c={c_used:9.3g} "
                  f"T_raw={T_raw:3d} T_shift={T_shift:3d} T_bf={T_bf:3d} T_used({method})={T_star:3d}")

        acc = False
        with np.errstate(over="ignore", invalid="ignore"):
            for lm in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
                k_list, K_list, ok = ilqr_backward_pass_true_cost(A_list, B_list, costs_k, terms_t, T_star, lm=lm)
                if not ok or not np.all(np.isfinite(k_list[0])) or any(not np.all(np.isfinite(Kk)) for Kk in K_list):
                    continue
                Xn, Un, Jn, acc = forward_pass_true_cost(F, X, U, ell, phi, w, T_star, k_list, K_list)
                if acc:
                    break
        if not acc:
            break
        X, U = Xn, Un

    T_final = history[-1]["T_used"] if history else T_min
    final_err = float(np.linalg.norm(X[T_final] - xg))
    success = final_err <= 0.5
    return {"history": history, "final_err": final_err, "success": success,
             "n_iters": len(history), "X": X, "U": U, "T_final": T_final}


def main():
    F, x0, xg, u_ref, *_ = make_cartpole()
    N, T_min, T_max = 120, 1, 60

    rng = np.random.default_rng(0)
    print(f"{'w':>7} {'q_theta':>8} {'r':>7} | "
          f"{'raw: iters':>10} {'succ':>5} {'err':>8} {'T_final':>7} | "
          f"{'shift: iters':>12} {'succ':>5} {'err':>8} {'T_final':>7} | "
          f"{'any raw!=bf':>11} {'any shift!=bf':>13}")

    n_trials = 8
    raw_success = shift_success = 0
    raw_bf_ever_disagree = shift_bf_ever_disagree = 0
    for _ in range(n_trials):
        w = float(rng.choice([0.005, 0.01, 0.02, 0.05, 0.1]))
        q_theta = float(rng.uniform(5.0, 20.0))
        r = float(rng.uniform(5e-3, 5e-2))
        ell, phi = make_swingup_cost(q_theta=q_theta, r=r)

        res_raw = run_ddp_loop(F, x0, xg, ell, phi, w, N, T_min, T_max, method="raw", max_iter=25)
        res_shift = run_ddp_loop(F, x0, xg, ell, phi, w, N, T_min, T_max, method="shifted", max_iter=25)

        raw_success += res_raw["success"]; shift_success += res_shift["success"]
        any_raw_bf = any(h["T_raw"] != h["T_bf"] for h in res_raw["history"])
        any_shift_bf = any(h["T_shift"] != h["T_bf"] for h in res_shift["history"])
        raw_bf_ever_disagree += any_raw_bf; shift_bf_ever_disagree += any_shift_bf

        print(f"{w:7.3f} {q_theta:8.2f} {r:7.4f} | "
              f"{res_raw['n_iters']:10d} {res_raw['success']!s:>5} {res_raw['final_err']:8.3f} {res_raw['T_final']:7d} | "
              f"{res_shift['n_iters']:12d} {res_shift['success']!s:>5} {res_shift['final_err']:8.3f} {res_shift['T_final']:7d} | "
              f"{any_raw_bf!s:>11} {any_shift_bf!s:>13}")

    print(f"\n{'-'*100}")
    print(f"Trials: {n_trials}")
    print(f"Raw (published) DDP loop:    success {raw_success}/{n_trials}   "
          f"T_raw disagreed with brute-force ground truth at some iteration: {raw_bf_ever_disagree}/{n_trials}")
    print(f"Shifted-chart DDP loop:      success {shift_success}/{n_trials}   "
          f"T_shift disagreed with brute-force ground truth at some iteration: {shift_bf_ever_disagree}/{n_trials}")


if __name__ == "__main__":
    main()
