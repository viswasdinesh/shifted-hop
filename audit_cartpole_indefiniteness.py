#!/usr/bin/env python3
"""Audit: does HOP-DDP's Q_aug_k go indefinite on cartpole swing-up's true
non-convex cost, and if so, does it change the selected horizon T*?

See Notes.md for the full background. Summary of what this checks:

  - HOP-LQR (Sec. IV of the paper, p186.pdf) needs Q_k, Q_T strictly positive
    definite (Assumption 2) because it inverts them directly (Theorem 1:
    E_k = Q_k^-1). For HOP-DDP on a nonlinear problem, Q_k^aug is built from
    the *true* stage cost's Hessian ell_xx,k (Eq. 30), Schur-complemented
    against any state/control cross term. For the swing-up cost used here
    (separable, no cross term), Q_tilde_k = ell_xx,k exactly.

  - For ell(x,u) = 0.5*q_theta*(1-cos(theta))^2 + ..., the theta-theta entry
    of ell_xx is -2*q_theta*(2 cos(theta)+1)(cos(theta)-1), which is negative
    exactly for theta in (120 deg, 240 deg) -- see Notes.md's toy check.
    A swing-up trajectory starts at theta=pi and must cross this window.

  - Critically, none of this is special-cased in ddp.py/lqr.py: those files
    implement the *constant*-Q path (ell_xx == the user's fixed Q matrix),
    which is convex by construction and can never trigger this. This script
    is a separate, cartpole-only implementation of the true-cost path
    (Eq. 28/30/32 in the paper) needed to actually exercise it. It does not
    modify ddp.py/lqr.py/systems.py/run_cartpole.py.

  - Algorithm 2's only regularization (Quu += lambda*I) happens in Step 3,
    on Q_uu, *after* T* is already chosen in Step 2. Step 2 itself calls
    HOP-LQR on the raw Q_aug_k with no regularization anywhere. So the
    question is not "does Q_aug go indefinite" (expected: yes, per the
    analytical argument) but "does that indefiniteness actually change T*
    compared to a version where Q_aug is regularized before horizon
    selection". That comparison is the ablation in run_trial() below.
"""

import numpy as np

from ddp import linearize_trajectory
from lqr import hop_horizon_search
from utils import _sym, chol_inv, chol_solve, rollout
from systems import make_cartpole


# ---------------------------------------------------------------------------
# True non-convex swing-up cost (Notes.md), independent of ddp.py/lqr.py's
# constant-Q path.
# ---------------------------------------------------------------------------

def make_swingup_cost(q_p=1.0, q_theta=10.0, q_pdot=0.1, q_thetadot=0.1, r=1e-2,
                       qf_p=5.0, qf_theta=50.0, qf_pdot=0.5, qf_thetadot=0.5):
    def ell(x, u):
        p, th, pdot, thdot = x
        return 0.5*(q_p*p**2 + q_theta*(1-np.cos(th))**2
                     + q_pdot*pdot**2 + q_thetadot*thdot**2) + 0.5*r*u[0]**2

    def phi(x):
        p, th, pdot, thdot = x
        return 0.5*(qf_p*p**2 + qf_theta*(1-np.cos(th))**2
                     + qf_pdot*pdot**2 + qf_thetadot*thdot**2)

    return ell, phi


def cost_derivatives(ell, x, u, eps=1e-4):
    """Finite-difference l0, l_x, l_u, l_xx, l_uu, l_xu at (x, u)."""
    x = np.asarray(x, dtype=float); u = np.asarray(u, dtype=float)
    n, m = x.size, u.size
    l0 = ell(x, u)
    lx = np.zeros(n); lu = np.zeros(m)
    lxx = np.zeros((n, n)); luu = np.zeros((m, m)); lxu = np.zeros((n, m))

    for i in range(n):
        xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
        lx[i] = (ell(xp, u) - ell(xm, u)) / (2*eps)
    for j in range(m):
        up, um = u.copy(), u.copy(); up[j] += eps; um[j] -= eps
        lu[j] = (ell(x, up) - ell(x, um)) / (2*eps)

    for i in range(n):
        xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
        lxx[i, i] = (ell(xp, u) - 2*l0 + ell(xm, u)) / eps**2
        for k in range(i+1, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += eps; xpp[k] += eps
            xpm[i] += eps; xpm[k] -= eps
            xmp[i] -= eps; xmp[k] += eps
            xmm[i] -= eps; xmm[k] -= eps
            v = (ell(xpp, u) - ell(xpm, u) - ell(xmp, u) + ell(xmm, u)) / (4*eps**2)
            lxx[i, k] = v; lxx[k, i] = v

    for j in range(m):
        up, um = u.copy(), u.copy(); up[j] += eps; um[j] -= eps
        luu[j, j] = (ell(x, up) - 2*l0 + ell(x, um)) / eps**2
        for l in range(j+1, m):
            upp, upm, ump, umm = u.copy(), u.copy(), u.copy(), u.copy()
            upp[j] += eps; upp[l] += eps
            upm[j] += eps; upm[l] -= eps
            ump[j] -= eps; ump[l] += eps
            umm[j] -= eps; umm[l] -= eps
            v = (ell(x, upp) - ell(x, upm) - ell(x, ump) + ell(x, umm)) / (4*eps**2)
            luu[j, l] = v; luu[l, j] = v

    for i in range(n):
        for j in range(m):
            xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
            up, um = u.copy(), u.copy(); up[j] += eps; um[j] -= eps
            lxu[i, j] = (ell(xp, up) - ell(xp, um) - ell(xm, up) + ell(xm, um)) / (4*eps**2)

    return l0, lx, lu, _sym(lxx), _sym(luu), lxu


def terminal_derivatives(phi, x, eps=1e-4):
    """Finite-difference phi0, phi_x, phi_xx at x."""
    x = np.asarray(x, dtype=float)
    n = x.size
    p0 = phi(x)
    px = np.zeros(n); pxx = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
        px[i] = (phi(xp) - phi(xm)) / (2*eps)
    for i in range(n):
        xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
        pxx[i, i] = (phi(xp) - 2*p0 + phi(xm)) / eps**2
        for k in range(i+1, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += eps; xpp[k] += eps
            xpm[i] += eps; xpm[k] -= eps
            xmp[i] -= eps; xmp[k] += eps
            xmm[i] -= eps; xmm[k] -= eps
            v = (phi(xpp) - phi(xpm) - phi(xmp) + phi(xmm)) / (4*eps**2)
            pxx[i, k] = v; pxx[k, i] = v
    return p0, px, _sym(pxx)


# ---------------------------------------------------------------------------
# True-cost augmented system (paper Eq. 28, 30, 32) -- generalizes
# lqr.py's build_augmented_system (which assumes ell_xx=const Q, ell_xu=0,
# ell_u=R@du) to an arbitrary twice-differentiable ell, phi.
# ---------------------------------------------------------------------------

def build_augmented_system_true_cost(F, A_list, B_list, X, U, costs_k, terms_t, w):
    N_steps, n, m = len(A_list), X.shape[1], U.shape[1]

    A_aug_list, B_aug_list, Q_aug_list, R_list = [], [], [], []
    stage_info = []  # per-stage diagnostics: theta, Qtilde eigvals, aug PD flag
    for k in range(N_steps):
        x, u = X[k], U[k]
        l0, lx, lu, lxx, luu, lxu = costs_k[k]
        luu_inv = chol_inv(luu)

        a_k = (F(x, u) - X[k + 1]).reshape(-1)
        lux = lxu.T

        Ak_aug = np.zeros((n+1, n+1))
        Ak_aug[:n, :n] = A_list[k] - B_list[k] @ luu_inv @ lux
        Ak_aug[:n, n] = a_k - (B_list[k] @ luu_inv @ lu).ravel()
        Ak_aug[n, n] = 1.0
        Bk_aug = np.zeros((n+1, m)); Bk_aug[:n, :] = B_list[k]

        Qtilde = _sym(lxx - lxu @ luu_inv @ lux)
        qtilde = lx - (lxu @ luu_inv @ lu).ravel()

        Qk = np.zeros((n+1, n+1))
        Qk[:n, :n] = Qtilde
        Qk[:n, n] = qtilde; Qk[n, :n] = qtilde
        Qk[n, n] = 2.0*(l0 + w - 0.5*float(lu @ luu_inv @ lu))
        Qk = _sym(Qk)

        A_aug_list.append(Ak_aug); B_aug_list.append(Bk_aug)
        Q_aug_list.append(Qk); R_list.append(_sym(luu))

        eig_tilde = np.linalg.eigvalsh(Qtilde)
        eig_full = np.linalg.eigvalsh(Qk)
        stage_info.append({
            "k": k, "theta_deg": float(np.degrees(x[1]) % 360),
            "min_eig_Qtilde": float(eig_tilde.min()),
            "min_eig_Qaug": float(eig_full.min()),
            "bottom_right": float(Qk[n, n]),
            "is_pd": bool(eig_full.min() > 0),
        })

    z0 = np.zeros(n + 1); z0[-1] = 1.0

    QT_list = []
    for t in range(1, N_steps + 1):
        p0, px, pxx = terms_t[t]
        Qt = np.zeros((n+1, n+1))
        Qt[:-1, :-1] = pxx; Qt[:-1, -1] = px; Qt[-1, :-1] = px
        Qt[-1, -1] = 2.0*p0
        QT_list.append(_sym(Qt))

    # R_mat: constant across k for this separable cost (ell_uu = r always);
    # hop_horizon_search takes a single R_mat, matching lqr.py's convention.
    R_mat = R_list[0]
    return A_aug_list, B_aug_list, Q_aug_list, R_mat, z0, QT_list, stage_info


def regularize_to_pd(Q, margin=1e-6):
    """Add lambda*I until Q is positive definite (LM-style, per Notes.md item 3)."""
    min_eig = float(np.linalg.eigvalsh(_sym(Q)).min())
    if min_eig > margin:
        return Q, 0.0
    lam = margin - min_eig
    return _sym(Q) + lam*np.eye(Q.shape[0]), lam


def clip_vxx(Vxx, min_eig=-50.0):
    """State-space ('Vxx') regularization: clip only the eigenvalues of the
    value-Hessian that fall below min_eig, leaving the rest untouched. This is
    the DDP-standard alternative to Quu += lambda*I (Todorov & Li's iLQG uses
    both forms) -- it bounds the backward recursion directly instead of only
    damping the control-gain computation, which is necessary here because an
    indefinite ell_xx makes the *value function itself* (not just Quu)
    unbounded below once the recursion runs long enough to exploit it."""
    w_eig, V = np.linalg.eigh(_sym(Vxx))
    w_clipped = np.clip(w_eig, min_eig, None)
    return _sym(V @ np.diag(w_clipped) @ V.T)


def bruteforce_true_cost(A_list, B_list, X, U, costs_k, terms_t, w, T_max, vxx_min_eig=-50.0):
    """Backward Riccati-like value backup using the true cost's local
    quadratic model -- never inverts Q, only (regularized) Quu, so it can't
    hit the Assumption-2 problem HOP-LQR has. Same role as ddp.bruteforce_all_Jt
    but with time-varying (lx,lu,lxx,luu,lux) instead of constant (Q,R).

    Also applies Vxx state-space regularization (clip_vxx) after each stage:
    with an indefinite ell_xx, the raw recursion's value function is
    genuinely unbounded below over a long enough horizon (this is not a
    dynamics-stability issue -- Quu-only regularization doesn't touch it)."""
    n, m = X.shape[1], U.shape[1]
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
            V0 = l0 + w + V0 - 0.5*float(Qu @ invQu)
        J[T-1] = V0
    return J


# ---------------------------------------------------------------------------
# One outer-iteration trial: build true-cost Q_aug on a nominal trajectory,
# log diagnostics, run the raw-vs-regularized ablation, and compare against
# the true-cost brute-force baseline.
# ---------------------------------------------------------------------------

def run_trial(F, x0, xg, u_ref, ell, phi, w, N, T_min, T_max, max_iter=15,
              in_window_lo=120.0, in_window_hi=240.0, verbose=False,
              flick_amp=1.0, flick_steps=5):
    n, m = x0.size, u_ref.size
    # A repeated u_ref nominal is degenerate here: x0 sits exactly at an
    # unstable fixed point of this dynamics, so the all-zero rollout never
    # moves and every stage shares an identical (and identically indefinite)
    # Q_aug. A small one-shot "flick" breaks that fixed point and gives a
    # genuinely time-varying nominal (mix of PD and indefinite stages).
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    if flick_amp and flick_steps:
        U[:flick_steps, 0] += flick_amp
    X = rollout(F, x0, U)

    history = []
    for it in range(max_iter + 1):
        A_list, B_list = linearize_trajectory(F, X, U)
        costs_k = [cost_derivatives(ell, X[k], U[k]) for k in range(N)]
        terms_t = [terminal_derivatives(phi, X[t]) for t in range(N + 1)]

        A_aug, B_aug, Q_aug, R_mat, z0, QT_list, stage_info = \
            build_augmented_system_true_cost(F, A_list, B_list, X, U, costs_k, terms_t, w)

        any_indefinite = any(not s["is_pd"] for s in stage_info)
        any_in_window = any(in_window_lo < s["theta_deg"] < in_window_hi for s in stage_info)
        indefinite_in_window = any(
            (not s["is_pd"]) and (in_window_lo < s["theta_deg"] < in_window_hi)
            for s in stage_info)
        indefinite_outside_window = any(
            (not s["is_pd"]) and not (in_window_lo < s["theta_deg"] < in_window_hi)
            for s in stage_info)

        def safe_argmin_T(J):
            J = np.where(np.isfinite(J), J, np.inf)
            return int(np.argmin(J[T_min-1:T_max]) + T_min)

        with np.errstate(over="ignore", invalid="ignore"):
            J_raw = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
            T_raw = safe_argmin_T(J_raw)

            Q_aug_reg = [regularize_to_pd(Qk)[0] for Qk in Q_aug]
            QT_reg = [regularize_to_pd(Qt)[0] for Qt in QT_list]
            J_reg = hop_horizon_search(A_aug, B_aug, Q_aug_reg, R_mat, z0, QT_reg, T_max)
            T_reg = safe_argmin_T(J_reg)

            J_bf = bruteforce_true_cost(A_list, B_list, X, U, costs_k, terms_t, w, T_max)
            T_bf = safe_argmin_T(J_bf)

        record = {
            "iter": it, "any_indefinite": any_indefinite,
            "any_in_window": any_in_window,
            "indefinite_in_window": indefinite_in_window,
            "indefinite_outside_window": indefinite_outside_window,
            "min_eig_overall": min(s["min_eig_Qaug"] for s in stage_info),
            "T_raw": T_raw, "T_reg": T_reg, "T_bf": T_bf,
            "raw_vs_reg_disagree": T_raw != T_reg,
            "raw_vs_bf_disagree": T_raw != T_bf,
            "reg_vs_bf_disagree": T_reg != T_bf,
        }
        history.append(record)
        if verbose:
            print(f"  it={it:2d} indef={any_indefinite!s:5} in_window={any_in_window!s:5} "
                  f"T_raw={T_raw:3d} T_reg={T_reg:3d} T_bf={T_bf:3d} "
                  f"min_eig={record['min_eig_overall']:.4g}")

        # Step 3+4: truncated iLQR backward pass + forward rollout at T_raw,
        # using the *true* cost's local quadratic model (as Algorithm 2 does).
        # Per the paper's Sec. V-B: if the forward pass finds no descent step,
        # increase the LM regularization parameter and retry the backward pass.
        acc = False
        with np.errstate(over="ignore", invalid="ignore"):
            for lm in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
                k_list, K_list, ok = ilqr_backward_pass_true_cost(A_list, B_list, costs_k, terms_t, T_raw, lm=lm)
                if not ok or not np.all(np.isfinite(k_list[0])) or any(not np.all(np.isfinite(Kk)) for Kk in K_list):
                    continue
                Xn, Un, Jn, acc = forward_pass_true_cost(F, X, U, ell, phi, w, T_raw, k_list, K_list)
                if acc:
                    break
        if not acc:
            break
        X, U = Xn, Un

    final_err = float(np.linalg.norm(X[history[-1]["T_raw"]] - xg)) if history else float("nan")
    success = final_err <= 0.5  # paper's own success threshold (Sec. VI)
    return {"history": history, "final_err": final_err, "success": success, "X": X, "U": U}


def ilqr_backward_pass_true_cost(A_list, B_list, costs_k, terms_t, T_star, lm=1e-3, vxx_min_eig=-50.0):
    n, m = A_list[0].shape[0], B_list[0].shape[1]
    p0, px, pxx = terms_t[T_star]
    Vx, Vxx = px, clip_vxx(pxx, vxx_min_eig)
    k_list, K_list = [None]*T_star, [None]*T_star
    for k in reversed(range(T_star)):
        l0, lx, lu, lxx, luu, lxu = costs_k[k]
        lux = lxu.T
        A, B = A_list[k], B_list[k]
        Qx = lx + A.T @ Vx; Qu = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx @ A; Quu = luu + B.T @ Vxx @ B; Qux = lux + B.T @ Vxx @ A
        Quu_reg, _ = regularize_to_pd(Quu, margin=lm)
        kk = -chol_solve(Quu_reg, Qu); Kk = -chol_solve(Quu_reg, Qux)
        k_list[k], K_list[k] = kk, Kk
        Vx = Qx + Kk.T @ Qu + Qux.T @ kk + Kk.T @ Quu @ kk
        Vxx = clip_vxx(_sym(Qxx + Kk.T @ Qux + Qux.T @ Kk + Kk.T @ Quu @ Kk), vxx_min_eig)
    return k_list, K_list, True


def cost_true_nonlinear(X, U, ell, phi, w, T_star):
    c = sum(ell(X[k], U[k]) + w for k in range(T_star))
    return c + phi(X[T_star])


def forward_pass_true_cost(F, X, U, ell, phi, w, T_star, k_list, K_list):
    N_steps = len(U)
    J_old = cost_true_nonlinear(X, U, ell, phi, w, T_star)
    for a in [1.0, 0.5, 0.25, 0.1, 0.05]:
        X_new = np.zeros_like(X); X_new[0] = X[0]
        U_new = U.copy(); ok = True
        for k in range(T_star):
            dx = X_new[k] - X[k]
            U_new[k] = U[k] + K_list[k] @ dx + a*k_list[k]
            X_new[k+1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k+1])):
                ok = False; break
        if not ok:
            continue
        for k in range(T_star, N_steps):
            X_new[k+1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k+1])):
                ok = False; break
        if not ok:
            continue
        J_new = cost_true_nonlinear(X_new, U_new, ell, phi, w, T_star)
        if J_new < J_old:
            return X_new, U_new, J_new, True
    return X, U, J_old, False


# ---------------------------------------------------------------------------
# Plot: J_raw / J_reg / J_bf over the full horizon range for one trial, for
# visual comparison against the paper's Fig. 5 (quadrotor) shape.
# ---------------------------------------------------------------------------

def plot_trial_J_curves(F, x0, ell, phi, w, N, T_max, out_path,
                         flick_amp=1.0, flick_steps=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    U = np.tile(np.zeros(1).reshape(1, -1), (N, 1))
    U[:flick_steps, 0] += flick_amp
    X = rollout(F, x0, U)

    A_list, B_list = linearize_trajectory(F, X, U)
    costs_k = [cost_derivatives(ell, X[k], U[k]) for k in range(N)]
    terms_t = [terminal_derivatives(phi, X[t]) for t in range(N + 1)]
    A_aug, B_aug, Q_aug, R_mat, z0, QT_list, stage_info = \
        build_augmented_system_true_cost(F, A_list, B_list, X, U, costs_k, terms_t, w)
    Q_aug_reg = [regularize_to_pd(Qk)[0] for Qk in Q_aug]
    QT_reg = [regularize_to_pd(Qt)[0] for Qt in QT_list]

    with np.errstate(over="ignore", invalid="ignore"):
        J_raw = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
        J_reg = hop_horizon_search(A_aug, B_aug, Q_aug_reg, R_mat, z0, QT_reg, T_max)
        J_bf = bruteforce_true_cost(A_list, B_list, X, U, costs_k, terms_t, w, T_max)

    Tax = np.arange(1, T_max + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Tax, J_raw, "#E91E63", lw=2, label="J_raw (published HOP-LQR)")
    ax.plot(Tax, J_reg, "#2196F3", lw=2, label="J_reg (Q_aug regularized per-stage)")
    ax.plot(Tax, J_bf, "#4CAF50", lw=2.5, label="J_bf (Vxx-clipped brute-force Riccati)")
    for J, c in [(J_raw, "#E91E63"), (J_reg, "#2196F3"), (J_bf, "#4CAF50")]:
        Tstar = int(np.argmin(np.where(np.isfinite(J), J, np.inf))) + 1
        ax.scatter([Tstar], [J[Tstar - 1]], color=c, s=90, zorder=5, edgecolor="white")
        ax.annotate(f"T*={Tstar}", (Tstar, J[Tstar - 1]), textcoords="offset points",
                    xytext=(8, 8), color=c, fontsize=10)
    ax.set(xlabel="Horizon T", ylabel="J(T)",
           title="Cartpole true-cost J(T): raw vs regularized vs brute-force")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep: vary w and secondary parameters (not x0/xg -- Notes.md's "engineer
# it directly" argument says the (120,240) window is hit on essentially
# every swing-up by construction, so randomizing theta0 adds no signal).
# ---------------------------------------------------------------------------

def main():
    F, x0, xg, u_ref, *_ = make_cartpole()
    N, T_min, T_max = 120, 1, 60

    rng = np.random.default_rng(0)
    trials = []
    for w in [0.005, 0.01, 0.02, 0.05, 0.1]:
        for trial_idx in range(4):
            q_theta = float(rng.uniform(5.0, 20.0))
            r = float(rng.uniform(5e-3, 5e-2))
            ell, phi = make_swingup_cost(q_theta=q_theta, r=r)
            trials.append((w, q_theta, r, ell, phi))

    print(f"{'w':>7} {'q_theta':>8} {'r':>7} {'it':>3} {'min_eig':>9} {'in_win':>7} "
          f"{'T_raw':>6} {'T_reg':>6} {'T_bf':>5} {'raw!=reg':>9} {'raw!=bf':>8} {'reg!=bf':>8}")
    n_indef = n_indef_in_window = n_raw_reg_disagree = n_raw_bf_disagree = n_reg_bf_disagree = n_success = 0
    for w, q_theta, r, ell, phi in trials:
        res = run_trial(F, x0, xg, u_ref, ell, phi, w, N, T_min, T_max, max_iter=3)
        any_indef = any(h["any_indefinite"] for h in res["history"])
        any_indef_in_win = any(h["indefinite_in_window"] for h in res["history"])
        any_raw_reg_dis = any(h["raw_vs_reg_disagree"] for h in res["history"])
        any_raw_bf_dis = any(h["raw_vs_bf_disagree"] for h in res["history"])
        any_reg_bf_dis = any(h["reg_vs_bf_disagree"] for h in res["history"])
        n_indef += any_indef; n_indef_in_window += any_indef_in_win
        n_raw_reg_disagree += any_raw_reg_dis; n_raw_bf_disagree += any_raw_bf_dis
        n_reg_bf_disagree += any_reg_bf_dis
        n_success += res["success"]
        for h in res["history"]:
            print(f"{w:7.3f} {q_theta:8.2f} {r:7.4f} {h['iter']:3d} {h['min_eig_overall']:9.3f} "
                  f"{h['any_in_window']!s:>7} {h['T_raw']:6d} {h['T_reg']:6d} {h['T_bf']:5d} "
                  f"{h['raw_vs_reg_disagree']!s:>9} {h['raw_vs_bf_disagree']!s:>8} {h['reg_vs_bf_disagree']!s:>8}")

    n = len(trials)
    print(f"\n{'-'*70}")
    print(f"Trials: {n}")
    print(f"Q_aug indefinite at some (iter,k):            {n_indef}/{n}")
    print(f"...specifically inside (120,240) deg window:  {n_indef_in_window}/{n}")
    print(f"raw HOP vs true-cost brute-force T* disagree: {n_raw_bf_disagree}/{n}")
    print(f"per-stage-regularized vs brute-force disagree:{n_reg_bf_disagree}/{n}  (per-stage reg is NOT a reliable ground truth)")
    print(f"raw vs per-stage-regularized T* disagree:     {n_raw_reg_disagree}/{n}")
    print(f"Success (||x_Tstar - xg|| <= 0.5):             {n_success}/{n}")

    out_path = "outputs/cartpole_indefiniteness_J_curves.png"
    plot_trial_J_curves(F, x0, *make_swingup_cost(q_theta=13.15, r=0.0471),
                         w=0.01, N=N, T_max=T_max, out_path=out_path)
    print(f"\nSaved representative J(T) comparison plot to {out_path}")


if __name__ == "__main__":
    main()
