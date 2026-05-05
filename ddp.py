"""HOP-DDP functions extracted from HOP_colab_notebook.ipynb."""

import time

import numpy as np

from lqr import build_augmented_system, hop_horizon_search
from utils import _sym, _wrap, as_terminal_weight, chol_solve, rollout

def linearize_trajectory(F, X, U, eps=1e-5):
    N_steps, n, m = len(U), X.shape[1], U.shape[1]
    A_list, B_list = [], []
    for k in range(N_steps):
        x, u = X[k].copy(), U[k].copy()
        A, B = np.zeros((n, n)), np.zeros((n, m))
        for i in range(n):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            A[:, i] = (F(xp, u) - F(xm, u)) / (2 * eps)
        for j in range(m):
            up, um = u.copy(), u.copy()
            up[j] += eps; um[j] -= eps
            B[:, j] = (F(x, up) - F(x, um)) / (2 * eps)
        A_list.append(A); B_list.append(B)
    return A_list, B_list


def iLQR_backward_pass(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star,
                        lm=1e-3, wrap_idx=None):
    n, m = X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)
    eT = _wrap(X[T_star] - xg, wrap_idx)
    Vx, Vxx = Qf @ eT, _sym(Qf)
    k_list, K_list = [None]*T_star, [None]*T_star

    for k in reversed(range(T_star)):
        e, du = _wrap(X[k]-xg, wrap_idx), U[k]-u_ref
        A, B = A_list[k], B_list[k]
        Qx  = Q@e + A.T@Vx;       Qu  = R@du + B.T@Vx
        Qxx = Q + A.T@Vxx@A;      Quu = R + B.T@Vxx@B;  Qux = B.T@Vxx@A
        Quu_reg = _sym(Quu) + lm*np.eye(m)
        try: np.linalg.cholesky(Quu_reg)
        except np.linalg.LinAlgError: return None, None, False
        kk = -chol_solve(Quu_reg, Qu);  Kk = -chol_solve(Quu_reg, Qux)
        k_list[k], K_list[k] = kk, Kk
        Vx  = Qx + Kk.T@Qu + Qux.T@kk + Kk.T@Quu@kk
        Vxx = _sym(Qxx + Kk.T@Qux + Qux.T@Kk + Kk.T@Quu@Kk)
    return k_list, K_list, True


def cost_true(X, U, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx=None):
    n = X.shape[1]; Qf = as_terminal_weight(alpha, n)
    c = 0.0
    for k in range(T_star):
        e, du = _wrap(X[k]-xg, wrap_idx), U[k]-u_ref
        c += 0.5*e@Q@e + 0.5*du@R@du + w
    eT = _wrap(X[T_star]-xg, wrap_idx)
    return c + 0.5*eT@Qf@eT


def forward_pass(F, X, U, xg, u_ref, Q, R, alpha, w, T_star,
                  k_list, K_list, wrap_idx=None):
    N_steps = len(U)
    J_old = cost_true(X, U, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx)
    for a in [1.0, 0.5, 0.25, 0.1, 0.05]:
        X_new = np.zeros_like(X); X_new[0] = X[0]
        U_new = U.copy(); ok = True
        for k in range(T_star):
            dx = _wrap(X_new[k]-X[k], wrap_idx)
            U_new[k] = U[k] + K_list[k]@dx + a*k_list[k]
            X_new[k+1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k+1])): ok = False; break
        if not ok: continue
        for k in range(T_star, N_steps):
            X_new[k+1] = F(X_new[k], U_new[k])
            if not np.all(np.isfinite(X_new[k+1])): ok = False; break
        if not ok: continue
        J_new = cost_true(X_new, U_new, xg, u_ref, Q, R, alpha, w, T_star, wrap_idx)
        if J_new < J_old: return X_new, U_new, J_new, True
    return X, U, J_old, False


def bruteforce_all_Jt(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max,
                      wrap_idx=None):
    # Value function expansion: tracks (Vxx, Vx, V0) to handle affine terms. O(N^2 n^3)
    n, m = X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)
    J = np.zeros(T_max)
    for T in range(1, T_max + 1):
        Vxx = [np.zeros((n,n)) for _ in range(T+1)]
        Vx  = [np.zeros(n) for _ in range(T+1)]
        V0  = [0.0 for _ in range(T+1)]
        eT = _wrap(X[T]-xg, wrap_idx)
        Vxx[T] = _sym(Qf); Vx[T] = Qf@eT; V0[T] = 0.5*(eT@Qf@eT).item()
        for t in reversed(range(T)):
            e, du = _wrap(X[t]-xg, wrap_idx), U[t]-u_ref
            lx, lu = Q@e, R@du
            l0 = 0.5*(e@Q@e).item() + 0.5*(du@R@du).item() + float(w)
            A, B = A_list[t], B_list[t]
            Qx = lx+A.T@Vx[t+1]; Qu = lu+B.T@Vx[t+1]
            Qxx = Q+A.T@Vxx[t+1]@A; Quu = R+B.T@Vxx[t+1]@B; Qux = B.T@Vxx[t+1]@A
            Quu_reg = _sym(Quu) + 1e-6*np.eye(m)
            invQu = chol_solve(Quu_reg, Qu); invQux = chol_solve(Quu_reg, Qux)
            Vxx[t] = _sym(Qxx - Qux.T@invQux)
            Vx[t]  = Qx - Qux.T@invQu
            V0[t]  = l0 + V0[t+1] - 0.5*(Qu@invQu).item()
        J[T-1] = float(V0[0])
    return J


def solve_hop(F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max,
              method="hop", max_iter=15, wrap_idx=None):
    n, m = x0.size, u_ref.size
    U = np.tile(u_ref.reshape(1, -1), (N, 1))
    X = rollout(F, x0, U)

    timers = {"linearize": 0.0, "select": 0.0, "backward": 0.0, "forward": 0.0}
    J_hist, T_hist = [], []
    J_curve_out = None

    def _do_select(A_list, B_list, X, U):
        if method == "hop":
            A_aug, B_aug, Q_aug, R_mat, z0, QT_list = build_augmented_system(
                F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx)
            return hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
        else:
            return bruteforce_all_Jt(A_list, B_list, X, U, xg, u_ref, Q, R, alpha, w, T_max, wrap_idx)

    for it in range(max_iter + 1):
        t0 = time.perf_counter()
        A_list, B_list = linearize_trajectory(F, X, U)
        timers["linearize"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        J_curve = _do_select(A_list, B_list, X, U)
        T_star = int(np.argmin(J_curve[T_min-1:T_max]) + T_min)
        J_curve_out = J_curve
        timers["select"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        k_list, K_list, ok = iLQR_backward_pass(
            A_list, B_list, X, U, xg, u_ref, Q, R, alpha, T_star, wrap_idx=wrap_idx)
        timers["backward"] += time.perf_counter() - t0

        if ok:
            t0 = time.perf_counter()
            Xn, Un, Jn, acc = forward_pass(
                F, X, U, xg, u_ref, Q, R, alpha, w, T_star, k_list, K_list, wrap_idx=wrap_idx)
            timers["forward"] += time.perf_counter() - t0
            if acc:
                X, U = Xn, Un
                J_hist.append(Jn)
                T_hist.append(T_star)

    return {"X": X, "U": U, "J_hist": J_hist, "T_hist": T_hist,
            "timers": timers, "J_curve": J_curve_out,
            "T_star": T_hist[-1] if T_hist else T_star}
