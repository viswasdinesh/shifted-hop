"""HOP-LQR functions extracted from HOP_colab_notebook.ipynb."""

import numpy as np

from utils import _sym, _wrap, as_terminal_weight, chol_inv

def bruteforce_horizon_search(A_list, B_list, Q, R, Qf, w, x0, T_max):
    n, m = x0.size, R.shape[0]
    J_curve = np.zeros(T_max)
    for T in range(1, T_max + 1):
        P = Qf.copy()
        for k in range(T - 1, -1, -1):
            A, B = A_list[k], B_list[k]
            S = R + B.T @ P @ B
            K = np.linalg.solve(S, B.T @ P @ A)
            P = _sym(Q + A.T @ P @ A - A.T @ P @ B @ K)
        J_curve[T - 1] = 0.5 * x0 @ P @ x0 + w * T
    return J_curve


def build_augmented_system(F, A_list, B_list, X, U, xg, u_ref, Q, R, w, alpha, wrap_idx=None,
                            eps_Q=1e-9, eps_scalar=1e-12):
    N_steps, n, m = len(A_list), X.shape[1], U.shape[1]
    Qf = as_terminal_weight(alpha, n)
    R_mat = _sym(R)

    A_aug_list, B_aug_list, Q_aug_list = [], [], []
    for k in range(N_steps):
        a_k = (F(X[k], U[k]) - X[k + 1]).reshape(-1, 1)
        du = (U[k] - u_ref).reshape(-1, 1)
        a_tilde = a_k - B_list[k] @ du

        Ak_aug = np.zeros((n+1, n+1))
        Ak_aug[:n, :n] = A_list[k]; Ak_aug[:n, n] = a_tilde.ravel(); Ak_aug[n, n] = 1.0
        Bk_aug = np.zeros((n+1, m)); Bk_aug[:n, :] = B_list[k]

        e = _wrap(X[k] - xg, wrap_idx).reshape(-1, 1)
        Qk = np.zeros((n+1, n+1))
        Qk[:n, :n] = _sym(Q) + eps_Q * np.eye(n)
        Qk[:n, n] = (Q @ e).ravel(); Qk[n, :n] = (Q @ e).ravel()
        Qk[n, n] = float((e.T @ Q @ e).item()) + 2.0 * w + eps_scalar

        A_aug_list.append(Ak_aug); B_aug_list.append(Bk_aug); Q_aug_list.append(_sym(Qk))

    z0 = np.zeros(n + 1); z0[-1] = 1.0

    QT_list = []; P = _sym(Qf)
    for t in range(1, N_steps + 1):
        e = _wrap(X[t] - xg, wrap_idx).reshape(-1, 1)
        px = P @ e; p0 = 0.5 * float((e.T @ P @ e).item())
        Qt = np.zeros((n+1, n+1))
        Qt[:-1, :-1] = P; Qt[:-1, -1] = px.ravel(); Qt[-1, :-1] = px.ravel()
        Qt[-1, -1] = 2.0 * p0 + eps_scalar
        QT_list.append(_sym(Qt))

    return A_aug_list, B_aug_list, Q_aug_list, R_mat, z0, QT_list


def hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max, jitter=1e-9):
    R_inv = chol_inv(R_mat, jitter=jitter)

    E_list, F_list, G_list = [], [], []
    for k in range(T_max):
        Ek = chol_inv(Q_aug[k], jitter=jitter)
        Fk = Ek @ A_aug[k].T
        Gk = A_aug[k] @ Ek @ A_aug[k].T + B_aug[k] @ R_inv @ B_aug[k].T
        E_list.append(Ek); F_list.append(Fk); G_list.append(_sym(Gk))

    # Phase 1: compose maps forward
    Ebar = [E_list[0].copy()]
    Fbar = [F_list[0].copy()]
    Gbar = [G_list[0].copy()]
    for k in range(1, T_max):
        W = chol_inv(E_list[k] + Gbar[-1], jitter=jitter)
        Ebar.append(_sym(Ebar[-1] - Fbar[-1] @ W @ Fbar[-1].T))
        Fbar.append(Fbar[-1] @ W @ F_list[k])
        Gbar.append(_sym(G_list[k] - F_list[k].T @ W @ F_list[k]))

    # Phase 2: query each horizon
    J = np.zeros(T_max)
    for t in range(1, T_max + 1):
        Xt = chol_inv(QT_list[t - 1], jitter=jitter)
        Wt = chol_inv(Xt + Gbar[t - 1], jitter=jitter)
        X0 = _sym(Ebar[t - 1] - Fbar[t - 1] @ Wt @ Fbar[t - 1].T)
        P0 = chol_inv(X0, jitter=jitter)
        J[t - 1] = 0.5 * float((z0 @ P0 @ z0).item())
    return J
