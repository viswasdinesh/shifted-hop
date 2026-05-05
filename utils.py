"""Utility functions extracted from HOP_colab_notebook.ipynb."""

import numpy as np

def _sym(A):
    return 0.5 * (A + A.T)

def chol_inv(A, jitter=1e-9):
    A = _sym(np.asarray(A, dtype=float))
    n = A.shape[0]; I = np.eye(n); eps = jitter
    for _ in range(8):
        try:
            L = np.linalg.cholesky(A + eps * I)
            return np.linalg.solve(L.T, np.linalg.solve(L, I))
        except np.linalg.LinAlgError:
            eps *= 10.0
    return np.linalg.solve(A + eps * I, I)

def chol_solve(A, B, jitter=1e-9):
    A = _sym(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    n = A.shape[0]; I = np.eye(n); eps = jitter
    for _ in range(8):
        try:
            L = np.linalg.cholesky(A + eps * I)
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            eps *= 10.0
    raise np.linalg.LinAlgError("chol_solve failed")

def rollout(F, x0, U):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1: U = U.reshape(-1, 1)
    N_steps, n = U.shape[0], x0.size
    X = np.zeros((N_steps + 1, n)); X[0] = x0
    for k in range(N_steps):
        X[k + 1] = F(X[k], U[k])
    return X

def as_terminal_weight(alpha, n):
    A = np.asarray(alpha, dtype=float)
    if A.ndim == 0: return float(A) * np.eye(n)
    if A.ndim == 1: return np.diag(A)
    return _sym(A)

def _wrap(e, wrap_idx):
    if not wrap_idx: return e
    e = np.asarray(e, dtype=float).copy()
    for i in wrap_idx:
        e[i] = (e[i] + np.pi) % (2*np.pi) - np.pi
    return e
