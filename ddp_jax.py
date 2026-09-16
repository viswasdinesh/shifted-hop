#!/usr/bin/env python3
"""JAX-accelerated linearization and cost augmentation for HOP-DDP.

ddp.linearize_trajectory and lqr.build_augmented_system both loop over the
N trajectory stages in a plain Python for-loop, and each stage's work
(finite-difference Jacobian, augmented-matrix assembly) is independent of
every other stage -- nothing carries state across k. That independence is
exactly what jax.vmap parallelizes: instead of N sequential Python-level
calls, one vmapped, JIT-compiled call computes all N stages at once, and
the Jacobians are exact (via autodiff) rather than the finite-difference
approximation ddp.linearize_trajectory uses.

hop_horizon_search (lqr.py) is deliberately NOT reimplemented here: its
forward composition (Ebar/Fbar/Gbar) is a genuine sequential recursion
across k -- each step's composed map depends on the previous one -- so it
does not vmap the same way. (It could be sped up with jax.lax.scan, which
is a different kind of win -- fused/compiled sequential iteration, not
parallelization -- and is out of scope here.)

This file is additive: it does not modify ddp.py, lqr.py, or systems.py.
The __main__ demo below cross-checks its output against the existing
numpy/finite-difference path for both correctness and speed, on the
quadrotor (nonlinear, n=12 -- the case where this actually matters) and
the double integrator (linear, as a sanity baseline).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # match numpy's float64 throughout the rest of the repo


# ---------------------------------------------------------------------------
# JAX-compatible dynamics, mirroring systems.py but written with jax.numpy
# (no data-dependent Python `if` on traced values, which is what makes them
# traceable through jax.jacobian/vmap/jit in the first place).
# ---------------------------------------------------------------------------

def make_double_integrator_jax(dt=0.05, N=120):
    def F(x, u):
        return jnp.array([x[0] + dt * x[1], x[1] + dt * u[0]])

    x0    = jnp.array([1.0, 0.0])
    xg    = jnp.array([2.0, 0.0])
    u_ref = jnp.array([0.0])
    Q     = jnp.diag(jnp.array([1.0, 0.1]))
    R     = jnp.array([[1e-2]])
    alpha = 50.0
    w     = 0.02
    T_min, T_max = 10, 80
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, None


def make_quadrotor_jax(dt=0.05, N=160):
    mass, g = 1.0, 9.81
    Ix, Iy, Iz = 0.02, 0.02, 0.04
    kv, kw = 0.05, 0.01
    I_body = jnp.diag(jnp.array([Ix, Iy, Iz]))
    I_inv  = jnp.diag(jnp.array([1 / Ix, 1 / Iy, 1 / Iz]))

    def rotm(phi, th, psi):
        s, c = jnp.sin, jnp.cos
        Rz = jnp.array([[c(psi), -s(psi), 0.], [s(psi), c(psi), 0.], [0., 0., 1.]])
        Ry = jnp.array([[c(th), 0., s(th)], [0., 1., 0.], [-s(th), 0., c(th)]])
        Rx = jnp.array([[1., 0., 0.], [0., c(phi), -s(phi)], [0., s(phi), c(phi)]])
        return Rz @ Ry @ Rx

    def Tmat(phi, th):
        s, c, t = jnp.sin, jnp.cos, jnp.tan
        return jnp.array([[1., s(phi) * t(th), c(phi) * t(th)],
                           [0., c(phi), -s(phi)],
                           [0., s(phi) / c(th), c(phi) / c(th)]])

    def F(x, u):
        phi, th, psi = x[6], x[7], x[8]
        vel, omg = x[3:6], x[9:12]
        Rb = rotm(phi, th, psi)
        e3 = jnp.array([0., 0., 1.])
        acc = Rb @ (e3 * u[0]) / mass - jnp.array([0., 0., g]) - kv * vel
        eulerdot = Tmat(phi, th) @ omg
        omgdot = I_inv @ (u[1:4] - jnp.cross(omg, I_body @ omg)) - kw * omg
        xdot = jnp.concatenate([vel, acc, eulerdot, omgdot])
        return x + dt * xdot

    x0 = jnp.zeros(12).at[:3].set(jnp.array([2., 2., 2.]))
    xg = jnp.zeros(12)
    u_ref = jnp.array([mass * g, 0., 0., 0.])
    Q_q = jnp.diag(jnp.array([5., 5., 5., 1., 1., 1., 20., 20., 10., 1., 1., 1.]))
    R_q = jnp.diag(jnp.array([1e-3, 1e-2, 1e-2, 1e-2]))
    return F, x0, xg, u_ref, Q_q, R_q, 300.0, 0.05, N, 20, 160, [6, 7, 8]


# ---------------------------------------------------------------------------
# Parallel linearization: exact autodiff Jacobians, vmapped over all N
# stages -- replaces ddp.linearize_trajectory's Python loop of central
# finite differences.
# ---------------------------------------------------------------------------

def make_linearize_fn(F):
    """Returns a jitted fn (X, U) -> (A, B) with A: (N,n,n), B: (N,n,m),
    computed for every stage k in one vmapped, XLA-compiled call."""
    jac = jax.jacfwd(F, argnums=(0, 1))

    @jax.jit
    def linearize_trajectory_jax(X, U):
        N = U.shape[0]
        return jax.vmap(jac)(X[:N], U)

    return linearize_trajectory_jax


# ---------------------------------------------------------------------------
# Parallel augmentation: lqr.build_augmented_system's per-stage math,
# vmapped over k instead of looped in Python.
# ---------------------------------------------------------------------------

def make_build_augmented_system_fn(F, wrap_idx=None):
    def _wrap(e):
        if not wrap_idx:
            return e
        for i in wrap_idx:
            e = e.at[i].set((e[i] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return e

    def _stage(A_k, B_k, x_k, x_kp1, u_k, xg, u_ref, Q, w, n, m):
        a_k = F(x_k, u_k) - x_kp1
        du = u_k - u_ref
        a_tilde = a_k - B_k @ du

        Ak_aug = jnp.zeros((n + 1, n + 1)).at[:n, :n].set(A_k).at[:n, n].set(a_tilde).at[n, n].set(1.0)
        Bk_aug = jnp.zeros((n + 1, m)).at[:n, :].set(B_k)

        e = _wrap(x_k - xg)
        Qe = Q @ e
        Qk = jnp.zeros((n + 1, n + 1))
        Qk = Qk.at[:n, :n].set(0.5 * (Q + Q.T) + 1e-9 * jnp.eye(n))
        Qk = Qk.at[:n, n].set(Qe).at[n, :n].set(Qe)
        Qk = Qk.at[n, n].set(e @ Q @ e + 2.0 * w + 1e-12)
        return Ak_aug, Bk_aug, 0.5 * (Qk + Qk.T)

    stage_v = jax.vmap(_stage, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None))

    def _terminal(x_t, P, xg, n):
        e = _wrap(x_t - xg)
        px = P @ e
        p0 = 0.5 * (e @ P @ e)
        Qt = jnp.zeros((n + 1, n + 1))
        Qt = Qt.at[:-1, :-1].set(P).at[:-1, -1].set(px).at[-1, :-1].set(px)
        Qt = Qt.at[-1, -1].set(2.0 * p0 + 1e-12)
        return 0.5 * (Qt + Qt.T)

    terminal_v = jax.vmap(_terminal, in_axes=(0, None, None, None))

    @jax.jit
    def build_augmented_system_jax(A, B, X, U, xg, u_ref, Q, R, w, alpha):
        n, m = X.shape[1], U.shape[1]
        Qf = alpha * jnp.eye(n) if jnp.ndim(alpha) == 0 else jnp.diag(alpha)
        A_aug, B_aug, Q_aug = stage_v(A, B, X[:-1], X[1:], U, xg, u_ref, Q, w, n, m)
        QT = terminal_v(X[1:], Qf, xg, n)
        R_mat = 0.5 * (R + R.T)
        z0 = jnp.zeros(n + 1).at[-1].set(1.0)
        return A_aug, B_aug, Q_aug, R_mat, z0, QT

    return build_augmented_system_jax


# ---------------------------------------------------------------------------
# Demo / cross-check against the existing numpy path.
# ---------------------------------------------------------------------------

def _demo_system(name, make_jax_fn, make_np_fn):
    from ddp import linearize_trajectory
    from lqr import build_augmented_system, hop_horizon_search
    from utils import rollout

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    F_np, x0_np, xg_np, uref_np, Q_np, R_np, alpha, w, N, T_min, T_max, *rest_np = make_np_fn()
    wrap_idx = rest_np[0] if rest_np else None
    U_np = np.tile(uref_np.reshape(1, -1), (N, 1))
    X_np = rollout(F_np, x0_np, U_np)

    F_jx, x0_jx, xg_jx, uref_jx, Q_jx, R_jx, *_ = make_jax_fn()
    X_jx, U_jx = jnp.asarray(X_np), jnp.asarray(U_np)

    linearize_jax = make_linearize_fn(F_jx)
    build_aug_jax = make_build_augmented_system_fn(F_jx, wrap_idx)

    # Correctness: compare against the numpy/finite-difference path.
    A_list, B_list = linearize_trajectory(F_np, X_np, U_np)
    A_np, B_np = np.stack(A_list), np.stack(B_list)

    A_jx, B_jx = linearize_jax(X_jx, U_jx)   # first call: includes JIT trace+compile
    A_jx, B_jx = linearize_jax(X_jx, U_jx)   # warm call: compiled
    print(f"max|A_np - A_jax| = {np.max(np.abs(A_np - np.asarray(A_jx))):.2e}  "
          f"(finite-diff vs exact autodiff -- small nonzero gap is expected)")
    print(f"max|B_np - B_jax| = {np.max(np.abs(B_np - np.asarray(B_jx))):.2e}")

    A_aug_np, B_aug_np, Q_aug_np, R_mat_np, z0_np, QT_np = build_augmented_system(
        F_np, A_list, B_list, X_np, U_np, xg_np, uref_np, Q_np, R_np, w, alpha, wrap_idx)

    A_aug_jx, B_aug_jx, Q_aug_jx, R_mat_jx, z0_jx, QT_jx = build_aug_jax(
        A_jx, B_jx, X_jx, U_jx, xg_jx, uref_jx, Q_jx, R_jx, w, alpha)
    A_aug_jx, B_aug_jx, Q_aug_jx, R_mat_jx, z0_jx, QT_jx = build_aug_jax(
        A_jx, B_jx, X_jx, U_jx, xg_jx, uref_jx, Q_jx, R_jx, w, alpha)

    print(f"max|Q_aug_np - Q_aug_jax| = {np.max(np.abs(np.stack(Q_aug_np) - np.asarray(Q_aug_jx))):.2e}")

    # Feed the JAX-computed matrices straight into the existing (numpy) HOP-LQR
    # solver, unmodified, to confirm the two front-ends are interchangeable.
    J_np = hop_horizon_search(A_aug_np, B_aug_np, Q_aug_np, R_mat_np, z0_np, QT_np, T_max)
    J_jx = hop_horizon_search(list(np.asarray(A_aug_jx)), list(np.asarray(B_aug_jx)),
                               list(np.asarray(Q_aug_jx)), np.asarray(R_mat_jx),
                               np.asarray(z0_jx), list(np.asarray(QT_jx)), T_max)
    T_np = int(np.argmin(J_np[T_min - 1:T_max]) + T_min)
    T_jx = int(np.argmin(J_jx[T_min - 1:T_max]) + T_min)
    print(f"T* from numpy-built inputs: {T_np}   T* from JAX-built inputs: {T_jx}   "
          f"match: {T_np == T_jx}")

    # Timing: numpy loop vs JAX (already-compiled) vmapped call.
    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        A_list, B_list = linearize_trajectory(F_np, X_np, U_np)
    t_np_lin = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        A_jx, B_jx = linearize_jax(X_jx, U_jx)
        jax.block_until_ready(A_jx)
    t_jx_lin = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        build_augmented_system(F_np, A_list, B_list, X_np, U_np, xg_np, uref_np, Q_np, R_np, w, alpha, wrap_idx)
    t_np_aug = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        out = build_aug_jax(A_jx, B_jx, X_jx, U_jx, xg_jx, uref_jx, Q_jx, R_jx, w, alpha)
        jax.block_until_ready(out[2])
    t_jx_aug = (time.perf_counter() - t0) / n_rep

    print(f"\nlinearize_trajectory:      numpy {t_np_lin*1e3:7.3f} ms   "
          f"jax(vmap+jit) {t_jx_lin*1e3:7.3f} ms   speedup {t_np_lin/t_jx_lin:5.1f}x")
    print(f"build_augmented_system:    numpy {t_np_aug*1e3:7.3f} ms   "
          f"jax(vmap+jit) {t_jx_aug*1e3:7.3f} ms   speedup {t_np_aug/t_jx_aug:5.1f}x")


def main():
    _demo_system("Double Integrator (n=2, N=120)", make_double_integrator_jax,
                 __import__("systems").make_double_integrator)
    _demo_system("Quadrotor (n=12, N=160)", make_quadrotor_jax,
                 __import__("systems").make_quadrotor)


if __name__ == "__main__":
    main()
