"""Validation pass on ltv_sweep.py's claims, plus a second-pass review fix.

Corrections to the original single-seed run (see THEOREM.md Sec. 5.4 for the
full write-up):

1. The "reference T* = 1 (PD case)" line ltv_sweep.py prints at the end is a
   red herring for row-by-row comparison: it calls make_Q() again *after*
   the main loop, so the RNG has moved on and it's an unrelated freshly-drawn
   PD matrix, not any row's actual ground truth. The correct per-row
   reference is that row's own riccati_reference() output. Comparing T*
   against the *right* reference changes the conclusion for row 9 (lam_min
   =-10): shifted's T*=6 is CORRECT (matches argmin(Jr)=6 exactly), not
   questionable as the original discussion worried.

2. check_reference_trustworthy originally cross-checked riccati_reference
   against shifted_horizon_search(c=1e4) and called that "exact per
   Theorem 1". That's wrong on two counts, caught on review: Theorem 1
   proves algebraic equivalence, not numerical exactness, and c=1e4 is
   empirically the WORST choice tested in Sec. 5.3 (error grows
   monotonically with c due to cancellation in P = Phat^-1 - cI). The 1.9e-6
   this produced was an artifact of that choice, not evidence of a problem.
   Fixed to use choose_shift() (near the accuracy floor, where Sec. 5.3 shows
   ~1e-12..1e-14). Also: this check reuses the method under test (the
   composed-LFT shifted chart) to validate the reference that judges it --
   a different code path from the per-horizon sweep, but not independent.
   dense_qp_reference() below closes that gap for small T with a genuinely
   separate computation (stack the controls, solve the dense QP's normal
   equations directly -- no Riccati recursion, no LFT composition at all).

Then: multi-seed sweep (default 50) reporting exact-T*-match rates AND
max-relative-J-error for HOP / reg+HOP / shifted against each instance's own
reference, per case.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from lqr import hop_horizon_search
from shifted_chart import _sym, choose_shift, shifted_horizon_search
from verify_cartpole import riccati_reference

CASES = [(1.0, 0), (1e-3, 0), (1e-6, 0), (0.0, 0), (0.0, 2), (0.0, 4),
         (-1e-3, 0), (-1.0, 0), (-10.0, 0)]


def make_Q(rng, n, lam_min, rank_def=0):
    ev = np.abs(rng.normal(size=n)) * 3 + 1.0
    ev[0] = lam_min
    for i in range(1, rank_def + 1):
        ev[i] = 0.0
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return _sym(U @ np.diag(ev) @ U.T)


def reg_then_hop(A, B, Q, R, z0, QT, T, eps0=1e-6):
    def regularize(Ms):
        out = []
        for M in Ms:
            mu, M2 = 0.0, _sym(M)
            while np.linalg.eigvalsh(M2 + mu * np.eye(M.shape[0]))[0] <= 1e-9:
                mu = eps0 if mu == 0 else mu * 10
            out.append(_sym(M) + mu * np.eye(M.shape[0]))
        return out
    return hop_horizon_search(A, B, regularize(Q), R, z0, regularize(QT), T)


def build_instance(rng, n, m, T):
    A = [np.eye(n) + 0.15 * rng.normal(size=(n, n)) for _ in range(T)]
    B = [rng.normal(size=(n, m)) for _ in range(T)]
    R = _sym(rng.normal(size=(m, m))); R = R @ R.T + np.eye(m)
    Qf = _sym(rng.normal(size=(n, n))); Qf = Qf @ Qf.T + 5 * np.eye(n)
    QT = [Qf] * T
    z0 = rng.normal(size=n)
    return A, B, R, QT, z0


def dense_qp_reference(A, B, Q, R, z0, QT, T):
    """Structurally independent ground truth: eliminate the states via
    forward substitution, stack U = [u_0;...;u_{T-1}], and solve the
    resulting dense QP's normal equations directly. No Riccati recursion,
    no LFT composition -- a different computational method entirely, so
    agreement with riccati_reference / shifted_horizon_search is not
    circular. O((Tm)^3), so only usable for small T."""
    n, m = z0.size, B[0].shape[1]
    # c_k := x_k with U=0 (pure drift); M_k := d x_k / d U, an (n, T*m) block matrix.
    c = [z0.copy()]
    M = [np.zeros((n, T * m))]
    for k in range(T):
        c.append(A[k] @ c[k])
        Mk1 = A[k] @ M[k]
        Mk1[:, k * m:(k + 1) * m] = B[k]
        M.append(Mk1)

    H = np.zeros((T * m, T * m))
    f = np.zeros(T * m)
    for k in range(T):
        H += M[k].T @ Q[k] @ M[k]
        f += M[k].T @ Q[k] @ c[k]
    H += M[T].T @ QT[T - 1] @ M[T]
    f += M[T].T @ QT[T - 1] @ c[T]
    H += np.kron(np.eye(T), _sym(R))

    U = np.linalg.solve(_sym(H), -f)
    Jval = 0.5 * float(U @ H @ U) + float(f @ U)
    Jval += 0.5 * sum(float(c[k] @ Q[k] @ c[k]) for k in range(T))
    Jval += 0.5 * float(c[T] @ QT[T - 1] @ c[T])
    return Jval


def check_reference_trustworthy(seed=7, n=6, m=3, T=60, T_small=6):
    """Two independent cross-checks of riccati_reference on the worst-case
    (most indefinite) row, at the c choice Sec. 5.3 actually recommends:
      (a) vs shifted_horizon_search(c=choose_shift(...)) -- different code
          path (composed LFT vs per-horizon backward sweep), near the
          accuracy floor rather than the worst-case c=1e4 used before.
      (b) vs dense_qp_reference at a small T -- fully independent method,
          closes the "uses the method under test" gap for (a)."""
    rng = np.random.default_rng(seed)
    A, B, R, QT, z0 = build_instance(rng, n, m, T)
    for lam, rd in CASES:
        Q = [make_Q(rng, n, lam, rd) for _ in range(T)]  # last case: lam_min=-10

    Jr = riccati_reference(A, B, Q, R, z0, QT, T)
    c = choose_shift(Q, QT, T)
    J_shift, c_used = shifted_horizon_search(A, B, Q, R, z0, QT, T, c=c)
    rel_shift = float(np.max(np.abs(Jr - J_shift) / np.maximum(np.abs(Jr), 1e-12)))

    J_qp = dense_qp_reference(A, B, Q, R, z0, QT, T_small)
    rel_qp = abs(Jr[T_small - 1] - J_qp) / max(abs(Jr[T_small - 1]), 1e-12)

    return rel_shift, c_used, rel_qp


def one_seed(seed, n=6, m=3, T=60):
    rng = np.random.default_rng(seed)
    A, B, R, QT, z0 = build_instance(rng, n, m, T)
    rows = []
    for lam, rd in CASES:
        Q = [make_Q(rng, n, lam, rd) for _ in range(T)]
        Jr = riccati_reference(A, B, Q, R, z0, QT, T)
        T_ref = int(np.argmin(Jr)) + 1
        result = {"lam": lam, "rd": rd, "T_ref": T_ref, "Jr": Jr}
        for name, fn in [("hop", lambda: hop_horizon_search(A, B, Q, R, z0, QT, T)),
                          ("reg", lambda: reg_then_hop(A, B, Q, R, z0, QT, T)),
                          ("shift", lambda: shifted_horizon_search(A, B, Q, R, z0, QT, T)[0])]:
            try:
                J = fn()
                result[f"T_{name}"] = int(np.argmin(J)) + 1
                result[f"relerr_{name}"] = float(np.max(np.abs(J - Jr) / np.maximum(np.abs(Jr), 1e-12)))
            except Exception:
                result[f"T_{name}"] = None
                result[f"relerr_{name}"] = float("nan")
        rows.append(result)
    return rows


def main():
    rel_shift, c_used, rel_qp = check_reference_trustworthy()
    print(f"Reference sanity check (lam_min=-10 case, T=60):")
    print(f"  vs shifted_horizon_search at c=choose_shift()={c_used:.4g} (near accuracy floor, "
          f"a different code path -- composed LFT vs per-horizon sweep): max rel diff = {rel_shift:.2e}")
    print(f"  vs dense_qp_reference at T=6 (fully independent: stacked-control dense QP, "
          f"no Riccati recursion at all): rel diff = {rel_qp:.2e}")
    print("  Both small -> the reference is not corrupted by the Quu-near-singularity "
          "pathology found in the cartpole DDP test (that toy's mild A_k=I+0.15N(0,1) "
          "never drives Quu near-singular the way a fast-swinging pendulum linearization does).\n")
    assert rel_shift < 1e-8 and rel_qp < 1e-8, "reference disagrees with two independent computations -- do not trust it"

    n_seeds = 50
    print(f"Multi-seed sweep: {n_seeds} seeds, n=6 m=3 T=60\n")
    tally = {c: {"hop": 0, "reg": 0, "shift": 0, "n": 0,
                 "hop_err": [], "reg_err": [], "shift_err": []} for c in CASES}
    for seed in range(n_seeds):
        for row in one_seed(seed):
            key = (row["lam"], row["rd"])
            tally[key]["n"] += 1
            for name in ("hop", "reg", "shift"):
                if row[f"T_{name}"] == row["T_ref"]:
                    tally[key][name] += 1
                tally[key][f"{name}_err"].append(row[f"relerr_{name}"])

    print(f"{'lam_min(Q)':>12} {'rank def':>9} | {'HOP match':>10} {'max relerr':>11} | "
          f"{'reg match':>10} {'max relerr':>11} | {'shift match':>11} {'max relerr':>11}")
    print("-" * 100)
    for (lam, rd), t in tally.items():
        print(f"{lam:>12.3g} {rd:>9d} | {t['hop']:>6d}/{t['n']:<3d} {np.nanmax(t['hop_err']):>11.2e} | "
              f"{t['reg']:>6d}/{t['n']:<3d} {np.nanmax(t['reg_err']):>11.2e} | "
              f"{t['shift']:>7d}/{t['n']:<3d} {np.nanmax(t['shift_err']):>11.2e}")


if __name__ == "__main__":
    main()
