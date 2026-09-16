"""
Self-contained numerical verification of THEOREM.md.  No HOP repo needed.

C1  Lemma 1     psi_k = h_k o phi lies in the family E - F(.+G)^-1 F^T,
                with the stated (E', F', G') and both E', G' symmetric.
C2  Theorem 1   the (Ebar,Fbar,Gbar) recursion equals sequential application
                of psi_{T-1}, ..., psi_0.
C3  Corollary   the composed map recovers the exact backward-Riccati P_0.
C4  Assumption  the unshifted HOP chart (c=0) fails on the same inputs.

All tests use random time-varying systems with a forced negative eigenvalue in
every Q_k -- the regime HOP's Assumption 2 excludes.

    python3 verify_theorem.py
"""

import numpy as np

from shifted_chart import _sym, psi_params, compose_maps

rng = np.random.default_rng(20260911)


def random_indefinite(n):
    """Symmetric with at least one strictly negative eigenvalue."""
    ev = rng.normal(size=n) * 6.0
    ev[0] = -abs(ev[0]) - 1.0
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return _sym(U @ np.diag(ev) @ U.T)


def random_instance(n, m, T):
    A = [rng.normal(size=(n, n)) * 0.4 + np.eye(n) for _ in range(T)]
    B = [rng.normal(size=(n, m)) for _ in range(T)]
    Q = [random_indefinite(n) for _ in range(T)]
    Rm = _sym(rng.normal(size=(m, m)))
    Rm = Rm @ Rm.T + np.eye(m)
    QT = _sym(rng.normal(size=(n, n)))
    QT = QT @ QT.T + np.eye(n)
    return A, B, Q, Rm, QT


def apply_family(E, F, G, X):
    return _sym(E - F @ np.linalg.inv(X + G) @ F.T)


def backward_riccati(A, B, Q, Rm, QT, T):
    P = _sym(QT)
    for k in reversed(range(T)):
        S = Rm + B[k].T @ P @ B[k]
        P = _sym(Q[k] + A[k].T @ P @ A[k]
                 - A[k].T @ P @ B[k] @ np.linalg.solve(S, B[k].T @ P @ A[k]))
    return P


# ----------------------------------------------------------------- C1
def check_lemma(trials=500):
    worst_family = worst_sym = 0.0
    used = 0
    for _ in range(trials):
        n, m = int(rng.integers(2, 7)), int(rng.integers(1, 4))
        A, B, Q, Rm, _ = random_instance(n, m, 1)
        A, B, Q = A[0], B[0], Q[0]
        c = float(abs(rng.normal()) * 20 + 40)
        try:
            Ep, Fp, Gp = psi_params(A, B, Q, np.linalg.inv(Rm), c)
        except np.linalg.LinAlgError:
            continue
        used += 1
        # direct: h(phi(X)) with the *unprimed* hat coefficients
        Qbi = np.linalg.inv(Q + c * np.eye(n))
        Eh, Fh = _sym(Qbi), Qbi @ A.T
        Gh = _sym(A @ Qbi @ A.T + B @ np.linalg.inv(Rm) @ B.T)
        X = _sym(rng.normal(size=(n, n)))
        X = X @ X.T + 0.5 * np.eye(n)
        phiX = X @ np.linalg.inv(np.eye(n) - c * X)
        direct = apply_family(Eh, Fh, Gh, phiX)
        pred = apply_family(Ep, Fp, Gp, X)
        worst_family = max(worst_family,
                           np.abs(pred - direct).max() / max(np.abs(direct).max(), 1e-12))
        worst_sym = max(worst_sym, np.abs(Ep - Ep.T).max(), np.abs(Gp - Gp.T).max())
    return used, worst_family, worst_sym


# ----------------------------------------------------------------- C2, C3
def check_composition(trials=200):
    worst_comp = worst_end = 0.0
    used = 0
    for _ in range(trials):
        n, m = int(rng.integers(2, 6)), int(rng.integers(1, 4))
        T = int(rng.integers(3, 25))
        A, B, Q, Rm, QT = random_instance(n, m, T)
        c = float(max(-min(np.linalg.eigvalsh(q)[0] for q in Q), 0.0) * 2 + 40)
        R_inv = np.linalg.inv(Rm)
        try:
            P = [psi_params(A[k], B[k], Q[k], R_inv, c) for k in range(T)]
        except np.linalg.LinAlgError:
            continue
        used += 1
        Ep = [p[0] for p in P]; Fp = [p[1] for p in P]; Gp = [p[2] for p in P]
        Ebar, Fbar, Gbar = compose_maps(Ep, Fp, Gp)

        PhatT = np.linalg.inv(_sym(QT) + c * np.eye(n))
        composed = apply_family(Ebar[-1], Fbar[-1], Gbar[-1], PhatT)

        seq = PhatT
        for k in reversed(range(T)):
            seq = apply_family(Ep[k], Fp[k], Gp[k], seq)
        worst_comp = max(worst_comp,
                         np.abs(composed - seq).max() / max(np.abs(seq).max(), 1e-12))

        P0_true = backward_riccati(A, B, Q, Rm, QT, T)
        P0_rec = _sym(np.linalg.inv(composed)) - c * np.eye(n)
        worst_end = max(worst_end,
                        np.abs(P0_rec - P0_true).max() / max(np.abs(P0_true).max(), 1e-12))
    return used, worst_comp, worst_end


# ----------------------------------------------------------------- C4
def check_unshifted_fails(trials=100):
    """c = 0 is exactly HOP's chart; Qbar = Q must fail when Q is indefinite."""
    failed = ok = 0
    for _ in range(trials):
        n, m = int(rng.integers(2, 6)), int(rng.integers(1, 4))
        A, B, Q, Rm, _ = random_instance(n, m, 1)
        try:
            psi_params(A[0], B[0], Q[0], np.linalg.inv(Rm), 0.0)
            ok += 1
        except np.linalg.LinAlgError:
            failed += 1
    return failed, ok


if __name__ == "__main__":
    print("=" * 72)
    print("Numerical verification of THEOREM.md   (indefinite Q throughout)")
    print("=" * 72)

    used, wf, ws = check_lemma()
    print(f"\nC1  Lemma 1 -- psi_k in the family          [{used} instances]")
    print(f"      max rel err  h(phi(X)) vs E'-F'(X+G')^-1 F'^T : {wf:.3e}")
    print(f"      max asymmetry of E', G'                       : {ws:.3e}")
    assert wf < 1e-8 and ws < 1e-10

    used, wc, we = check_composition()
    print(f"\nC2/C3  Theorem 1 -- composition             [{used} instances]")
    print(f"      max rel err  composed vs sequential psi       : {wc:.3e}")
    print(f"      max rel err  recovered P_0 vs backward Riccati: {we:.3e}")
    assert wc < 1e-5 and we < 1e-6

    failed, ok = check_unshifted_fails()
    print(f"\nC4  Assumption 2 -- unshifted chart (c = 0)")
    print(f"      rejected (Q not PD): {failed}/{failed+ok}   succeeded: {ok}")
    assert ok == 0

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED")
    print("=" * 72)
