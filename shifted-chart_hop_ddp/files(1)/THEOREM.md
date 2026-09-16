# Horizon-optimal LQR with indefinite stage costs: a shifted-chart composition theorem

**Status: verified numerically, proof written, not yet peer-reviewed. Read the
Limitations section before building on this.**

---

## 1. Background and the gap

Dai & Ren (RSS 2026) solve the horizon-optimal time-varying LQR problem in
`O(Nn^3)` rather than `O(N^2 n^3)` by rewriting the Riccati recursion as a linear
fractional transformation on the *inverse* cost-to-go and exploiting the fact
that such maps compose within their own class. Writing `Ptilde_k = P_k^{-1}`,
their Theorem 1 gives

```
gtilde_k(Ptilde) = E_k - F_k (Ptilde + G_k)^{-1} F_k^T
E_k = Q_k^{-1},   F_k = Q_k^{-1} A_k^T,   G_k = A_k Q_k^{-1} A_k^T + B_k R_k^{-1} B_k^T
```

and their Theorem 2 shows `gtilde_{0:k}` stays in the same three-parameter family,
with a forward recursion (their Eq. 14) that builds every composed map in a
single sweep.

Every coefficient contains `Q_k^{-1}`. This is their **Assumption 2**: all
`Q_k` and `Q_T` must be strictly positive definite — stronger than ordinary
LQR/DDP, which needs only `Q_k >= 0`.

The gap this matters for is HOP-DDP. The paper's Section V-A derives the
augmented per-stage cost

```
Qtilde_k = l_xx,k - l_xu,k l_uu,k^{-1} l_ux,k                            (Eq. 30)
```

which is a Schur complement of the true cost Hessian and carries no
definiteness guarantee. For any nonconvex stage cost it goes indefinite. The
canonical example is a swing-up penalty `0.5 q_th (1 - cos th)^2`, whose second
derivative

```
q_th (sin^2 th + cos th - cos^2 th)
```

is negative exactly when `cos th < -1/2`, i.e. `th` in (120, 240) degrees —
precisely where a swing-up trajectory begins. Collision-avoidance penalties,
named in the paper's own future-work section, are nonconvex for the same reason.

So HOP-LQR's horizon selection is undefined on a class of problems HOP-DDP
claims to handle.

## 2. Result

Track a **shifted** inverse instead:

```
Phat_k = (P_k + cI)^{-1},        c > 0 constant
```

**Lemma 1.** With `Qbar_k = Q_k + cI` and

```
Ehat_k = Qbar_k^{-1},  Fhat_k = Qbar_k^{-1} A_k^T,
Ghat_k = A_k Qbar_k^{-1} A_k^T + B_k R_k^{-1} B_k^T,
K_k    = (I - c Ghat_k)^{-1}
```

the one-step map `psi_k` taking `Phat_{k+1}` to `Phat_k` is

```
psi_k(X) = E'_k - F'_k (X + G'_k)^{-1} F'_k^T
E'_k = Ehat_k + c Fhat_k K_k Fhat_k^T,    F'_k = Fhat_k K_k,    G'_k = Ghat_k K_k
```

with `E'_k` and `G'_k` symmetric.

**Theorem 1.** The composed maps `psi_{0:k}` satisfy the same recursion as
HOP's Eq. 14, applied to the primed coefficients:

```
W_k    = (E'_k + Gbar_{k-1})^{-1}
Ebar_k = Ebar_{k-1} - Fbar_{k-1} W_k Fbar_{k-1}^T
Fbar_k = Fbar_{k-1} W_k F'_k
Gbar_k = G'_k - F'_k^T W_k F'_k
```

with base case `(Ebar_0, Fbar_0, Gbar_0) = (E'_0, F'_0, G'_0)`.

**Consequences.** Requires only `Q_k + cI > 0`, achievable for *any* symmetric
`Q_k` by taking `c > -lambda_min(Q_k)`. The shift is exact, not a perturbation:
`P_k = Phat_k^{-1} - cI` recovers the solution of the original problem.
Complexity stays `O(Nn^3)`.

## 3. Proofs

### 3.1 Lemma 1

Adding `cI` to both sides of HOP's Eq. 10 `P_k = Q_k + A_k^T (Ptilde_{k+1} + S_k)^{-1} A_k`
(with `S_k = B_k R_k^{-1} B_k^T`) gives

```
P_k + cI = Qbar_k + A_k^T (Ptilde_{k+1} + S_k)^{-1} A_k
```

Their Theorem 1 derivation now applies verbatim with `Q_k -> Qbar_k`, yielding
`h_k(X) = Ehat_k - Fhat_k (X + Ghat_k)^{-1} Fhat_k^T`. Every inverse in the
coefficients is `Qbar_k^{-1}`, which exists by construction.

But `h_k` consumes an *unshifted* `Ptilde`, while the chart carries `Phat`. From
`P = Phat^{-1} - cI`,

```
Ptilde = P^{-1} = Phat (I - c Phat)^{-1} =: phi(Phat)
```

so `psi_k = h_k o phi`. It remains to show this lands back in the family.
Dropping subscripts, write `E, F, G` for the hat coefficients.

**(a) Factor.** `X` commutes with `(I - cX)^{-1}`, so

```
phi(X) + G = (I - cX)^{-1} [ X + (I - cX) G ] = (I - cX)^{-1} [ X(I - cG) + G ]
           = (I - cX)^{-1} [ X + G(I - cG)^{-1} ] (I - cG)
```

(check the last step by expanding). Set `G' := G (I - cG)^{-1}`.

**(b) Invert.**

```
(phi(X) + G)^{-1} = (I - cG)^{-1} (X + G')^{-1} (I - cX)
```

**(c) Remove the X-dependence of the trailing factor.** Since
`I - cX = (I + cG') - c(X + G')`,

```
(X + G')^{-1} (I - cX) = (X + G')^{-1} (I + cG') - cI
```

**(d) Key identity.**

```
I + cG' = (I - cG)(I - cG)^{-1} + cG(I - cG)^{-1} = [(I - cG) + cG](I - cG)^{-1} = (I - cG)^{-1}
```

This is what makes the result symmetric: the leftover factor is exactly the
transpose partner of the one already on the left.

**(e) Assemble.**

```
h(phi(X)) = E - F(I-cG)^{-1} [ (X+G')^{-1}(I-cG)^{-1} - cI ] F^T
          = [E + cF(I-cG)^{-1}F^T] - [F(I-cG)^{-1}] (X+G')^{-1} [(I-cG)^{-1}F^T]
```

giving `E'`, `F'`, `G'` as stated.

**Symmetry.** `Ghat` symmetric implies `(I - c Ghat)^{-1}` symmetric; it is a
rational function of `Ghat`, so the two commute and `G'^T = (I-cG)^{-1}G = G'`.
`E'` is `Ehat` plus `Fhat M Fhat^T` with `M` symmetric. Both symmetric.

**Scalar check.** `E' = E + cF^2/(1-cG) = [E - c(EG - F^2)]/(1-cG)`, matching the
2x2 Mobius product of `h` and `phi` after projective normalisation. QED.

### 3.2 Theorem 1

Induction. Assume `psi_{0:k-1}(X) = Ebar_{k-1} - Fbar_{k-1}(X + Gbar_{k-1})^{-1} Fbar_{k-1}^T`.
Then `psi_{0:k} = psi_{0:k-1} o psi_k`:

```
psi_{0:k}(X) = Ebar_{k-1} - Fbar_{k-1} [ W_k^{-1} - F'_k (X + G'_k)^{-1} F'_k^T ]^{-1} Fbar_{k-1}^T
W_k := (E'_k + Gbar_{k-1})^{-1}
```

Apply Woodbury to the bracket with `A = W_k^{-1}`, `U = -F'_k`,
`C = (X + G'_k)^{-1}`, `V = F'_k^T`:

```
M = W_k + W_k F'_k [ (X + G'_k) - F'_k^T W_k F'_k ]^{-1} F'_k^T W_k
```

Substituting and grouping gives the stated `Ebar_k`, `Fbar_k`, `Gbar_k`. This is
structurally identical to HOP's Eq. 16-18; only the coefficient set differs. QED

### 3.3 Side condition

`I - c Ghat_k` must be invertible. Since `Qbar_k > 0` and `R_k > 0`, `Ghat_k > 0`,
so this fails exactly at `c = 1/lambda_i(Ghat_k)`. Combined with the lower bound
`c > -lambda_min(Q_k)`, admissible `c` is an open set with finitely many excluded
points per stage. Note `I - c Ghat_k` is symmetric but **indefinite** whenever
`c > 1/lambda_max(Ghat_k)`, which is the normal operating regime — factor it with
LU or LDL^T, never Cholesky.

## 4. Algorithm

Algorithm 1 of the paper changes in exactly one place: a preprocessing step
mapping `(Ehat_k, Fhat_k, Ghat_k) -> (E'_k, F'_k, G'_k)`. Phase 1's composition
loop and Phase 2's per-horizon query are untouched.

```
choose c                                        # one eigenvalue pass, Sec. 5.3
for k in 0..N-1:                                # Phase 1a
    Qbar_k = Q_k + cI;  Ehat, Fhat, Ghat as above
    factor (I - c Ghat_k) once, two solves  ->  F'_k, G'_k;  then E'_k
compose via Eq. 14 on (E', F', G')              # Phase 1b, unchanged
for t in 1..N:                                  # Phase 2, unchanged
    PhatT = (Q_T^(t) + cI)^{-1}
    Phat0 = Ebar_{t-1} - Fbar_{t-1}(PhatT + Gbar_{t-1})^{-1} Fbar_{t-1}^T
    J_t   = 0.5 z0^T (Phat0^{-1} - cI) z0
```

**Cost.** Per stage, one extra `n x n` factorization and ~3 extra matmuls.
Across the sweep: ~6N inverses and ~10N matmuls versus HOP's ~5N and ~7N —
roughly +20% factorizations, +40% matmuls, all in the once-per-iteration Phase 1.
Phase 2 is bit-identical in cost, so the overhead does not grow with the number
of candidate horizons. Complexity `O(Nn^3)`; brute force remains `O(N^2 n^3)`.

## 5. Numerical evidence

### 5.1 Synthetic (`verify_theorem.py`, no repo dependency)

Random time-varying systems, `n` in 2..6, `m` in 1..3, `T` in 3..24, with a
**forced negative eigenvalue in every `Q_k`**:

| check | quantity | result |
|---|---|---|
| C1 | `h(phi(X))` vs `E' - F'(X+G')^{-1}F'^T`, 500 instances | 2.0e-12 |
| C1 | asymmetry of `E'`, `G'` | 0.0 |
| C2 | composed recursion vs sequential `psi`, 200 instances | 1.5e-09 |
| C3 | recovered `P_0` vs backward Riccati | 5.1e-09 |
| C4 | unshifted chart (`c = 0`) on the same inputs | 100/100 rejected |

### 5.2 Cartpole against the HOP repo (`verify_cartpole.py`)

**Repo's own `Q_aug`** (quadratic in wrapped error; Schur complement of the
top-left block is exactly `2w > 0`, so Assumption 2 holds at the stages):

| method | max rel. err vs per-horizon Riccati |
|---|---|
| HOP (unshifted) | 4.97e-04 |
| shifted, `c = 10` | **1.41e-12** |

Nine orders of magnitude more accurate *on HOP's own problem*. Cause: the
terminal matrices are rank-`n` by construction —
`Q_T = [[P, Pe],[e^T P, e^T P e + 1e-12]]` has Schur complement exactly `1e-12`,
measured `lambda_min = -8.5e-14` — and `chol_inv`'s default jitter of `1e-9` is
~10^4 times larger, so inverting them loses roughly nine digits. The shift
regularises them correctly instead. Accuracy stays below 1e-10 for `c` in
[0.1, 1e3].

**True Eq. 30 `Q_aug`** with the `(1-cos th)^2` swing-up cost
(`lambda_min = -20` at stages, `-300` at terminals):

| method | max rel. err |
|---|---|
| HOP (unshifted) | **9.99e-01** (100% wrong) |
| shifted, `c = 600` | 1.66e-14 |

HOP does not degrade gracefully here; it returns garbage without raising.
(`chol_inv` exhausts its eight jitter retries at 1e-9..1e-2 and falls through to
a generic solve at `eps = 0.1`, which is negligible against `lambda_min = -300`.)

### 5.3 The `c` window

Floor is `c > max(0, -lambda_min)` over all stages and terminals — computable in
one eigenvalue pass, not a tuned hyperparameter. On the indefinite cartpole
(floor 300, set by the terminals):

| `c` / floor | 1.001 | 1.1 | 2 | 10 | 100 | 1e3 | 1e4 |
|---|---|---|---|---|---|---|---|
| max rel. err | 3.7e-14 | 1.9e-14 | 1.7e-14 | 2.9e-12 | 1.1e-11 | 1.3e-09 | 7.2e-08 |

Four orders of usable margin above the floor. Degradation past ~1e3x comes from
cancellation in `P = Phat^{-1} - cI`, as predicted. `cond(I - c Ghat)` peaked at
1.5e7 near the floor and ~1e6 thereafter; no singularity was encountered.

Suggested rule: `c = 2 max(0, -lambda_min(Q_k), -lambda_min(Q_T^(t)))`, recomputed
per DDP iteration. Implemented as `choose_shift()`.

### 5.4 Pure LTV sweep, no DDP, no physical system (`ltv_sweep.py`, `ltv_sweep_validated.py`)

Random time-varying `A_k, B_k` (`n=6, m=3, T=60`), `Q_k` swept from PD through
singular-PSD to indefinite, against a per-horizon backward-Riccati reference,
three methods: raw (unshifted) HOP, "regularize-`Q` then run stock HOP"
(Algorithm-2-style `mu*I` escalation), and the shifted chart. This isolates
the LTV mechanism with no confound from DDP convergence or the cartpole cost.

**Validation of the reference itself.** `riccati_reference` is a fully
unregularized backward-Riccati sweep, and a fully unregularized sweep is
exactly what developed spurious near-`Quu`-singularity spikes on the
cartpole DDP trajectory (Sec. 6.1). Before trusting it here it needs its own
check — done twice, independently:

- vs. `shifted_horizon_search(c=choose_shift(...))` (`c=20` here, near the
  accuracy floor): max relative difference `1.93e-12`. (An earlier version
  of this check used `c=1e4` and got `1.9e-6`, then reported that as
  agreement — backwards. Sec. 5.3 already shows `c=1e4` is the *worst*
  choice tested, `7.2e-08` even on a case where the floor is 300 -- error
  from `P = Phat^{-1} - cI` cancellation grows monotonically with `c`. The
  `1.9e-6` was measuring that cancellation, not the reference's
  trustworthiness. It also isn't a fully independent check: composed-LFT
  and the per-horizon sweep are different code paths, but both ultimately
  encode the same Riccati recursion, so agreement here is necessary but not
  sufficient.)
- vs. `dense_qp_reference` at `T=6`: stack the controls into one vector,
  eliminate the states by forward substitution, and solve the resulting
  dense QP's normal equations directly — no Riccati recursion, no LFT
  composition, a genuinely separate computation. Result: `1.69e-14`.

Both small: the LTV reference is trustworthy. This is because the sweep's
own `A_k` are `I + 0.15*N(0,1)` (a mild, well-conditioned perturbation of
the identity), not a fast-swinging pendulum linearization, so `Quu` never
approaches singular here (see Sec. 6.1's update for what happens when it
does, and why that turns out to still be a `Q`-indefiniteness effect rather
than an independent one).

**Correcting a comparison error found during validation.** The original
single-seed table's write-up flagged shifted's `T*=6` on the `lambda_min=-10`
row as questionable, benchmarked against a "reference `T*=1`" printed at the
script's end — but that line calls `make_Q()` again *after* the main loop, so
by then the RNG has moved on and it is an unrelated, freshly-drawn PD matrix,
not that row's ground truth. Compared against the row's *own*
`riccati_reference` output (true `argmin = 6`, a real minimum 22% below the
surrounding plateau, not a tie), shifted's `T*=6` is exactly correct; raw
HOP and reg+HOP both report `T*=1` there, which is wrong.

**50-seed exact-`T*`-match rate and max relative `J` error**, same 9 cases,
reference re-derived per seed per case:

| `lambda_min(Q)` | rank def. | HOP match | HOP max relerr | reg+HOP match | reg max relerr | shifted match | shifted max relerr |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 44/50 | 3.9e-08 | 44/50 | 3.9e-08 | 43/50 | 9.2e-15 |
| 1e-3 | 0 | 42/50 | 4.1e-08 | 42/50 | 4.1e-08 | 41/50 | 1.1e-08 |
| 1e-6 | 0 | 41/50 | 6.3e-04 | 41/50 | 6.3e-04 | 42/50 | 9.5e-04 |
| 0 | 0 | 2/50 | 21.6 | 45/50 | 7.0e-04 | 44/50 | 2.2e-06 |
| 0 | 2 | 0/50 | 43.0 | 9/50 | 1.2e-03 | 12/50 | 3.2e-06 |
| 0 | 4 | 0/50 | 94.8 | 46/50 | 7.3e-04 | 46/50 | 7.1e-06 |
| -1e-3 | 0 | 2/50 | 49.1 | 41/50 | 7.4e-03 | 39/50 | 2.5e-09 |
| **-1** | 0 | 7/50 | 6.06 | 7/50 | 36.2 | **44/50** | **3.6e-12** |
| **-10** | 0 | 8/50 | **1.15e4** | 6/50 | **2.75e4** | **50/50** | **8.4e-09** |

The `max relerr` columns are the more defensible claim than the match
counts, and they tell a sharper story on their own: HOP's error reaches
`1.15e4` (worst case, `lambda_min=-10`) and stays above `6` even where it
occasionally gets `T*` right; shifted never exceeds `9.5e-4`, and that one
elevated value is at `lambda_min=1e-6` (essentially PD, where all three
methods sit near the same modest floor from general floating-point
sensitivity, not indefiniteness).

**Why the match counts need the error column next to them.** This toy
system has no time penalty (`J = 0.5 z0^T P z0`, no `+wT` term), so for a
well-conditioned system the cost-to-go plateaus once the horizon reaches
(near-)steady-state, producing ties. Diagnosed directly by separating each
mismatch into "tie" (`relative gap < 1e-6` between the reference's and the
method's chosen `T`) vs. "real" (larger gap):

| rank def. (`lambda_min=0`) | HOP mismatches | shifted mismatches |
|---|---|---|
| 0 (rank 5) | 7 ties + **41 real** | 6 ties + **0 real** |
| 2 (rank 3) | 1 tie + **49 real** | 38 ties + **0 real** |
| 4 (rank 1) | 2 ties + **48 real** | 4 ties + **0 real** |

**Shifted's mismatches are 100% ties, at every rank deficiency tested —
zero real errors.** That resolves the `rank_def=2` puzzle cleanly: it isn't
that shifted performs worse there: rank 3 simply produces more tied optima
(38/50) than rank 5 or rank 1 do, and shifted correctly lands on *a* tied
optimum every time, just not always the reference's arbitrary tie-break
index. HOP's mismatches, by contrast, are overwhelmingly real at every
rank deficiency (41, 49, 48 out of a possible 50) — it is not almost-right
and unlucky on ties, it is actually wrong. The genuinely diagnostic rows
for a match-rate reading are still `lambda_min=-1` and `-10`, where the
true minimum is an isolated dip, not a tie: shifted 44/50 and 50/50 there,
both with real max relerr in the `1e-9`..`1e-12` range, versus HOP's 6-8/50
with real errors up to `2.75e4`. Adding a small `w > 0` per-step penalty
would remove the plateau and make the PD/singular rows' match rates
meaningful in the same direct way -- not done here to keep this a minimal,
targeted check.

### 5.5 Higher dimension (`n_aug = 13`, matching the quadrotor)

Closes former Limitation #2 in part. Two separate tests, since they check
different things.

**(a) The quadrotor's own cost, `n_aug=13, T=160`** (repo's `Q_aug` via
`build_augmented_system`, all-zero-control nominal): `min eig` is `7.7e-3`
at the stages and `2.4e-14` at the terminals — essentially PD already, so
this tests *conditioning at scale*, not indefiniteness at scale.

| method | max rel. err | rel. err at `t=160` | `T*` |
|---|---|---|---|
| HOP (unshifted) | 6.71e-05 | 9.31e-06 | 52 |
| shifted, `c=10` | 2.72e-13 | 1.52e-13 | 52 |
| shifted, `c=choose_shift()=0.001` | 8.04e-11 | 2.71e-11 | 52 |

No degradation from `t=52` (the optimum) out to `t=160`: the composed-map
mechanism stays well-conditioned at the dimension that broke the earlier
chart-free `[X;Y]` pair-propagation attempt (Limitation #2, `cond` growing
`~1e5` per step, unusable by `t=5`). This clears the conditioning risk that
motivated flagging the quadrotor as untested.

**(b) Synthetic indefinite `Q_k` at the same scale**, `n=13, m=4, T=160`
(same construction as Sec. 5.4's LTV sweep, not the quadrotor's actual
dynamics): this closes the gap that (a) leaves open, since the quadrotor's
own cost never exercises indefiniteness at `n=13`.

| `lambda_min(Q)` | `c` | HOP max relerr (`T*`, ref) | shifted max relerr (`T*`) |
|---|---|---|---|
| 1 | 0.001 | 3.52e-08 (5, ref 5) | 1.39e-15 (5) |
| 0 | 0.001 | **7.13e-01** (20, ref 5) | 3.66e-07 (5) |
| -1 | 2 | **3.80e-01** (28, ref 5) | 1.80e-14 (5) |
| -10 | 20 | **5.69e-01** (4, ref 28) | 2.99e-09 (28) |

Same story as `n=6` in Sec. 5.4, unchanged by the six-fold jump in
dimension: shifted tracks the reference to `1e-7`..`1e-15` and always finds
the true `T*`; HOP is 40-70% wrong and picks the wrong horizon whenever `Q`
is not PD. No conditioning blowup at any of the four cases -- `choose_shift`'s
floor-based `c` was sufficient throughout, matching the outlook Limitation
#2 originally hoped for ("HOP's own composition is stable... but run it
before claiming generality"). Now run: the outstanding item is a *physical*
n=13 nonconvex cost (an actual quadrotor maneuver whose cost is genuinely
indefinite, analogous to cartpole's `(1-cos th)^2`), not a synthetic
random one -- see updated Limitation #2.

### 5.6 HOP-LQR fails on `Q >= 0`, which ordinary LQR accepts

Reframed from a former limitation (see Limitation #5's history): ordinary
fixed-horizon LQR only needs `Q_k, Q_T >= 0` (Sec. III-A of the background);
Assumption 2 for HOP-LQR's horizon search additionally requires strict
positive-definiteness, because Theorem 1 inverts `Q_k` directly. Sec. 5.4's
`rank_def in {0,2,4}` rows are exactly this case (`lambda_min(Q)=0`, no
indefiniteness at all, just a `Q` that doesn't penalize every state
direction) and HOP fails there completely: `0-2` out of 50 seeds correct,
real (non-tie) relative error `21`-`95`. No nonconvex cost, no swing-up, no
obstacle needed to trigger this -- a cost that simply doesn't penalize
velocity, say, is already enough.

This does not contradict the repo's own passing tests: `build_augmented_system`
unconditionally adds `1e-9*I` to the top-left block of every `Q_k^aug`
(`lqr.py`, the `Qk[:n,:n] = _sym(Q) + 1e-9*np.eye(n)` line), so any
singular-but-PSD `Q` a user supplies through that path never reaches
`hop_horizon_search` as singular -- the `1e-9` loading absorbs it silently,
with a measured error contribution below `1e-8` (Sec. 5.2). The failure
above is what happens when `Q` reaches the horizon search directly, as it
would from any caller that does not route through that specific function
(e.g. computing `Q_aug` a different way, or using `hop_horizon_search`
standalone, both of which are supported call patterns since the two
functions are separately exported).

## 6. Limitations

Stated plainly, because several earlier hypotheses in this line of work did not
survive testing.

1. **Not validated inside a converging DDP loop.** All cartpole numbers use the
   zero-control nominal, where the pole sits at `th = pi` for every stage. `J(T)`
   is therefore monotone and the reference `T*` sits at the search boundary. The
   accuracy comparison is sound; a claim about correct *horizon selection*
   is not yet supported and needs a converging swing-up.

   **Partially addressed, and a second failure mode found -- then
   re-diagnosed on review.** A one-shot "flick" warm start (nonzero `u` for
   the first few steps) breaks the exact `th = pi` fixed point and lets the
   DDP loop actually progress (8+ outer iterations observed, vs. stalling at
   iteration 0 from the zero-control nominal). On that genuinely
   time-varying trajectory, `J_shift(T)` tracks the (Quu-regularized)
   brute-force ground truth closely for small `T`, then develops large
   spurious negative spikes at specific horizons (observed: `-831` at
   `T=20`, `-3351` at `T=29`, `-14274` at `T=47`, against a true minimum of
   `-569` at `T=17`) that match a *fully unregularized* backward Riccati
   sweep to 3+ digits.

   The first write-up called this a control-side (`Quu`) conditioning issue
   *orthogonal* to the `Q_k`-indefiniteness problem the shift targets, and
   asked whether raw `hop_horizon_search` spikes identically on this
   trajectory (it should, if the two problems are truly independent, since
   HOP's Phase 2 does not regularize `Quu` either). Tested directly, and the
   answer is no: on the *exact same trajectory*, using the repo's own
   quadratic cost (`Q_aug` PD by construction) instead of the true
   `(1-cos th)^2` cost, neither `hop_horizon_search` nor `shifted_horizon_search`
   spikes anywhere in `T=1..100` -- both track the reference smoothly.
   Sharper still: taking the *exact* indefinite `Q_aug` that produced the
   spikes and regularizing it to barely-PD (`min eig = 1e-3`, same
   trajectory, same `A_k/B_k`, nothing else changed) removes every spike.
   So this is **not** an independent, `Q`-agnostic defect in Step 2 -- it is
   caused by `Q`'s own indefiniteness, which drives the value function `P_k`
   into a regime where `Quu = R + B^T P_{k+1} B` happens to pass
   near-singular at specific horizons. The practical conclusion is
   unchanged (`argmin(J_shift)` is not yet safe to use naively inside a real
   DDP loop on an indefinite cost, and Algorithm 2's `Quu += lambda I`
   arrives too late in Step 3 to help Step 2), but the mechanism is: severe
   `Q`-indefiniteness can produce a *second-order* symptom in `Quu`, not two
   unrelated failure modes. Reproducers: `validate_shifted_chart_ddp.py` in
   the main repo (the original finding) and the PD-cost / regularized-`Q_aug`
   controls described above (same trajectory, `w=0.01, q_theta=13.15,
   r=0.0471`, iteration 0) for the re-diagnosis.

2. **Quadrotor: conditioning at `n_aug=13` now tested and clears; a genuinely
   indefinite quadrotor cost is still untested.** The `n = 13` augmented
   quadrotor is where an earlier chart-free formulation (symplectic `[X;Y]`
   pair propagation with prefix products) failed catastrophically — `cond`
   growing ~1e5 per step, unusable by `t = 5` — despite passing every
   single-step check. Sec. 5.5 now runs the shifted chart at this exact
   dimension two ways: on the quadrotor's own (near-PD) cost out to
   `T=160` (no degradation, `2.7e-13`..`8.0e-11` depending on `c`), and on
   synthetic indefinite `Q_k` at the same `n=13, T=160` (shifted stays at
   `1e-7`..`1e-15`, HOP is 40-70% wrong). Both clear cleanly -- no `cond`
   blowup anywhere, unlike the earlier pair-propagation attempt. What's
   *not* tested: a physically-motivated quadrotor cost that is itself
   nonconvex (the analogue of cartpole's `(1-cos th)^2`), since no such cost
   exists in the tutorial repo for quadrotor. The synthetic test rules out a
   dimension-driven conditioning failure; it does not by itself establish
   that a real quadrotor maneuver would hit the indefinite regime at all.

3. **`c`-window mapping is coarse.** Nine points on one system. The excluded set
   `c = 1/lambda_i(Ghat_k)` is measure-zero and was never hit, but a fine sweep
   near those values has not been done.

4. **Novelty is in the composition-preserving form, not the ideas.** Projective
   and symplectic representations of Riccati recursions are classical (Vaughan;
   Bittanti-Laub-Willems; Anderson & Moore; Krein-space methods of
   Hassibi-Sayed-Kailath). Shifting a matrix to make it definite is elementary.
   The claim is narrow and specific: the shifted chart preserves the exact
   three-parameter family that HOP's Eq. 14 induction requires, so the `O(N)`
   horizon-search mechanism survives the removal of Assumption 2. A literature
   check against the indefinite-Riccati and `H_inf` literature has **not** been
   completed.

5. **No claim about `Q_k >= 0` without a shift** — by construction: the
   shift is exactly what's needed once `Q` is not strictly PD, `c=0`
   recovers unshifted HOP. This used to be filed here as a caveat because
   the repo's own `1e-9` diagonal loading absorbs singular-PSD `Q` with no
   measurable error *through that specific code path*. Moved to Sec. 5.6 as
   a result instead: fed directly to `hop_horizon_search` (bypassing that
   loading, a supported call pattern), singular `Q` is not a benign edge
   case -- HOP fails on it outright (`0-2/50` seeds correct, real relative
   error `21`-`95`), with no nonconvex cost or swing-up involved. The
   `1e-9` loading is incidental protection specific to one call path, not a
   property of the algorithm.

## 7. Files

| file | purpose |
|---|---|
| `THEOREM.md` | this document |
| `shifted_chart.py` | reference implementation: `psi_params`, `compose_maps`, `shifted_horizon_search`, `choose_shift` |
| `verify_theorem.py` | self-contained checks C1-C4; no repo needed |
| `verify_cartpole.py` | cartpole validation against HOP; requires the repo on the import path |
| `ltv_sweep.py` | single-seed PD->singular->indefinite LTV sweep, no DDP/cartpole; requires the repo on the import path (`hop_horizon_search`) |
| `ltv_sweep_validated.py` | Sec. 5.4: two independent reference cross-checks (`dense_qp_reference` needs no repo code at all) + 50-seed match-rate/max-relerr tables |
| `quadrotor_scale_check.py` | Sec. 5.5: `n_aug=13` (quadrotor dimension) conditioning check on the repo's own cost + a synthetic indefinite-`Q` sweep at the same scale |

```
python3 verify_theorem.py        # standalone
python3 verify_cartpole.py       # needs ddp.py, lqr.py, systems.py, utils.py
python3 ltv_sweep.py             # needs lqr.py (hop_horizon_search) only
python3 ltv_sweep_validated.py   # same
python3 quadrotor_scale_check.py # part (a) needs the full repo; part (b) needs only lqr.py
```

## 8. References

- M. Dai, Z. Ren. *HOP: Fast Differential Dynamic Programming for
  Horizon-Optimal Trajectory Planning.* RSS 2026.
  Code: `github.com/rap-lab-org/public_HOP_horizon_optimal_tutorial`
- K. Stachowicz, E. Theodorou. *Optimal-Horizon Model Predictive Control with
  Differential Dynamic Programming.* ICRA 2022. arXiv:2111.09207
- B.D.O. Anderson, J.B. Moore. *Optimal Control: Linear Quadratic Methods.* 1990.
- S. Bittanti, A. Laub, J.C. Willems (eds). *The Riccati Equation.* 1991.
