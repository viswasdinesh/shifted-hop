# Task: Check whether HOP-DDP silently violates its own positive-definiteness assumption

## Background

Repo: `rap-lab-org/public_HOP_horizon_optimal_tutorial` (paper: Dai & Ren,
*HOP: Fast DDP for Horizon-Optimal Trajectory Planning*, RSS 2026).

HOP-LQR's LFT-Riccati (Theorem 1) computes, per stage k:

```
E_k = Q_k^{-1}
F_k = Q_k^{-1} A_k^T
G_k = A_k Q_k^{-1} A_k^T + B_k R_k^{-1} B_k^T
```

**Assumption 2** in the paper requires `Q_k, Q_T ≻ 0` (strictly positive
definite) for all k — the derivation inverts `Q_k` directly. This is stronger
than the semidefinite requirement of ordinary LQR/DDP.

For HOP-DDP on nonlinear problems, the per-stage `Q_k` isn't the user's cost —
it's the augmented cost matrix built in Section V-A:

```
Q_k^aug = [ Q̃_k                q̃_k               ]
          [ q̃_k^T   2(ℓ(x̄_k,ū_k) + w - (1/2) ℓ_u^T ℓ_uu^{-1} ℓ_u) ]

Q̃_k = ℓ_xx,k - ℓ_xu,k ℓ_uu,k^{-1} ℓ_ux,k
```

`Q̃_k` is a Schur complement of the true cost Hessian — it is **not**
guaranteed positive definite for a nonconvex stage cost (e.g. any cost with
`cos`/`sin` terms, as in cartpole/pendulum swing-up). The scalar bottom-right
entry can also go negative when `(1/2) ℓ_u^T ℓ_uu^{-1} ℓ_u` exceeds
`ℓ + w` — i.e. exactly when the gradient is large, which is common in early
iterations far from convergence.

**The question:** does `Q_k^aug` actually go indefinite during HOP-DDP's
iterations on a nonconvex problem, and if so, does it cause the algorithm to
silently select a wrong horizon (rather than erroring or degrading gracefully)?

The paper's own results are circumstantial evidence this happens: Cartpole is
their hardest case (~60% success rate for both HOP and their brute-force
baseline), notably worse than the other three systems, all of which are
linear or have simpler nonconvex structure. But the paper doesn't report
`Q_k^aug`'s eigenvalues anywhere, so this hasn't actually been checked.

## Where indefiniteness happens — this is known analytically, don't search blind

For a separable swing-up cost with no state/control cross term
(`ℓ_xu = 0`), the Schur complement machinery does nothing: `Q̃_k = ℓ_xx,k`
exactly, and for the standard cost

```
ℓ(x,u) = 0.5*[ q_p*p^2 + q_θ*(1-cos θ)^2 + q_ṗ*ṗ^2 + q_θ̇*θ̇^2 ] + 0.5*r*u^2
```

`ℓ_xx,k` is diagonal. Three of the four diagonal entries (`q_p, q_ṗ, q_θ̇`)
are constants, positive by construction. Only the `θ` entry moves:

```
d²/dθ² [ q_θ*(1-cos θ)^2 ] = q_θ * [ 2 + 2cos θ - 4cos²θ ]
                            = -2*q_θ*(2cos θ + 1)(cos θ - 1)
```

This is **negative exactly when `cos θ < -1/2`, i.e. `θ ∈ (120°, 240°)`**,
zero at the boundaries and at `θ=0`, positive elsewhere. So `Q̃_k` has a
negative eigenvalue at every stage whose linearization point sits within 60°
of the downward position (`θ=180°`) — a swing-up trajectory starts at
`θ=π` and this window is exactly where it starts.

**Implication for how to find failure cases: don't rely on random start/goal
sampling to stumble into this.** Engineer it directly:

- Initialize the DDP nominal trajectory with zero (or small) control on the
  first outer iteration, so the pole stays near `θ=π` for the first several
  stages — this is a guaranteed hit on iteration 1, no tuning needed.
- Use a longer horizon relative to natural swing time, or a smaller time
  penalty `w`, so the optimizer isn't under pressure to shorten the horizon
  quickly — this keeps more stages dwelling near `π` across more outer
  iterations before the trajectory converges out of the bad region.
- **Before touching the full cartpole/DDP pipeline**, verify the mechanism
  in isolation with a 1-D toy: `x_{k+1} = x_k + u_k`,
  `ℓ = q*(1-cos x)^2 + r*u^2`, sweep `x_0` over `[0, 2π]`, print `ℓ_xx`.
  Five minutes, and it de-risks the cartpole implementation before
  investing in it — confirms the sign-flip window before any DDP code is
  touched.

## What to do

### 1. Instrument, don't modify (first pass)

In the HOP-DDP loop (wherever `Q_aug_k` / `Q̃_k` is built each backward-pass
iteration, per Section V-A / Algorithm 2 Step 1), add **logging only** — do
not change any algorithmic behavior:

- Eigenvalues of `Q̃_k` (the top-left block) at every stage `k` and every
  outer iteration.
- The scalar bottom-right entry of `Q_k^aug`.
- Flag: was `Q_k^aug` (the full augmented matrix, not just `Q̃_k`) positive
  definite at this (iteration, k)?
- The nominal `θ_k` at that stage, so indefinite flags can be cross-checked
  against the analytical `(120°, 240°)` prediction above.

Do this for the existing nonlinear examples first (Quadrotor) before writing
anything new — its cost may already be nonconvex enough to show something,
and it's free to check.

### 2. Add a Cartpole example if the repo doesn't have one

Match the exact interface used by the existing Double Integrator / Segway /
Quadrotor examples (same dynamics-function signature, same cost-function
signature, same way of calling into `HOP-DDP`) — do not build a separate
pipeline. Standard cartpole swing-up dynamics (cart mass `M`, pole mass `m`,
length `l`, gravity `g`):

```
x = [p, θ, ṗ, θ̇]      (cart position, pole angle from upright, velocities)
u = [F]                (horizontal force on cart)

denom = M + m*sin(θ)^2
p̈ = (F + m*sin(θ)*(l*θ̇^2 + g*cos(θ))) / denom
θ̈ = (-F*cos(θ) - m*l*θ̇^2*cos(θ)*sin(θ) - (M+m)*g*sin(θ)) / (l*denom)
```

Discretize with RK4 or the same integrator the existing examples use (check
`Quadrotor` or `Segway` example for convention — match it).

Swing-up cost as given above (`ℓ`) plus terminal cost:

```
φ(x_T) = 0.5 * [ qf_p*p^2 + qf_θ*(1 - cos θ)^2 + qf_ṗ*ṗ^2 + qf_θ̇*θ̇^2 ]
```

Start at `θ=π` (down), target `θ=0` (up) — the standard swing-up setup, and
by construction the trajectory must pass through the `(120°,240°)` window.

Run with the same experimental protocol as the paper (25 random start/goal
cases or similar), but see the "engineer it directly" bullets above —
default random sampling in `θ` may not be needed at all here since the
window is hit on essentially every swing-up by construction; the useful
randomization is over `w`, initial guess, and secondary parameters, not
over whether the bad region is reached.

### 3. The regularization ablation — this is the actual gate, not just logging indefiniteness

Knowing `Q̃_k` goes indefinite is not enough to show it matters. Check where
regularization actually sits in Algorithm 2:

- **Step 2 (Horizon Selection):** calls HOP-LQR on the raw `Q_k^aug`, with
  no regularization anywhere in Theorem 1/2 or Algorithm 1. This is what
  picks `T*`.
- **Step 3 (Truncated Backward Pass):** `Quu ← Quu + λI` (Levenberg-
  Marquardt) happens here — but only *after* `T*` is already fixed, and
  only on `Q_uu` for the chosen horizon.

The regularization that exists in the published algorithm never touches the
computation that selects the horizon. If an indefinite `Q̃_k` at some
intermediate stage distorts the `J_t` curve HOP-LQR produces in Step 2,
nothing downstream corrects it — Step 3 will regularize and converge the
backward pass for whatever `T*` it was handed, right or wrong. That is the
concrete mechanism for *silent* failure (wrong horizon selected, no error,
no NaN) rather than a crash.

**Ablation to run:** on the same nominal trajectory, execute Step 2 twice:

1. As published — raw `Q_k^aug` fed to HOP-LQR.
2. With LM-style regularization applied *before* HOP-LQR:
   `Q_k^aug ← Q_k^aug + λI`, increasing `λ` until positive definite, at
   every stage where it's needed.

Compare the resulting `J_t` curves (cost vs. candidate horizon `t`) and the
argmin `T*` from both runs.

- **Curves match, same `T*`:** the indefiniteness happens to wash out
  numerically (a negative eigenvalue at one stage doesn't necessarily flip
  the sign of the composed value at `k=0`). The mechanism is real but
  doesn't bite — this direction is much weaker than it looks.
- **Curves diverge, different `T*`:** this is the result. The published
  algorithm silently selects a different horizon than a trivially-
  regularized version would, purely because of an unguarded matrix
  inversion, on a problem class (nonconvex swing-up costs) squarely inside
  what the paper claims to handle.

This ablation is the actual deliverable of this pass — the eigenvalue
logging in (1) is diagnostic scaffolding for it, not the result itself.

### 4. What a positive result looks like, end to end

Cross-reference, per (iteration, trial):

- Was `Q_k^aug` indefinite at any stage `k` during that outer iteration's
  backward pass? (from (1), cross-checked against the `θ ∈ (120°,240°)`
  prediction)
- Did the regularized-vs-raw ablation in (3) disagree on `T*` for that
  iteration?
- Did HOP-DDP's (raw) selected `T*` match the brute-force (BF) baseline's
  selection, run on the same linearized subproblem?
- Did that trial ultimately succeed (reach the goal within threshold) or
  fail?

The interesting case is: indefinite `Q_k^aug` at some stage, the ablation
shows the regularized and raw `J_t` curves disagreeing on `T*`, and raw
HOP-DDP disagrees with BF. That's silent mis-selection with a demonstrated
cause, not just a correlation.

If `Q_k^aug` goes indefinite frequently (expected, per the analytical
argument above) but the ablation shows `T*` never changes: the failure mode
is real but inert for this cost structure, and the gap is narrower than it
looked — worth checking with a lower `w` (time penalty) or looser secondary
regularization, since the informal argument suggests the effect should be
worse when the trajectory is far from convergence, which a weaker
regularizer would prolong.

### 5. Do not fix anything yet

The goal of this pass is measurement, not a patch. If indefiniteness →
mis-selection is confirmed by the ablation, the actual fix (projective/
symplectic `[X_k; Y_k]` propagation instead of the `P̃ = P^{-1}` chart,
avoiding the `Q_k` inversion in Theorem 1 entirely) is a separate, larger
piece of work — decide whether to pursue it based on what this measurement
shows.