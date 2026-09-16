# Shifted-chart horizon-optimal LQR

Removes HOP's Assumption 2 (`Q_k > 0`) from the horizon-selection step by
tracking `Phat_k = (P_k + cI)^{-1}` instead of `P_k^{-1}`, while preserving the
`O(Nn^3)` composed-map mechanism.

Read **THEOREM.md** first — statement, proofs, numerical evidence, limitations.

## Run

```bash
python3 verify_theorem.py     # standalone, no dependencies beyond numpy
```

```bash
# copy these files into rap-lab-org/public_HOP_horizon_optimal_tutorial,
# or put that repo on PYTHONPATH, then:
python3 verify_cartpole.py
python3 ltv_sweep.py             # single-seed PD->singular->indefinite sweep, no DDP/cartpole
python3 ltv_sweep_validated.py   # 50-seed match-rate + max-relerr version, two independent reference checks (THEOREM.md §5.4)
python3 quadrotor_scale_check.py # n_aug=13 conditioning + synthetic indefinite-Q test at that scale (§5.5)
```

## Use

```python
from shifted_chart import shifted_horizon_search
J, c = shifted_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
T_star = int(np.argmin(J[T_min-1:T_max]) + T_min)
```

Drop-in for `lqr.hop_horizon_search`; `c` is chosen automatically if omitted.

## Not yet done

- validation inside a converging DDP loop (see THEOREM.md §6.1) -- attempted;
  the shift itself checks out, but surfaced a second failure mode (Quu
  near-singularity on dynamic trajectories). On review this turned out to be
  *caused by* Q's indefiniteness (regularizing the same troublesome Q_aug to
  barely-PD, same trajectory otherwise, removes it) rather than an
  independent defect -- see §6.1 and `validate_shifted_chart_ddp.py` in the
  main repo. Combining the shift with Quu-side regularization for the loop
  case is still the open follow-up.
- quadrotor: conditioning at n_aug=13 now tested and clears (both on the
  quadrotor's own near-PD cost and on synthetic indefinite Q at the same
  scale, §5.5 / `quadrotor_scale_check.py`). Still open: a physically
  motivated *nonconvex* quadrotor cost, since none exists in the tutorial
  repo (§6.2).
- literature check against indefinite-Riccati and H-infinity work (§6.4)
