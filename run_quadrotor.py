#!/usr/bin/env python3
"""Run the 12-DOF Quadrotor brute-force vs HOP tutorial example."""

import argparse
import time
from pathlib import Path

import numpy as np

from ddp import solve_hop
from systems import make_quadrotor


def _final_cost(result):
    if result["J_hist"]:
        return float(result["J_hist"][-1])
    return float("nan")


def _plot_summary(plt, outdir, res_bf_q, res_hop_q):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    results = []
    if res_bf_q is not None:
        results.append((res_bf_q, "#2196F3", "--", "Brute-Force"))
    results.append((res_hop_q, "#E91E63", "-", "HOP"))

    for res, color, linestyle, label in results:
        J_curve = res["J_curve"]
        ax.plot(np.arange(1, len(J_curve) + 1), J_curve, color=color, ls=linestyle, lw=2.5, label=label, alpha=0.85)
        T_star = res["T_star"]
        ax.scatter([T_star], [J_curve[T_star - 1]], color=color, s=100, zorder=5, edgecolors="white", lw=2)
        ax.annotate(f"$T^*={T_star}$", (T_star, J_curve[T_star - 1]), textcoords="offset points", xytext=(10, 12), fontsize=11, color=color)

    ax.set(xlabel="Horizon $T$", ylabel="$J(T)$", title="Quadrotor: Cost vs. Horizon")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    timer_keys = ["linearize", "select", "backward", "forward"]
    colors_bar = ["#80CBC4", "#EF9A9A", "#90CAF9", "#A5D6A7"]
    labels = ["HOP"] if res_bf_q is None else ["Brute-Force", "HOP"]
    plotted_results = [res_hop_q] if res_bf_q is None else [res_bf_q, res_hop_q]
    x_pos = np.arange(len(plotted_results))
    bottoms = np.zeros(len(plotted_results))
    for i, timer_key in enumerate(timer_keys):
        vals = [res["timers"][timer_key] for res in plotted_results]
        ax2.bar(x_pos, vals, bottom=bottoms, label=timer_key, color=colors_bar[i], width=0.45, edgecolor="white")
        bottoms += vals

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set(ylabel="Time (s)", title="Runtime Breakdown")
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "quadrotor_cost_runtime.png", dpi=160)
    print("Saved optional figure to outputs/quadrotor_cost_runtime.png")


def _plot_state_trajectories(plt, outdir, F_q, x0_q, xg_q, res_bf_q, res_hop_q):
    fig = plt.figure(figsize=(18, 14))

    style = {
        "BF": ("#2196F3", "-", 4, 0.5, "Brute-Force"),
        "HOP": ("#E91E63", "-", 1.5, 1.0, "HOP"),
    }
    results = []
    if res_bf_q is not None:
        results.append(("BF", res_bf_q))
    results.append(("HOP", res_hop_q))

    ax3 = fig.add_subplot(3, 4, 1, projection="3d")
    for tag, res in results:
        X, T_star = res["X"], res["T_star"]
        color, linestyle, linewidth, alpha, label = style[tag]
        ax3.plot(X[: T_star + 1, 0], X[: T_star + 1, 1], X[: T_star + 1, 2],
                 color=color, ls=linestyle, lw=linewidth, alpha=alpha, label=label)
    ax3.scatter(*x0_q[:3], color="green", s=70, marker="o", label="Start")
    ax3.scatter(*xg_q[:3], color="red", s=70, marker="*", label="Goal")
    ax3.set(xlabel="X", ylabel="Y", zlabel="Z", title="3D Trajectory")
    ax3.legend(fontsize=7)

    state_names = [
        "$p_x$", "$p_y$", "$p_z$",
        "$v_x$", "$v_y$", "$v_z$",
        "$\\phi$", "$\\theta$", "$\\psi$",
        "$\\omega_x$", "$\\omega_y$",
    ]
    state_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i, (state_i, state_label) in enumerate(zip(state_idx, state_names)):
        ax = fig.add_subplot(3, 4, i + 2)
        for tag, res in results:
            X, T_star = res["X"], res["T_star"]
            color, linestyle, linewidth, alpha, label = style[tag]
            t = np.arange(T_star + 1) * F_q.dt
            ax.plot(t, X[: T_star + 1, state_i], color=color, ls=linestyle, lw=linewidth, alpha=alpha, label=label)
        ax.axhline(xg_q[state_i], color="gray", ls=":", alpha=0.4)
        ax.set(xlabel="Time (s)", title=state_label)
        if i == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(outdir / "quadrotor_states.png", dpi=160)
    print("Saved optional figure to outputs/quadrotor_states.png")


def main():
    parser = argparse.ArgumentParser(description="Run the Quadrotor HOP tutorial example.")
    parser.add_argument("--max-iter", type=int, default=15, help="Maximum iLQR iterations for each solver.")
    parser.add_argument("--skip-bruteforce", action="store_true", help="Only run HOP. The brute-force check is slower.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional summary figure.")
    args = parser.parse_args()

    F_q, x0_q, xg_q, uref_q, Q_q, R_q, alpha_q, w_q, N_q, Tmin_q, Tmax_q, wrap_q = make_quadrotor()

    res_bf_q = None
    time_bf_q = None
    if not args.skip_bruteforce:
        print("Running Brute-Force...")
        tic = time.perf_counter()
        res_bf_q = solve_hop(
            F_q, x0_q, xg_q, uref_q, Q_q, R_q, alpha_q, w_q,
            N_q, Tmin_q, Tmax_q, method="bruteforce", max_iter=args.max_iter, wrap_idx=wrap_q
        )
        time_bf_q = time.perf_counter() - tic
        print(f"  T*={res_bf_q['T_star']}, J*={_final_cost(res_bf_q):.2f}, time={time_bf_q:.1f}s")

    print("\nRunning HOP...")
    tic = time.perf_counter()
    res_hop_q = solve_hop(
        F_q, x0_q, xg_q, uref_q, Q_q, R_q, alpha_q, w_q,
        N_q, Tmin_q, Tmax_q, method="hop", max_iter=args.max_iter, wrap_idx=wrap_q
    )
    time_hop_q = time.perf_counter() - tic
    print(f"  T*={res_hop_q['T_star']}, J*={_final_cost(res_hop_q):.2f}, time={time_hop_q:.1f}s")

    if res_bf_q is not None:
        sel_bf = res_bf_q["timers"]["select"]
        sel_hop = res_hop_q["timers"]["select"]
        print(f"\n{'-' * 50}")
        print(f"Total speedup:  {time_bf_q / max(time_hop_q, 1e-9):.1f}x")
        print(f"Select speedup: {sel_bf / max(sel_hop, 1e-9):.1f}x  (BF={sel_bf:.2f}s, HOP={sel_hop:.2f}s)")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping the optional figure.")
        return

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    _plot_summary(plt, outdir, res_bf_q, res_hop_q)
    _plot_state_trajectories(plt, outdir, F_q, x0_q, xg_q, res_bf_q, res_hop_q)


if __name__ == "__main__":
    main()
