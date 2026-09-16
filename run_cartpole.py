#!/usr/bin/env python3
"""Run the Cartpole swing-up brute-force vs HOP tutorial example."""

import time
from pathlib import Path

import numpy as np

from ddp import solve_hop
from systems import make_cartpole


def _final_cost(result):
    if result["J_hist"]:
        return float(result["J_hist"][-1])
    return float("nan")


def _plot_summary(plt, outdir, res_bf, res_hop):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    results = []
    if res_bf is not None:
        results.append((res_bf, "#2196F3", "--", "Brute-Force"))
    results.append((res_hop, "#E91E63", "-", "HOP"))

    for res, color, linestyle, label in results:
        J_curve = res["J_curve"]
        ax.plot(np.arange(1, len(J_curve) + 1), J_curve, color=color, ls=linestyle, lw=2.5, label=label, alpha=0.85)
        T_star = res["T_star"]
        ax.scatter([T_star], [J_curve[T_star - 1]], color=color, s=100, zorder=5, edgecolors="white", lw=2)
        ax.annotate(f"$T^*={T_star}$", (T_star, J_curve[T_star - 1]), textcoords="offset points", xytext=(10, 12), fontsize=11, color=color)

    ax.set(xlabel="Horizon $T$", ylabel="$J(T)$", title="Cartpole: Cost vs. Horizon")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    timer_keys = ["linearize", "select", "backward", "forward"]
    colors_bar = ["#80CBC4", "#EF9A9A", "#90CAF9", "#A5D6A7"]
    labels = ["HOP"] if res_bf is None else ["Brute-Force", "HOP"]
    plotted_results = [res_hop] if res_bf is None else [res_bf, res_hop]
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
    fig.savefig(outdir / "cartpole_cost_runtime.png", dpi=160)
    print("Saved optional figure to outputs/cartpole_cost_runtime.png")


def _plot_state_trajectories(plt, outdir, F, res_bf, res_hop):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    style = {
        "BF": ("#2196F3", "-", 4, 0.5, "Brute-Force"),
        "HOP": ("#E91E63", "-", 1.5, 1.0, "HOP"),
    }
    results = []
    if res_bf is not None:
        results.append(("BF", res_bf))
    results.append(("HOP", res_hop))

    state_names = ["$p$ (cart pos.)", r"$\theta$ (pole angle)", r"$\dot p$", r"$\dot\theta$"]
    for i, ax in enumerate(axes.ravel()):
        for tag, res in results:
            X, T_star = res["X"], res["T_star"]
            color, linestyle, linewidth, alpha, label = style[tag]
            t = np.arange(T_star + 1) * F.dt
            ax.plot(t, X[: T_star + 1, i], color=color, ls=linestyle, lw=linewidth, alpha=alpha, label=label)
        ax.axhline(0.0, color="gray", ls=":", alpha=0.4)
        ax.set(xlabel="Time (s)", title=state_names[i])
        if i == 0:
            ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "cartpole_states.png", dpi=160)
    print("Saved optional figure to outputs/cartpole_states.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Cartpole HOP tutorial example.")
    parser.add_argument("--max-iter", type=int, default=15, help="Maximum iLQR iterations for each solver.")
    parser.add_argument("--skip-bruteforce", action="store_true", help="Only run HOP. The brute-force check is slower.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional summary figures.")
    args = parser.parse_args()

    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx = make_cartpole()

    res_bf = None
    time_bf = None
    if not args.skip_bruteforce:
        print("Running Brute-Force...")
        tic = time.perf_counter()
        res_bf = solve_hop(
            F, x0, xg, u_ref, Q, R, alpha, w,
            N, T_min, T_max, method="bruteforce", max_iter=args.max_iter, wrap_idx=wrap_idx
        )
        time_bf = time.perf_counter() - tic
        print(f"  T*={res_bf['T_star']}, J*={_final_cost(res_bf):.2f}, time={time_bf:.1f}s")

    print("\nRunning HOP...")
    tic = time.perf_counter()
    res_hop = solve_hop(
        F, x0, xg, u_ref, Q, R, alpha, w,
        N, T_min, T_max, method="hop", max_iter=args.max_iter, wrap_idx=wrap_idx
    )
    time_hop = time.perf_counter() - tic
    print(f"  T*={res_hop['T_star']}, J*={_final_cost(res_hop):.2f}, time={time_hop:.1f}s")

    if res_bf is not None:
        sel_bf = res_bf["timers"]["select"]
        sel_hop = res_hop["timers"]["select"]
        print(f"\n{'-' * 50}")
        print(f"Total speedup:  {time_bf / max(time_hop, 1e-9):.1f}x")
        print(f"Select speedup: {sel_bf / max(sel_hop, 1e-9):.1f}x  (BF={sel_bf:.2f}s, HOP={sel_hop:.2f}s)")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping the optional figures.")
        return

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    _plot_summary(plt, outdir, res_bf, res_hop)
    _plot_state_trajectories(plt, outdir, F, res_bf, res_hop)


if __name__ == "__main__":
    main()
