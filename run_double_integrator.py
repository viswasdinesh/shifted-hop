#!/usr/bin/env python3
"""Run the Double Integrator brute-force vs HOP tutorial example."""

import argparse
import time
from pathlib import Path

import numpy as np

from ddp import linearize_trajectory
from lqr import bruteforce_horizon_search, build_augmented_system, hop_horizon_search
from systems import make_double_integrator
from utils import as_terminal_weight, rollout


def main():
    parser = argparse.ArgumentParser(description="Run the Double Integrator HOP tutorial example.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional notebook-style figure.")
    args = parser.parse_args()

    F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max = make_double_integrator()
    n = x0.size
    Qf = as_terminal_weight(alpha, n)
    dt = F.dt
    A_di = np.array([[1.0, dt], [0.0, 1.0]])
    B_di = np.array([[0.0], [dt]])
    A_list_di = [A_di] * N
    B_list_di = [B_di] * N

    tic = time.perf_counter()
    J_bf = bruteforce_horizon_search(A_list_di, B_list_di, Q, R, Qf, w, x0, T_max)
    bf_time = time.perf_counter() - tic
    T_bf = int(np.argmin(J_bf[T_min - 1:T_max]) + T_min)

    X_di = rollout(F, x0, np.tile(u_ref.reshape(1, -1), (N, 1)))
    U_di = np.tile(u_ref.reshape(1, -1), (N, 1))
    A_list, B_list = linearize_trajectory(F, X_di, U_di)
    A_aug, B_aug, Q_aug, R_mat, z0, QT_list = build_augmented_system(
        F, A_list, B_list, X_di, U_di, xg, u_ref, Q, R, w, alpha
    )

    tic = time.perf_counter()
    J_hop = hop_horizon_search(A_aug, B_aug, Q_aug, R_mat, z0, QT_list, T_max)
    hop_time = time.perf_counter() - tic
    T_hop = int(np.argmin(J_hop[T_min - 1:T_max]) + T_min)
    J_star_hop = J_hop[T_hop - 1]

    print(f"{'Method':<16} {'T*':>5} {'J*':>12} {'Time':>12}")
    print("-" * 48)
    print(f"{'Brute-Force':<16} {T_bf:>5} {J_bf[T_bf - 1]:>12.4f} {bf_time * 1000:>10.1f} ms")
    print(f"{'HOP':<16} {T_hop:>5} {J_star_hop:>12.4f} {hop_time * 1000:>10.1f} ms")
    print("-" * 48)
    print(f"T* match:  {T_bf == T_hop}")
    print(f"J* diff:   {abs(J_bf[T_bf - 1] - J_star_hop):.2e}")
    print(f"Speedup:   {bf_time / max(hop_time, 1e-9):.1f}x")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping the optional figure.")
        return

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    T_axis = np.arange(1, T_max + 1)
    ax.plot(T_axis, J_bf, "#2196F3", lw=3, ls="--", label="Brute-Force", alpha=0.9)
    ax.plot(T_axis, J_hop, "#E91E63", lw=2.5, ls="-", label="HOP", alpha=0.8)
    ax.scatter([T_bf], [J_bf[T_bf - 1]], color="#2196F3", s=100, zorder=5, edgecolors="white", linewidth=2)
    ax.scatter([T_hop], [J_star_hop], color="#E91E63", s=80, zorder=5, marker="s", edgecolors="white", linewidth=2)
    ax.set(xlabel="Horizon $T$", ylabel="$J(T)$", title="$J(T)$ curves (BF vs HOP)")
    ax.legend(fontsize=11)

    ax2 = axes[1]
    bars = ax2.bar(
        ["Brute-Force", "HOP"],
        [bf_time * 1000, hop_time * 1000],
        color=["#2196F3", "#E91E63"],
        width=0.5,
        edgecolor="white",
    )
    ax2.set_ylabel("Time (ms)")
    ax2.set_title(f"Runtime ({bf_time / max(hop_time, 1e-9):.0f}x speedup)")
    for bar, val in zip(bars, [bf_time * 1000, hop_time * 1000]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(outdir / "double_integrator_bf_vs_hop.png", dpi=160)
    print("Saved optional figure to outputs/double_integrator_bf_vs_hop.png")


if __name__ == "__main__":
    main()
