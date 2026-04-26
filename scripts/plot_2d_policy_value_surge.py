from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxi_sim.learning import (
    AnalyticTaxiMDP,
    LearningProgress,
    run_policy_iteration,
    run_smdp_policy_iteration,
    save_learned_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and plot 2D SMDP policy, value, and surge maps.")
    parser.add_argument("--side", type=int, default=15, help="2D grid side length.")
    parser.add_argument("--phase", type=int, default=0, help="Time phase k to plot when --phases is not provided.")
    parser.add_argument("--phases", type=int, nargs="*", default=None, help="Time phases k to plot.")
    parser.add_argument("--radius", type=int, default=15, help="Action radius R. Use -1 for full actions.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.97, help="Discount gamma.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=100.0, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=1.5, help="Tariff tau.")
    parser.add_argument(
        "--destination-radius",
        type=int,
        default=-1,
        help="Optional local destination support radius. Default -1 means full support.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["smdp", "legacy"],
        default="smdp",
        help="Bellman model to train.",
    )
    parser.add_argument(
        "--reward-timing",
        choices=["discounted_rewards", "transition"],
        default="transition",
        help="For SMDP only: transition uses R + gamma^T V.",
    )
    parser.add_argument("--evaluation-sweeps", type=int, default=40, help="Policy evaluation sweeps per PI round.")
    parser.add_argument("--max-iterations", type=int, default=40, help="Maximum PI improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Evaluation stopping tolerance.")
    parser.add_argument("--verbose", action="store_true", help="Print policy iteration progress.")
    parser.add_argument("--log-every", type=int, default=5, help="Print every N evaluation sweeps in verbose mode.")
    parser.add_argument(
        "--output",
        default="saved_policies/smdp_experiments/policy_value_surge_2d.png",
        help="Output PNG path.",
    )
    parser.add_argument("--save-policy", default=None, help="Optional path to save the learned policy JSON.")
    return parser


def make_progress_callback(enabled: bool, log_every: int):
    def callback(progress: LearningProgress) -> None:
        if not enabled:
            return
        if progress.phase == "evaluation":
            if progress.iteration % max(1, log_every) == 0:
                print(f"[eval] sweep={progress.iteration} loss={progress.loss:.6e}", flush=True)
        else:
            print(
                f"[improve] iter={progress.iteration} "
                f"eval_loss={progress.loss:.6e} "
                f"sweeps={progress.sweeps_used} "
                f"policy_changes={progress.policy_changes}",
                flush=True,
            )

    return callback if enabled else None


def make_mdp(args: argparse.Namespace) -> AnalyticTaxiMDP:
    destination_radius = None if args.destination_radius < 0 else args.destination_radius
    return AnalyticTaxiMDP(
        n_cells=args.side * args.side,
        grid_width=args.side,
        grid_height=args.side,
        horizon_steps=args.horizon,
        analytic_dt=args.dt,
        destination_sigma=args.sigma,
        move_cost_per_cell=args.cost,
        tariff_per_cell=args.tariff,
        destination_radius=destination_radius,
    )


def main() -> None:
    args = build_parser().parse_args()
    action_radius = None if args.radius < 0 else args.radius
    phases = [phase % args.horizon for phase in (args.phases if args.phases else [args.phase])]
    mdp = make_mdp(args)

    print(
        f"[stage] training 2D {args.algorithm} policy: grid={args.side}x{args.side}, "
        f"H={args.horizon}, phases={phases}, R={'full' if action_radius is None else action_radius}",
        flush=True,
    )
    started = time.perf_counter()
    if args.algorithm == "legacy":
        if action_radius is not None:
            print("[warn] legacy policy_iteration ignores --radius and uses full action set", flush=True)
        policy = run_policy_iteration(
            mdp,
            discount=args.discount,
            evaluation_sweeps=args.evaluation_sweeps,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            progress_callback=make_progress_callback(args.verbose, args.log_every),
        )
        stats_text = "iterations=n/a; evaluation_sweeps=n/a"
    else:
        result = run_smdp_policy_iteration(
            mdp,
            discount=args.discount,
            action_radius=action_radius,
            use_decomposition=True,
            reward_timing=args.reward_timing,
            evaluation_sweeps=args.evaluation_sweeps,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            progress_callback=make_progress_callback(args.verbose, args.log_every),
        )
        policy = result.policy
        stats_text = f"iterations={result.stats.iterations}; evaluation_sweeps={result.stats.evaluation_sweeps}"
    elapsed = time.perf_counter() - started
    print(f"[stage] trained in {elapsed:.3f}s; {stats_text}", flush=True)
    if args.save_policy is not None:
        save_path = Path(args.save_policy)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_learned_policy(policy, str(save_path))
        print(f"Saved policy {save_path}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    side = args.side
    fig, axes = plt.subplots(len(phases), 3, figsize=(18, 5.2 * len(phases)), squeeze=False, constrained_layout=True)

    for row_idx, phase in enumerate(phases):
        actions_x = []
        actions_y = []
        dxs = []
        dys = []
        action_distances = []
        values = [[0.0 for _ in range(side)] for _ in range(side)]
        surges = [[0.0 for _ in range(side)] for _ in range(side)]

        for idx in range(mdp.n_cells):
            x, y = mdp.index_to_coord(idx)
            action = policy.action_for(idx, phase)
            ax, ay = mdp.index_to_coord(action)
            raw_dx = ax - x
            raw_dy = ay - y
            norm = math.hypot(raw_dx, raw_dy)
            actions_x.append(x)
            actions_y.append(y)
            dxs.append(0.0 if norm == 0.0 else raw_dx / norm)
            dys.append(0.0 if norm == 0.0 else raw_dy / norm)
            action_distances.append(mdp.grid_distance(idx, action))
            values[y][x] = policy.values[(idx, phase)]
            surges[y][x] = mdp.surge(idx, phase)

        ax0, ax1, ax2 = axes[row_idx]
        ax0.set_facecolor("#f7f7f7")
        arrows = ax0.quiver(
            actions_x,
            actions_y,
            dxs,
            dys,
            action_distances,
            angles="xy",
            scale_units="xy",
            scale=1.6,
            width=0.004,
            cmap="coolwarm",
        )
        ax0.set_title(f"Policy direction, color=d(x,a), k={phase}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_xlim(-0.5, side - 0.5)
        ax0.set_ylim(-0.5, side - 0.5)
        ax0.set_aspect("equal")
        arrow_bar = fig.colorbar(arrows, ax=ax0, fraction=0.046, pad=0.04)
        arrow_bar.set_label("step length")
        arrow_bar.locator = MaxNLocator(integer=True, nbins=6)
        arrow_bar.update_ticks()

        image1 = ax1.imshow(values, origin="lower", cmap="magma")
        ax1.set_title(f"Value V(x,y,k), k={phase}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        colorbar1 = fig.colorbar(image1, ax=ax1, fraction=0.046, pad=0.04)
        colorbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        image2 = ax2.imshow(surges, origin="lower", cmap="plasma")
        ax2.set_title(f"Surge S(x,y,k), k={phase}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        colorbar2 = fig.colorbar(image2, ax=ax2, fraction=0.046, pad=0.04)
        colorbar2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        for ax in axes[row_idx]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    fig.suptitle(
        f"2D policy/value/surge, {args.side}x{args.side}, phases={phases}, "
        f"algorithm={args.algorithm}, R={'full' if action_radius is None else action_radius}"
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
