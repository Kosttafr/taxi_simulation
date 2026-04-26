from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxi_sim.learning import AnalyticTaxiMDP, LearningProgress, run_policy_iteration, run_smdp_policy_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and plot 1D SMDP policy, value, and surge heatmaps.")
    parser.add_argument("--cells", type=int, default=50, help="Number of 1D cells.")
    parser.add_argument("--radius", type=int, default=15, help="Action radius R. Use -1 for full actions.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount gamma.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=100.0, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=4.0, help="Tariff tau.")
    parser.add_argument(
        "--algorithm",
        choices=["smdp", "legacy"],
        default="smdp",
        help="Bellman model to train: smdp uses gamma^duration and arrival surge; legacy matches old policy_iteration.",
    )
    parser.add_argument(
        "--reward-timing",
        choices=["discounted_rewards", "transition"],
        default="transition",
        help=(
            "For SMDP only: discounted_rewards discounts pickup reward by approach time; "
            "transition uses R + gamma^T V and does not discount transition reward."
        ),
    )
    parser.add_argument("--evaluation-sweeps", type=int, default=50, help="Policy evaluation sweeps per PI round.")
    parser.add_argument("--max-iterations", type=int, default=40, help="Maximum PI improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Evaluation stopping tolerance.")
    parser.add_argument("--verbose", action="store_true", help="Print policy iteration progress.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N evaluation sweeps in verbose mode.")
    parser.add_argument(
        "--output",
        default="saved_policies/smdp_experiments/policy_value_surge_1d_heatmap.png",
        help="Output PNG path.",
    )
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
    return AnalyticTaxiMDP(
        n_cells=args.cells,
        grid_width=args.cells,
        grid_height=1,
        horizon_steps=args.horizon,
        analytic_dt=args.dt,
        destination_sigma=args.sigma,
        move_cost_per_cell=args.cost,
        tariff_per_cell=args.tariff,
    )


def main() -> None:
    args = build_parser().parse_args()
    action_radius = None if args.radius < 0 else args.radius
    mdp = make_mdp(args)

    print(
        f"[stage] training 1D {args.algorithm} policy: N={args.cells}, H={args.horizon}, "
        f"R={'full' if action_radius is None else action_radius}",
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
    print(
        f"[stage] trained in {elapsed:.3f}s; {stats_text}",
        flush=True,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    actions = [
        [policy.action_for(position, phase) for phase in range(args.horizon)]
        for position in range(args.cells)
    ]
    values = [
        [policy.values[(position, phase)] for phase in range(args.horizon)]
        for position in range(args.cells)
    ]
    surges = [
        [mdp.surge(position, phase) for phase in range(args.horizon)]
        for position in range(args.cells)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    image0 = axes[0].imshow(actions, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Chosen pickup action a")
    axes[0].set_xlabel("time phase k")
    axes[0].set_ylabel("driver position x")
    colorbar0 = fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar0.locator = MaxNLocator(integer=True, nbins=8)
    colorbar0.update_ticks()

    image1 = axes[1].imshow(values, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Value V(x,k)")
    axes[1].set_xlabel("time phase k")
    axes[1].set_ylabel("driver position x")
    colorbar1 = fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    image2 = axes[2].imshow(surges, origin="lower", aspect="auto", cmap="plasma")
    axes[2].set_title("Surge S(x,k)")
    axes[2].set_xlabel("time phase k")
    axes[2].set_ylabel("position x")
    colorbar2 = fig.colorbar(image2, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    fig.suptitle(
        f"1D SMDP policy/value/surge heatmaps, N={args.cells}, H={args.horizon}, "
        f"algorithm={args.algorithm}, R={'full' if action_radius is None else action_radius}"
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
