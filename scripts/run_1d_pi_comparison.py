from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxi_sim.learning import AnalyticTaxiMDP, LearnedPolicy, LearningProgress, run_smdp_policy_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare classic and modified SMDP PI on a 1D field.")
    parser.add_argument("--cells", type=int, default=50, help="Number of 1D cells.")
    parser.add_argument("--radius", type=int, default=15, help="Truncated action radius R.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount gamma.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=1.5, help="Tariff tau.")
    parser.add_argument("--evaluation-sweeps", type=int, default=50, help="Policy evaluation sweeps per PI round.")
    parser.add_argument("--max-iterations", type=int, default=40, help="Maximum PI improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Evaluation stopping tolerance.")
    parser.add_argument("--eval-trips", type=int, default=1000, help="Monte Carlo trips for final policy evaluation.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument("--verbose", action="store_true", help="Print PI evaluation/improvement progress.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N evaluation sweeps in verbose mode.")
    return parser


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


def make_progress_callback(enabled: bool, log_every: int, label: str):
    def callback(progress: LearningProgress) -> None:
        if not enabled:
            return
        if progress.phase == "evaluation":
            if progress.iteration % max(1, log_every) == 0:
                print(f"[{label} eval] sweep={progress.iteration} loss={progress.loss:.6e}", flush=True)
        else:
            print(
                f"[{label} improve] iter={progress.iteration} "
                f"eval_loss={progress.loss:.6e} "
                f"sweeps={progress.sweeps_used} "
                f"policy_changes={progress.policy_changes}",
                flush=True,
            )

    return callback if enabled else None


def evaluate_policy(mdp: AnalyticTaxiMDP, policy: LearnedPolicy, *, trips: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    position = 0
    time_step = 0
    total_reward = 0.0
    total_empty = 0
    for _ in range(trips):
        action = policy.action_for(position, time_step)
        empty_distance = mdp.grid_distance(position, action)
        pickup_phase = (time_step + empty_distance) % mdp.horizon_steps
        destinations, weights = zip(*mdp.destination_distribution(action), strict=True)
        destination = rng.choices(destinations, weights=weights, k=1)[0]
        trip_distance = mdp.grid_distance(action, destination)
        duration = empty_distance + trip_distance
        total_reward += (
            mdp.surge(action, pickup_phase) * mdp.tariff_per_cell * trip_distance
            - mdp.move_cost_per_cell * duration
        )
        total_empty += empty_distance
        position = destination
        time_step = (time_step + duration) % mdp.horizon_steps
    return total_reward / trips, total_empty / trips


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    text_rows = [[str(item) for item in row] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in text_rows))
        for idx in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(" | ".join(item.rjust(widths[idx]) for idx, item in enumerate(row)))


def main() -> None:
    args = build_parser().parse_args()
    mdp = make_mdp(args)
    configs = [
        ("classic PI", None, False, False),
        ("truncated PI", args.radius, False, False),
        ("truncated PI + decomposition", args.radius, True, False),
        ("truncated PI + decomposition + BnB", args.radius, True, True),
    ]

    print("[stage] 1D PI comparison", flush=True)
    print(
        f"[params] N={args.cells}, H={args.horizon}, R={args.radius}, gamma={args.discount}, "
        f"sweeps={args.evaluation_sweeps}, max_iterations={args.max_iterations}",
        flush=True,
    )

    full_values: dict[tuple[int, int], float] | None = None
    rows = []
    for label, radius, use_decomposition, use_bnb in configs:
        print(f"[stage] training {label}", flush=True)
        started = time.perf_counter()
        result = run_smdp_policy_iteration(
            mdp,
            discount=args.discount,
            action_radius=radius,
            use_decomposition=use_decomposition,
            use_branch_and_bound=use_bnb,
            evaluation_sweeps=args.evaluation_sweeps,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            progress_callback=make_progress_callback(args.verbose, args.log_every, label),
        )
        elapsed = time.perf_counter() - started
        avg_reward, avg_empty = evaluate_policy(mdp, result.policy, trips=args.eval_trips, seed=args.seed)

        if full_values is None:
            full_values = result.policy.values
            value_error = 0.0
        else:
            value_error = max(abs(result.policy.values[state] - full_values[state]) for state in mdp.iter_states())

        total_candidates = result.stats.exact_action_evaluations + result.stats.pruned_action_evaluations
        avg_exact = result.stats.exact_action_evaluations / max(1, mdp.n_cells * mdp.horizon_steps * result.stats.iterations)
        pruned_fraction = result.stats.pruned_action_evaluations / max(1, total_candidates)
        action_count = args.cells if radius is None else min(2 * radius + 1, args.cells)

        rows.append(
            [
                label,
                "full" if radius is None else radius,
                action_count,
                f"{elapsed:.3f}",
                result.stats.iterations,
                f"{avg_exact:.2f}",
                f"{pruned_fraction:.3f}",
                f"{value_error:.3e}",
                f"{avg_reward:.3f}",
                f"{avg_empty:.3f}",
            ]
        )

    print("\nResult")
    print_table(
        [
            "method",
            "R",
            "max_actions",
            "time_s",
            "iters",
            "avg_exact_actions",
            "pruned_frac",
            "value_error_inf",
            "avg_reward",
            "avg_empty",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
