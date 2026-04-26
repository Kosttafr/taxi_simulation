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
    parser = argparse.ArgumentParser(
        description="Run 50x50 truncated PI with decomposition and branch-and-bound."
    )
    parser.add_argument("--radius", type=int, default=15, help="Action radius R.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount gamma.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=1.5, help="Tariff tau.")
    parser.add_argument(
        "--destination-radius",
        type=int,
        default=6,
        help="Local destination support radius for making 50x50 practical.",
    )
    parser.add_argument("--evaluation-sweeps", type=int, default=4, help="Policy evaluation sweeps per PI round.")
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum PI improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Evaluation stopping tolerance.")
    parser.add_argument("--eval-trips", type=int, default=600, help="Monte Carlo trips for final policy evaluation.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument("--log-every", type=int, default=1, help="Print every N evaluation sweeps.")
    return parser


def make_mdp(args: argparse.Namespace) -> AnalyticTaxiMDP:
    return AnalyticTaxiMDP(
        n_cells=50 * 50,
        grid_width=50,
        grid_height=50,
        horizon_steps=args.horizon,
        analytic_dt=args.dt,
        destination_sigma=args.sigma,
        move_cost_per_cell=args.cost,
        tariff_per_cell=args.tariff,
        destination_radius=args.destination_radius,
    )


def make_progress_callback(log_every: int):
    def callback(progress: LearningProgress) -> None:
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

    return callback


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


def average_radius_action_count(mdp: AnalyticTaxiMDP, radius: int) -> float:
    return sum(len(mdp.action_candidates(position, radius)) for position in range(mdp.n_cells)) / mdp.n_cells


def main() -> None:
    args = build_parser().parse_args()
    mdp = make_mdp(args)

    print("[stage] 50x50: truncated PI + decomposition + branch-and-bound", flush=True)
    print(
        f"[params] R={args.radius}, H={args.horizon}, gamma={args.discount}, "
        f"sweeps={args.evaluation_sweeps}, max_iterations={args.max_iterations}, "
        f"destination_radius={args.destination_radius}",
        flush=True,
    )
    print(
        f"[params] avg_actions_in_radius={average_radius_action_count(mdp, args.radius):.2f}, "
        f"action_fraction={average_radius_action_count(mdp, args.radius) / mdp.n_cells:.4f}",
        flush=True,
    )

    started = time.perf_counter()
    result = run_smdp_policy_iteration(
        mdp,
        discount=args.discount,
        action_radius=args.radius,
        use_decomposition=True,
        use_branch_and_bound=True,
        evaluation_sweeps=args.evaluation_sweeps,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        progress_callback=make_progress_callback(args.log_every),
    )
    elapsed = time.perf_counter() - started

    total_candidates = result.stats.exact_action_evaluations + result.stats.pruned_action_evaluations
    avg_exact = result.stats.exact_action_evaluations / max(1, mdp.n_cells * mdp.horizon_steps * result.stats.iterations)
    pruned_fraction = result.stats.pruned_action_evaluations / max(1, total_candidates)

    print("[stage] 50x50: evaluating learned policy", flush=True)
    avg_reward, avg_empty = evaluate_policy(mdp, result.policy, trips=args.eval_trips, seed=args.seed)

    print("\nResult")
    print("method | R | time_s | iterations | avg_exact_actions | pruned_frac | avg_reward | avg_empty")
    print("-------+---+--------+------------+-------------------+-------------+------------+----------")
    print(
        f"pi+decomp+bnb | {args.radius} | {elapsed:.3f} | {result.stats.iterations} | "
        f"{avg_exact:.2f} | {pruned_fraction:.3f} | {avg_reward:.3f} | {avg_empty:.3f}"
    )


if __name__ == "__main__":
    main()
