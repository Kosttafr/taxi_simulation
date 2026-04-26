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
    parser = argparse.ArgumentParser(description="Run 15x15 classic full-action SMDP policy iteration.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount gamma.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=1.5, help="Tariff tau.")
    parser.add_argument(
        "--destination-radius",
        type=int,
        default=-1,
        help="Local destination support radius. Default -1 means full destination distribution.",
    )
    parser.add_argument("--evaluation-sweeps", type=int, default=20, help="Policy evaluation sweeps per PI round.")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum PI improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Evaluation stopping tolerance.")
    parser.add_argument("--eval-trips", type=int, default=1000, help="Monte Carlo trips for final policy evaluation.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument("--log-every", type=int, default=1, help="Print every N evaluation sweeps.")
    return parser


def make_mdp(args: argparse.Namespace) -> AnalyticTaxiMDP:
    destination_radius = None if args.destination_radius < 0 else args.destination_radius
    return AnalyticTaxiMDP(
        n_cells=15 * 15,
        grid_width=15,
        grid_height=15,
        horizon_steps=args.horizon,
        analytic_dt=args.dt,
        destination_sigma=args.sigma,
        move_cost_per_cell=args.cost,
        tariff_per_cell=args.tariff,
        destination_radius=destination_radius,
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


def main() -> None:
    args = build_parser().parse_args()
    mdp = make_mdp(args)
    destination_label = "full" if args.destination_radius < 0 else str(args.destination_radius)

    print("[stage] 15x15: classic full-action PI", flush=True)
    print(
        f"[params] states={mdp.n_cells * mdp.horizon_steps}, actions_per_state={mdp.n_cells}, "
        f"H={args.horizon}, gamma={args.discount}, sweeps={args.evaluation_sweeps}, "
        f"max_iterations={args.max_iterations}, destination_radius={destination_label}",
        flush=True,
    )

    started = time.perf_counter()
    result = run_smdp_policy_iteration(
        mdp,
        discount=args.discount,
        action_radius=None,
        use_decomposition=False,
        use_branch_and_bound=False,
        evaluation_sweeps=args.evaluation_sweeps,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        progress_callback=make_progress_callback(args.log_every),
    )
    elapsed = time.perf_counter() - started

    print("[stage] 15x15: evaluating learned policy", flush=True)
    avg_reward, avg_empty = evaluate_policy(mdp, result.policy, trips=args.eval_trips, seed=args.seed)

    print("\nResult")
    print("method | grid | actions_per_state | time_s | iterations | avg_reward | avg_empty")
    print("-------+------+-------------------+--------+------------+------------+----------")
    print(
        f"classic_pi | 15x15 | {mdp.n_cells} | {elapsed:.3f} | {result.stats.iterations} | "
        f"{avg_reward:.3f} | {avg_empty:.3f}"
    )


if __name__ == "__main__":
    main()
