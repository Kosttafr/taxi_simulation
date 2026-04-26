from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxi_sim.learning import AnalyticTaxiMDP, LearnedPolicy, LearningProgress, load_learned_policy, run_smdp_policy_iteration


@dataclass(frozen=True)
class TrajectoryResult:
    strategy: str
    run: int
    balance: float
    revenue: float
    move_cost: float
    empty_distance: int
    paid_distance: int
    no_passenger_time: int
    time_steps: int
    trips: int


class GreedyPolicy:
    def __init__(self, mdp: AnalyticTaxiMDP, *, arrival_aware: bool = False) -> None:
        self.mdp = mdp
        self.arrival_aware = arrival_aware

    def action_for(self, position: int, time_step: int) -> int:
        phase = time_step % self.mdp.horizon_steps
        return max(
            self.mdp.iter_actions(),
            key=lambda action: (
                self.mdp.surge(
                    action,
                    (phase + self.mdp.grid_distance(position, action)) % self.mdp.horizon_steps
                    if self.arrival_aware
                    else phase,
                ),
                -self.mdp.grid_distance(position, action),
                -action,
            ),
        )


class SmartPolicy:
    def __init__(self, mdp: AnalyticTaxiMDP, *, arrival_aware: bool = False) -> None:
        self.mdp = mdp
        self.arrival_aware = arrival_aware
        self.expected_distances = {
            action: sum(mdp.grid_distance(action, z) * p for z, p in mdp.destination_distribution(action))
            for action in mdp.iter_actions()
        }

    def action_for(self, position: int, time_step: int) -> int:
        phase = time_step % self.mdp.horizon_steps
        return max(
            self.mdp.iter_actions(),
            key=lambda action: (
                (
                    self.mdp.surge(
                        action,
                        (phase + self.mdp.grid_distance(position, action)) % self.mdp.horizon_steps
                        if self.arrival_aware
                        else phase,
                    ) * self.mdp.tariff_per_cell
                    - self.mdp.move_cost_per_cell
                ) * self.expected_distances[action]
                - self.mdp.move_cost_per_cell * self.mdp.grid_distance(position, action),
                -action,
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch comparison of greedy, smart, and optimal strategies.")
    parser.add_argument("--cells", type=int, default=50, help="1D field size.")
    parser.add_argument("--width", type=int, default=None, help="Grid width. If set, overrides --cells.")
    parser.add_argument("--height", type=int, default=1, help="Grid height.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument("--discount", type=float, default=0.97, help="Discount gamma for optimal PI.")
    parser.add_argument("--dt", type=float, default=0.05, help="Analytic time step.")
    parser.add_argument("--sigma", type=float, default=100.0, help="Destination Gaussian sigma.")
    parser.add_argument("--cost", type=float, default=1.0, help="Move cost c.")
    parser.add_argument("--tariff", type=float, default=1.5, help="Tariff tau.")
    parser.add_argument("--radius", type=int, default=15, help="Optimal policy action radius. Use -1 for full actions.")
    parser.add_argument("--load-optimal-policy", default=None, help="Load optimal policy JSON instead of training it.")
    parser.add_argument("--runs", type=int, default=200, help="Number of trajectories per strategy.")
    parser.add_argument("--trips", type=int, default=None, help="Optional completed-trip limit per trajectory.")
    parser.add_argument("--max-time-steps", type=int, default=1000, help="Time-step limit per trajectory.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--evaluation-sweeps", type=int, default=80, help="Optimal PI evaluation sweeps.")
    parser.add_argument("--max-iterations", type=int, default=80, help="Optimal PI maximum improvement rounds.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Optimal PI evaluation tolerance.")
    parser.add_argument("--verbose", action="store_true", help="Print optimal PI progress and batch progress.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N PI evaluation sweeps.")
    parser.add_argument("--output-dir", default="figures", help="Directory for PNG figure and CSV outputs.")
    parser.add_argument(
        "--arrival-aware-heuristics",
        action="store_true",
        help="Make greedy/smart evaluate surge at pickup arrival phase k+d(x,a).",
    )
    return parser


def make_mdp(args: argparse.Namespace) -> AnalyticTaxiMDP:
    width = args.cells if args.width is None else args.width
    height = args.height
    return AnalyticTaxiMDP(
        n_cells=width * height,
        grid_width=width,
        grid_height=height,
        horizon_steps=args.horizon,
        analytic_dt=args.dt,
        destination_sigma=args.sigma,
        move_cost_per_cell=args.cost,
        tariff_per_cell=args.tariff,
    )


def make_progress_callback(enabled: bool, log_every: int):
    def callback(progress: LearningProgress) -> None:
        if not enabled:
            return
        if progress.phase == "evaluation":
            if progress.iteration % max(1, log_every) == 0:
                print(f"[optimal eval] sweep={progress.iteration} loss={progress.loss:.6e}", flush=True)
        else:
            print(
                f"[optimal improve] iter={progress.iteration} "
                f"eval_loss={progress.loss:.6e} "
                f"sweeps={progress.sweeps_used} "
                f"policy_changes={progress.policy_changes}",
                flush=True,
            )

    return callback if enabled else None


def simulate_trajectory(
    mdp: AnalyticTaxiMDP,
    policy,
    *,
    trips: int | None,
    max_time_steps: int,
    seed: int,
    strategy: str,
    run: int,
) -> TrajectoryResult:
    rng = random.Random(seed)
    position = rng.randrange(mdp.n_cells)
    elapsed_time = rng.randrange(mdp.horizon_steps)
    revenue = 0.0
    move_cost = 0.0
    empty_distance_total = 0
    paid_distance_total = 0
    no_passenger_time = 0
    completed_trips = 0

    while elapsed_time < max_time_steps and (trips is None or completed_trips < trips):
        action = policy.action_for(position, elapsed_time)
        empty_distance = mdp.grid_distance(position, action)
        pickup_phase = (elapsed_time + empty_distance) % mdp.horizon_steps
        destinations, weights = zip(*mdp.destination_distribution(action), strict=True)
        destination = rng.choices(destinations, weights=weights, k=1)[0]
        paid_distance = mdp.grid_distance(action, destination)
        duration = empty_distance + paid_distance
        if elapsed_time + duration > max_time_steps:
            break
        trip_revenue = mdp.surge(action, pickup_phase) * mdp.tariff_per_cell * paid_distance
        trip_cost = mdp.move_cost_per_cell * (empty_distance + paid_distance)
        revenue += trip_revenue
        move_cost += trip_cost
        empty_distance_total += empty_distance
        paid_distance_total += paid_distance
        no_passenger_time += empty_distance
        completed_trips += 1
        position = destination
        elapsed_time += duration

    return TrajectoryResult(
        strategy=strategy,
        run=run,
        balance=revenue - move_cost,
        revenue=revenue,
        move_cost=move_cost,
        empty_distance=empty_distance_total,
        paid_distance=paid_distance_total,
        no_passenger_time=no_passenger_time,
        time_steps=elapsed_time,
        trips=completed_trips,
    )


def summarize(results: list[TrajectoryResult], strategy: str) -> dict[str, float]:
    subset = [result for result in results if result.strategy == strategy]
    n = len(subset)
    return {
        "balance_mean": sum(r.balance for r in subset) / n,
        "revenue_mean": sum(r.revenue for r in subset) / n,
        "cost_mean": sum(r.move_cost for r in subset) / n,
        "no_passenger_fraction_mean": sum(
            0.0 if r.time_steps == 0 else r.no_passenger_time / r.time_steps for r in subset
        ) / n,
        "empty_mean": sum(r.empty_distance for r in subset) / n,
        "paid_mean": sum(r.paid_distance for r in subset) / n,
        "trips_mean": sum(r.trips for r in subset) / n,
    }


def write_csv(path: Path, results: list[TrajectoryResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "run",
                "balance",
                "revenue",
                "move_cost",
                "empty_distance",
                "paid_distance",
                "no_passenger_time",
                "time_steps",
                "trips",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def _kde_curve(values: list[float], xs: list[float]) -> list[float]:
    if not values:
        return [0.0 for _ in xs]
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / max(1, n - 1)
    std = variance ** 0.5
    bandwidth = 1.06 * std * (n ** -0.2) if std > 0.0 else 1.0
    bandwidth = max(bandwidth, 1e-6)
    normalizer = n * bandwidth * ((2.0 * 3.141592653589793) ** 0.5)
    densities = []
    for x in xs:
        densities.append(sum(pow(2.718281828459045, -0.5 * ((x - value) / bandwidth) ** 2) for value in values) / normalizer)
    return densities


def plot_income_distributions(path: Path, results: list[TrajectoryResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    all_incomes = [result.balance for result in results]
    low = min(all_incomes)
    high = max(all_incomes)
    padding = 0.08 * max(1.0, high - low)
    xs = [low - padding + (high - low + 2.0 * padding) * idx / 399 for idx in range(400)]
    for strategy in ["greedy", "smart", "optimal"]:
        incomes = [result.balance for result in results if result.strategy == strategy]
        densities = _kde_curve(incomes, xs)
        ax.plot(xs, densities, linewidth=2.2, label=strategy)
        ax.fill_between(xs, densities, alpha=0.12)
    ax.set_xlabel("final balance")
    ax.set_ylabel("density")
    ax.set_title("Strategy final balance distributions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    mdp = make_mdp(args)
    action_radius = None if args.radius < 0 else args.radius
    output_dir = Path(args.output_dir)

    if args.load_optimal_policy:
        print(f"[stage] loading optimal policy from {args.load_optimal_policy}", flush=True)
        optimal_policy = load_learned_policy(args.load_optimal_policy)
        if optimal_policy.horizon_steps != mdp.horizon_steps:
            raise ValueError(
                f"Loaded policy horizon={optimal_policy.horizon_steps}, but simulation horizon={mdp.horizon_steps}"
            )
    else:
        print(
            f"[stage] training optimal policy: grid={mdp.grid_width}x{mdp.grid_height}, "
            f"R={'full' if action_radius is None else action_radius}, "
            f"H={args.horizon}, gamma={args.discount}, sigma={args.sigma}",
            flush=True,
        )
        started = time.perf_counter()
        optimal_result = run_smdp_policy_iteration(
            mdp,
            discount=args.discount,
            action_radius=action_radius,
            use_decomposition=True,
            reward_timing="transition",
            evaluation_sweeps=args.evaluation_sweeps,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            progress_callback=make_progress_callback(args.verbose, args.log_every),
        )
        optimal_policy = optimal_result.policy
        print(
            f"[stage] optimal trained in {time.perf_counter() - started:.3f}s; "
            f"iterations={optimal_result.stats.iterations}; sweeps={optimal_result.stats.evaluation_sweeps}",
            flush=True,
        )

    policies = {
        "greedy": GreedyPolicy(mdp, arrival_aware=args.arrival_aware_heuristics),
        "smart": SmartPolicy(mdp, arrival_aware=args.arrival_aware_heuristics),
        "optimal": optimal_policy,
    }

    results: list[TrajectoryResult] = []
    for strategy, policy in policies.items():
        print(f"[stage] simulating {strategy}", flush=True)
        for run in range(args.runs):
            if args.verbose and (run + 1) % max(1, args.runs // 10) == 0:
                print(f"[batch] {strategy}: run {run + 1}/{args.runs}", flush=True)
            results.append(
                simulate_trajectory(
                    mdp,
                    policy,
                    trips=args.trips,
                    max_time_steps=args.max_time_steps,
                    seed=args.seed + 10000 * list(policies).index(strategy) + run,
                    strategy=strategy,
                    run=run,
                )
            )

    csv_path = output_dir / "strategy_comparison_batch.csv"
    png_path = output_dir / "strategy_income_distributions.png"
    write_csv(csv_path, results)
    plot_income_distributions(png_path, results)

    print("\nSummary")
    print(
        "strategy | balance_mean | revenue_mean | cost_mean | no_passenger_frac | trips_mean"
    )
    print(
        "---------+--------------+--------------+-----------+-------------------+-----------"
    )
    for strategy in ["greedy", "smart", "optimal"]:
        s = summarize(results, strategy)
        print(
            f"{strategy:>8} | {s['balance_mean']:12.3f} | {s['revenue_mean']:12.3f} | "
            f"{s['cost_mean']:9.3f} | {s['no_passenger_fraction_mean']:17.3f} | {s['trips_mean']:9.3f}"
        )
    print(f"\nSaved {csv_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
