from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable

from .simulation import SimulationConfig, TaxiSimulation


@dataclass
class RunResult:
    balance: float
    revenue: float
    move_cost: float
    time_steps: int
    trips: int
    empty_distance: int
    paid_distance: int
    surge_updates: int


def run_batch(
    config: SimulationConfig,
    trips: int | None,
    runs: int,
    seed: int | None = None,
    max_time_steps: int | None = None,
) -> list[RunResult]:
    if runs <= 0:
        raise ValueError("runs must be > 0")
    if trips is None and max_time_steps is None:
        raise ValueError("At least one stopping condition must be provided: trips or max_time_steps")

    results: list[RunResult] = []
    base_seed = config.seed if seed is None else seed
    for i in range(runs):
        run_seed = None if base_seed is None else base_seed + i
        run_cfg = SimulationConfig(
            n_cells=config.n_cells,
            field_dimension=config.field_dimension,
            grid_width=config.grid_width,
            grid_height=config.grid_height,
            surge_update_every_k_steps=config.surge_update_every_k_steps,
            move_cost_per_cell=config.move_cost_per_cell,
            tariff_per_cell=config.tariff_per_cell,
            initial_balance=config.initial_balance,
            randomization_model=config.randomization_model,
            dispatch_mode=config.dispatch_mode,
            strategy_name=config.strategy_name,
            strategy_distance_penalty=config.strategy_distance_penalty,
            strategy_epsilon=config.strategy_epsilon,
            offer_radius=config.offer_radius,
            offer_acceptance_radius=config.offer_acceptance_radius,
            offer_score_slack=config.offer_score_slack,
            offer_probability_scale=config.offer_probability_scale,
            commute_mode=config.commute_mode,
            commute_period_ticks=config.commute_period_ticks,
            analytic_dt=config.analytic_dt,
            analytic_destination_sigma=config.analytic_destination_sigma,
            rl_horizon_steps=config.rl_horizon_steps,
            rl_discount=config.rl_discount,
            policy_iteration_evaluation_sweeps=config.policy_iteration_evaluation_sweeps,
            policy_iteration_max_iterations=config.policy_iteration_max_iterations,
            policy_iteration_tolerance=config.policy_iteration_tolerance,
            q_learning_episodes=config.q_learning_episodes,
            q_learning_alpha=config.q_learning_alpha,
            q_learning_epsilon=config.q_learning_epsilon,
            q_learning_epsilon_decay=config.q_learning_epsilon_decay,
            q_learning_min_epsilon=config.q_learning_min_epsilon,
            q_learning_episode_decisions=config.q_learning_episode_decisions,
            learning_verbose=config.learning_verbose,
            learning_log_every=config.learning_log_every,
            load_policy_path=config.load_policy_path,
            seed=run_seed,
        )
        sim = TaxiSimulation(run_cfg)
        sim.run(trips=trips, max_time_steps=max_time_steps, renderer=None, render_each_tick=False)
        s = sim.stats
        results.append(
            RunResult(
                balance=s.balance,
                revenue=s.total_revenue,
                move_cost=s.total_move_cost,
                time_steps=s.time_steps,
                trips=s.trips_completed,
                empty_distance=s.empty_distance,
                paid_distance=s.paid_distance,
                surge_updates=s.surge_updates,
            )
        )
    return results


def summarize(values: Iterable[float]) -> dict[str, float]:
    data = sorted(values)
    if not data:
        raise ValueError("No values to summarize")
    return {
        "count": float(len(data)),
        "mean": mean(data),
        "std": pstdev(data) if len(data) > 1 else 0.0,
        "min": data[0],
        "p10": _quantile(data, 0.10),
        "p25": _quantile(data, 0.25),
        "p50": _quantile(data, 0.50),
        "p75": _quantile(data, 0.75),
        "p90": _quantile(data, 0.90),
        "max": data[-1],
    }


def histogram(values: Iterable[float], bins: int = 12) -> list[tuple[float, float, int]]:
    data = list(values)
    if not data:
        return []
    lo, hi = min(data), max(data)
    if lo == hi:
        return [(lo, hi, len(data))]

    bins = max(1, bins)
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1

    out: list[tuple[float, float, int]] = []
    for i, c in enumerate(counts):
        start = lo + i * step
        end = start + step
        out.append((start, end, c))
    return out


def format_summary_table(name: str, summary: dict[str, float]) -> str:
    return (
        f"{name} | "
        f"mean={summary['mean']:.2f}, std={summary['std']:.2f}, "
        f"min={summary['min']:.2f}, p25={summary['p25']:.2f}, p50={summary['p50']:.2f}, "
        f"p75={summary['p75']:.2f}, p90={summary['p90']:.2f}, max={summary['max']:.2f}"
    )


def format_histogram(name: str, hist: list[tuple[float, float, int]], width: int = 30) -> str:
    if not hist:
        return f"{name}: (no data)"
    peak = max(c for _, _, c in hist) or 1
    lines = [f"{name} distribution:"]
    for start, end, count in hist:
        bar = "#" * max(1, int((count / peak) * width)) if count > 0 else ""
        lines.append(f"[{start:8.2f}, {end:8.2f}) {count:5d} {bar}")
    return "\n".join(lines)


def _quantile(sorted_values: list[float], q: float) -> float:
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
