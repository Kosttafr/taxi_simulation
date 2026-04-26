from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxi_sim.learning import (
    AnalyticTaxiMDP,
    LearnedPolicy,
    LearningProgress,
    decomposed_bellman_update_smdp,
    naive_bellman_update_smdp,
    radius_bound_parameters_1d,
    run_smdp_policy_iteration,
)


H = 40
DT = 0.05
SIGMA = 1.5
GAMMA = 0.9
COST = 1.0
TARIFF = 1.5
VERBOSE = False
LOG_EVERY = 10


def make_mdp(width: int, height: int = 1, destination_radius: int | None = None) -> AnalyticTaxiMDP:
    return AnalyticTaxiMDP(
        n_cells=width * height,
        grid_width=width,
        grid_height=height,
        horizon_steps=H,
        analytic_dt=DT,
        destination_sigma=SIGMA,
        move_cost_per_cell=COST,
        tariff_per_cell=TARIFF,
        destination_radius=destination_radius,
    )


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


def progress_callback(progress: LearningProgress) -> None:
    if not VERBOSE:
        return
    if progress.phase == "evaluation":
        if progress.iteration % max(1, LOG_EVERY) == 0:
            print(f"[eval] sweep={progress.iteration} loss={progress.loss:.6e}")
    else:
        print(
            f"[improve] iter={progress.iteration} "
            f"eval_loss={progress.loss:.6e} "
            f"sweeps={progress.sweeps_used} "
            f"policy_changes={progress.policy_changes}"
        )


def learning_progress_callback():
    return progress_callback if VERBOSE else None


def experiment_1() -> None:
    rows = []
    for n_cells in [10, 15, 20, 30, 50, 100, 200, 500, 1000]:
        p = radius_bound_parameters_1d(n_cells=n_cells, discount=GAMMA, q=TARIFF / COST)
        rows.append(
            [
                n_cells,
                f"{p['C']:.3f}",
                f"{p['L']:.3f}",
                f"{p['U']:.3f}",
                int(p["r_cut"]),
                int(p["R_star"]),
                f"{p['action_fraction']:.4f}",
            ]
        )
    print("\nExperiment 1. Radius bound parameters")
    print_table(["N", "C", "L", "U", "r_cut", "R*", "action_frac"], rows)


def experiment_2() -> None:
    mdp = make_mdp(50)
    print("\nExperiment 2. Full PI vs truncated PI, 1D")
    start = time.perf_counter()
    full = run_smdp_policy_iteration(
        mdp,
        discount=GAMMA,
        evaluation_sweeps=50,
        max_iterations=40,
        tolerance=1e-8,
        progress_callback=learning_progress_callback(),
    )
    full_time = time.perf_counter() - start
    full_reward, _empty = evaluate_policy(mdp, full.policy)

    rows = [["full", 50, f"{full_time:.3f}", "0.000e+00", f"{full_reward:.3f}"]]
    for radius in [3, 5, 7, 10, 15]:
        start = time.perf_counter()
        result = run_smdp_policy_iteration(
            mdp,
            discount=GAMMA,
            action_radius=radius,
            evaluation_sweeps=50,
            max_iterations=40,
            tolerance=1e-8,
            progress_callback=learning_progress_callback(),
        )
        elapsed = time.perf_counter() - start
        error = max(abs(result.policy.values[state] - full.policy.values[state]) for state in mdp.iter_states())
        reward, _empty = evaluate_policy(mdp, result.policy)
        action_count = min(2 * radius + 1, mdp.n_cells)
        rows.append([radius, action_count, f"{elapsed:.3f}", f"{error:.3e}", f"{reward:.3f}"])
    print_table(["R", "actions", "time_s", "value_error_inf", "avg_reward"], rows)


def experiment_3() -> None:
    print("\nExperiment 3. Naive Bellman update vs decomposition")
    rows = []
    for n_cells in [20, 30, 40, 50]:
        mdp = make_mdp(n_cells)
        values = {(position, time_step): 0.01 * ((position + 3 * time_step) % 17) for position, time_step in mdp.iter_states()}

        start = time.perf_counter()
        naive = naive_bellman_update_smdp(mdp, values, discount=GAMMA)
        naive_time = time.perf_counter() - start

        start = time.perf_counter()
        decomposed = decomposed_bellman_update_smdp(mdp, values, discount=GAMMA)
        decomposed_time = time.perf_counter() - start

        diff = max(abs(naive[state] - decomposed[state]) for state in naive)
        speedup = naive_time / decomposed_time if decomposed_time > 0 else float("inf")
        rows.append([n_cells, f"{naive_time:.3f}", f"{decomposed_time:.3f}", f"{speedup:.2f}", f"{diff:.3e}"])
    print_table(["N", "naive_s", "decomp_s", "speedup", "max_diff"], rows)


def experiment_4() -> None:
    mdp = make_mdp(50)
    print("\nExperiment 4. Branch-and-bound, N=50, R=15")
    rows = []
    for label, use_bnb in [("no_bnb", False), ("bnb", True)]:
        start = time.perf_counter()
        result = run_smdp_policy_iteration(
            mdp,
            discount=GAMMA,
            action_radius=15,
            use_branch_and_bound=use_bnb,
            evaluation_sweeps=50,
            max_iterations=40,
            tolerance=1e-8,
            progress_callback=learning_progress_callback(),
        )
        elapsed = time.perf_counter() - start
        total = result.stats.exact_action_evaluations + result.stats.pruned_action_evaluations
        avg_exact = result.stats.exact_action_evaluations / max(1, mdp.n_cells * H * result.stats.iterations)
        pruned = result.stats.pruned_action_evaluations / max(1, total)
        rows.append([label, min(31, mdp.n_cells), f"{avg_exact:.2f}", f"{pruned:.3f}", f"{elapsed:.3f}"])
    print_table(["mode", "max_actions", "avg_exact_actions", "pruned_frac", "time_s"], rows)


def experiment_5(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nExperiment 5. 2D demonstration")
    print("[stage] 15x15: training truncated PI + decomposition for figures")
    small_mdp = make_mdp(15, 15)
    small_policy = run_smdp_policy_iteration(
        small_mdp,
        discount=GAMMA,
        action_radius=5,
        use_decomposition=True,
        evaluation_sweeps=25,
        max_iterations=25,
        tolerance=1e-7,
        progress_callback=learning_progress_callback(),
    ).policy
    plot_2d_demo(small_mdp, small_policy, output_dir)

    mdp = make_mdp(50, 50, destination_radius=6)
    print("[stage] 50x50: building non-learning baseline policies")
    rows = []
    policies = [
        ("greedy", None, GreedyActionPolicy(mdp)),
        ("cost-aware", None, CostAwareActionPolicy(mdp)),
        ("smart", None, SmartActionPolicy(mdp)),
    ]
    for label, radius, policy in policies:
        print(f"[stage] 50x50: evaluating {label}")
        reward, empty = evaluate_policy(mdp, policy, trips=600)
        rows.append([label, "-", "-", "0.000", f"{reward:.3f}", f"{empty:.3f}"])

    for label, radius, decomp, bnb in [
        ("truncated PI", 15, False, False),
        ("truncated PI + decomposition", 15, True, False),
        ("truncated PI + branch-and-bound", 15, True, True),
    ]:
        print(f"[stage] 50x50: training {label} (R={radius})")
        start = time.perf_counter()
        result = run_smdp_policy_iteration(
            mdp,
            discount=GAMMA,
            action_radius=radius,
            use_decomposition=decomp,
            use_branch_and_bound=bnb,
            evaluation_sweeps=4,
            max_iterations=4,
            tolerance=1e-6,
            progress_callback=learning_progress_callback(),
        )
        elapsed = time.perf_counter() - start
        print(f"[stage] 50x50: evaluating {label}")
        reward, empty = evaluate_policy(mdp, result.policy, trips=600)
        action_fraction = average_radius_action_count(mdp, radius) / mdp.n_cells
        rows.append([label, radius, f"{action_fraction:.4f}", f"{elapsed:.3f}", f"{reward:.3f}", f"{empty:.3f}"])
    print_table(["method", "R", "action_frac", "time_s", "avg_reward", "avg_empty"], rows)
    print(f"Saved 2D figures to {output_dir}")


def evaluate_policy(mdp: AnalyticTaxiMDP, policy, trips: int = 1000, seed: int = 123) -> tuple[float, float]:
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
        total_reward += mdp.surge(action, pickup_phase) * mdp.tariff_per_cell * trip_distance - mdp.move_cost_per_cell * duration
        total_empty += empty_distance
        position = destination
        time_step = (time_step + duration) % mdp.horizon_steps
    return total_reward / trips, total_empty / trips


def greedy_policy(mdp: AnalyticTaxiMDP) -> LearnedPolicy:
    actions = {}
    values = {}
    for position, time_step in mdp.iter_states():
        action = max(mdp.iter_actions(), key=lambda a: (mdp.surge(a, time_step), -mdp.grid_distance(position, a)))
        actions[(position, time_step)] = action
        values[(position, time_step)] = 0.0
    return LearnedPolicy(mdp.horizon_steps, actions, values, "greedy")


class GreedyActionPolicy:
    def __init__(self, mdp: AnalyticTaxiMDP) -> None:
        self.horizon_steps = mdp.horizon_steps
        self.actions_by_phase = [
            max(mdp.iter_actions(), key=lambda action: (mdp.surge(action, phase), -action))
            for phase in range(mdp.horizon_steps)
        ]

    def action_for(self, position: int, time_step: int) -> int:
        return self.actions_by_phase[time_step % self.horizon_steps]


class CostAwareActionPolicy:
    def __init__(self, mdp: AnalyticTaxiMDP) -> None:
        self.mdp = mdp

    def action_for(self, position: int, time_step: int) -> int:
        phase = time_step % self.mdp.horizon_steps
        return max(
            self.mdp.iter_actions(),
            key=lambda action: (
                self.mdp.surge(action, phase) * self.mdp.tariff_per_cell
                - self.mdp.move_cost_per_cell * self.mdp.grid_distance(position, action),
                -action,
            ),
        )


class SmartActionPolicy:
    def __init__(self, mdp: AnalyticTaxiMDP) -> None:
        self.mdp = mdp
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
                    self.mdp.surge(action, phase) * self.mdp.tariff_per_cell
                    - self.mdp.move_cost_per_cell
                ) * self.expected_distances[action]
                - self.mdp.move_cost_per_cell * self.mdp.grid_distance(position, action),
                -action,
            ),
        )


def cost_aware_policy(mdp: AnalyticTaxiMDP) -> LearnedPolicy:
    actions = {}
    values = {}
    for position, time_step in mdp.iter_states():
        action = max(
            mdp.iter_actions(),
            key=lambda a: (mdp.surge(a, time_step) * mdp.tariff_per_cell - mdp.move_cost_per_cell * mdp.grid_distance(position, a), -a),
        )
        actions[(position, time_step)] = action
        values[(position, time_step)] = 0.0
    return LearnedPolicy(mdp.horizon_steps, actions, values, "cost_aware")


def smart_policy(mdp: AnalyticTaxiMDP) -> LearnedPolicy:
    expected_distances = {
        action: sum(mdp.grid_distance(action, z) * p for z, p in mdp.destination_distribution(action))
        for action in mdp.iter_actions()
    }
    actions = {}
    values = {}
    for position, time_step in mdp.iter_states():
        action = max(
            mdp.iter_actions(),
            key=lambda a: (
                (mdp.surge(a, time_step) * mdp.tariff_per_cell - mdp.move_cost_per_cell) * expected_distances[a]
                - mdp.move_cost_per_cell * mdp.grid_distance(position, a),
                -a,
            ),
        )
        actions[(position, time_step)] = action
        values[(position, time_step)] = 0.0
    return LearnedPolicy(mdp.horizon_steps, actions, values, "smart")


def average_radius_action_count(mdp: AnalyticTaxiMDP, radius: int) -> float:
    return sum(len(mdp.action_candidates(position, radius)) for position in range(mdp.n_cells)) / mdp.n_cells


def plot_2d_demo(mdp: AnalyticTaxiMDP, policy: LearnedPolicy, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = 0
    demand = [[mdp.surge(y * mdp.grid_width + x, phase) for x in range(mdp.grid_width)] for y in range(mdp.grid_height)]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(demand, origin="lower", cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Demand / surge heatmap, phase 0")
    fig.tight_layout()
    fig.savefig(output_dir / "smdp_2d_demand_heatmap.png", dpi=180)
    plt.close(fig)

    xs, ys, us, vs = [], [], [], []
    for idx in range(mdp.n_cells):
        x, y = mdp.index_to_coord(idx)
        action = policy.action_for(idx, phase)
        ax_, ay_ = mdp.index_to_coord(action)
        xs.append(x)
        ys.append(y)
        us.append(ax_ - x)
        vs.append(ay_ - y)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(demand, origin="lower", cmap="Greys", alpha=0.35)
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set_xlim(-0.5, mdp.grid_width - 0.5)
    ax.set_ylim(-0.5, mdp.grid_height - 0.5)
    ax.set_title("Truncated PI policy directions, phase 0")
    fig.tight_layout()
    fig.savefig(output_dir / "smdp_2d_policy_directions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    global VERBOSE, LOG_EVERY

    parser = argparse.ArgumentParser(description="Run numerical experiments for the periodic taxi SMDP.")
    parser.add_argument(
        "--experiment",
        choices=["all", "1", "2", "3", "4", "5"],
        default="all",
        help="Experiment to run. Experiment 5 can be slow on 50x50.",
    )
    parser.add_argument("--output-dir", default="saved_policies/smdp_experiments", help="Directory for 2D figures.")
    parser.add_argument("--verbose", action="store_true", help="Print policy-iteration evaluation/improvement progress.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N policy-evaluation sweeps in verbose mode.")
    args = parser.parse_args()
    VERBOSE = args.verbose
    LOG_EVERY = args.log_every

    selected = {"1", "2", "3", "4", "5"} if args.experiment == "all" else {args.experiment}
    if "1" in selected:
        experiment_1()
    if "2" in selected:
        experiment_2()
    if "3" in selected:
        experiment_3()
    if "4" in selected:
        experiment_4()
    if "5" in selected:
        experiment_5(Path(args.output_dir))


if __name__ == "__main__":
    main()
