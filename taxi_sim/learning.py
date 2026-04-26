from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Callable


@dataclass(frozen=True)
class LearnedPolicy:
    horizon_steps: int
    actions: dict[tuple[int, int], int]
    values: dict[tuple[int, int], float]
    method: str

    def action_for(self, position: int, time_step: int) -> int:
        return self.actions[(position, time_step % self.horizon_steps)]

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_steps": self.horizon_steps,
            "method": self.method,
            "actions": [
                {"position": position, "time_step": time_step, "action": action}
                for (position, time_step), action in sorted(self.actions.items())
            ],
            "values": [
                {"position": position, "time_step": time_step, "value": value}
                for (position, time_step), value in sorted(self.values.items())
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LearnedPolicy":
        actions = {
            (int(item["position"]), int(item["time_step"])): int(item["action"])
            for item in payload["actions"]  # type: ignore[index]
        }
        values = {
            (int(item["position"]), int(item["time_step"])): float(item["value"])
            for item in payload["values"]  # type: ignore[index]
        }
        return cls(
            horizon_steps=int(payload["horizon_steps"]),
            actions=actions,
            values=values,
            method=str(payload["method"]),
        )


def save_learned_policy(policy: LearnedPolicy, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy.to_dict(), f, indent=2, sort_keys=True)


def load_learned_policy(path: str) -> LearnedPolicy:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return LearnedPolicy.from_dict(payload)


@dataclass(frozen=True)
class LearningProgress:
    method: str
    phase: str
    iteration: int
    loss: float
    policy_changes: int | None = None
    epsilon: float | None = None
    sweeps_used: int | None = None


@dataclass(frozen=True)
class PolicyIterationStats:
    iterations: int
    evaluation_sweeps: int
    possible_action_evaluations: int
    exact_action_evaluations: int
    pruned_action_evaluations: int

    @property
    def average_exact_actions_per_state(self) -> float:
        total = self.exact_action_evaluations + self.pruned_action_evaluations
        if total == 0:
            return 0.0
        # This property is filled against improvement candidates, so the caller
        # should prefer exact_action_evaluations / state_improvement_count when
        # that denominator is available. It remains useful as a coarse ratio.
        return self.exact_action_evaluations / max(1, self.iterations)

    @property
    def pruned_fraction(self) -> float:
        total = self.exact_action_evaluations + self.pruned_action_evaluations
        if total == 0:
            return 0.0
        return self.pruned_action_evaluations / total


@dataclass(frozen=True)
class PolicyIterationResult:
    policy: LearnedPolicy
    stats: PolicyIterationStats


class AnalyticTaxiMDP:
    def __init__(
        self,
        *,
        n_cells: int,
        grid_width: int,
        grid_height: int,
        horizon_steps: int,
        analytic_dt: float,
        destination_sigma: float,
        move_cost_per_cell: float,
        tariff_per_cell: float,
        destination_radius: int | None = None,
    ) -> None:
        if n_cells < 2:
            raise ValueError("n_cells must be >= 2")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")

        self.n_cells = n_cells
        self.grid_width = max(1, grid_width)
        self.grid_height = max(1, grid_height)
        self.horizon_steps = horizon_steps
        self.analytic_dt = max(1e-6, analytic_dt)
        self.destination_sigma = max(1e-6, destination_sigma)
        self.move_cost_per_cell = move_cost_per_cell
        self.tariff_per_cell = tariff_per_cell
        self.destination_radius = destination_radius
        self._destination_cache: dict[int, list[tuple[int, float]]] = {}

    def iter_states(self) -> list[tuple[int, int]]:
        return [(position, time_step) for position in range(self.n_cells) for time_step in range(self.horizon_steps)]

    def iter_actions(self) -> range:
        return range(self.n_cells)

    def surge(self, cell_idx: int, time_step: int) -> float:
        x_idx, y_idx = self.index_to_coord(cell_idx)
        x_phase = 0.0 if self.grid_width == 1 else x_idx / (self.grid_width - 1)
        y_phase = 1.0 if self.grid_height == 1 else y_idx / (self.grid_height - 1)
        time_value = time_step * self.analytic_dt
        return math.cos(math.pi * x_phase * y_phase + math.pi * time_value) + 2.0

    def index_to_coord(self, idx: int) -> tuple[int, int]:
        return idx % self.grid_width, idx // self.grid_width

    def grid_distance(self, a: int, b: int) -> int:
        ax, ay = self.index_to_coord(a)
        bx, by = self.index_to_coord(b)
        return abs(ax - bx) + abs(ay - by)

    def destination_distribution(self, origin: int) -> list[tuple[int, float]]:
        if origin in self._destination_cache:
            return self._destination_cache[origin]

        candidates = [
            idx
            for idx in range(self.n_cells)
            if idx != origin and (
                self.destination_radius is None
                or self.grid_distance(idx, origin) <= self.destination_radius
            )
        ]
        if not candidates:
            candidates = [idx for idx in range(self.n_cells) if idx != origin]
        weights = [
            math.exp(-(self._squared_distance(idx, origin)) / (2.0 * self.destination_sigma * self.destination_sigma))
            for idx in candidates
        ]
        total = sum(weights)
        distribution = [(idx, weight / total) for idx, weight in zip(candidates, weights)]
        self._destination_cache[origin] = distribution
        return distribution

    def transitions(self, position: int, time_step: int, action: int) -> list[tuple[tuple[int, int], float, float]]:
        empty_distance = self.grid_distance(action, position)
        pickup_surge = self.surge(action, time_step)
        outcomes: list[tuple[tuple[int, int], float, float]] = []
        for destination, probability in self.destination_distribution(action):
            trip_distance = self.grid_distance(destination, action)
            duration = empty_distance + trip_distance
            reward = (
                pickup_surge * self.tariff_per_cell * trip_distance
                - self.move_cost_per_cell * duration
            )
            next_state = (destination, (time_step + duration) % self.horizon_steps)
            outcomes.append((next_state, probability, reward))
        return outcomes

    def expected_action_value(
        self,
        values: dict[tuple[int, int], float],
        state: tuple[int, int],
        action: int,
        discount: float,
    ) -> float:
        total = 0.0
        for next_state, probability, reward in self.transitions(state[0], state[1], action):
            total += probability * (reward + discount * values[next_state])
        return total

    def action_candidates(self, position: int, radius: int | None = None) -> list[int]:
        if radius is None:
            return list(self.iter_actions())
        return [action for action in self.iter_actions() if self.grid_distance(position, action) <= radius]

    def smdp_expected_action_value(
        self,
        values: dict[tuple[int, int], float],
        state: tuple[int, int],
        action: int,
        discount: float,
        reward_timing: str = "discounted_rewards",
    ) -> float:
        x, k = state
        empty_distance = self.grid_distance(x, action)
        pickup_phase = (k + empty_distance) % self.horizon_steps
        total = 0.0
        for destination, probability in self.destination_distribution(action):
            trip_distance = self.grid_distance(action, destination)
            duration = empty_distance + trip_distance
            reward_after_pickup = (
                self.surge(action, pickup_phase) * self.tariff_per_cell * trip_distance
                - self.move_cost_per_cell * trip_distance
            )
            next_state = (destination, (pickup_phase + trip_distance) % self.horizon_steps)
            if reward_timing == "transition":
                total += probability * (
                    reward_after_pickup
                    - self.move_cost_per_cell * empty_distance
                    + (discount ** duration) * values[next_state]
                )
            else:
                total += probability * (reward_after_pickup + (discount ** trip_distance) * values[next_state])
        if reward_timing == "transition":
            return total
        return -self.move_cost_per_cell * empty_distance + (discount ** empty_distance) * total

    def decomposed_g_values(
        self,
        values: dict[tuple[int, int], float],
        discount: float,
        reward_timing: str = "discounted_rewards",
    ) -> dict[tuple[int, int], float | tuple[float, float]]:
        g_values: dict[tuple[int, int], float | tuple[float, float]] = {}
        for action in self.iter_actions():
            distribution = self.destination_distribution(action)
            for pickup_phase in range(self.horizon_steps):
                surge = self.surge(action, pickup_phase)
                total = 0.0
                reward_total = 0.0
                discounted_value_total = 0.0
                for destination, probability in distribution:
                    trip_distance = self.grid_distance(action, destination)
                    next_phase = (pickup_phase + trip_distance) % self.horizon_steps
                    reward_after_pickup = (
                        surge * self.tariff_per_cell * trip_distance
                        - self.move_cost_per_cell * trip_distance
                    )
                    if reward_timing == "transition":
                        reward_total += probability * reward_after_pickup
                        discounted_value_total += probability * (
                            (discount ** trip_distance) * values[(destination, next_phase)]
                        )
                    else:
                        total += probability * (
                            reward_after_pickup
                            + (discount ** trip_distance) * values[(destination, next_phase)]
                        )
                if reward_timing == "transition":
                    g_values[(action, pickup_phase)] = (reward_total, discounted_value_total)
                else:
                    g_values[(action, pickup_phase)] = total
        return g_values

    def decomposed_action_value(
        self,
        g_values: dict[tuple[int, int], float | tuple[float, float]],
        state: tuple[int, int],
        action: int,
        discount: float,
        reward_timing: str = "discounted_rewards",
    ) -> float:
        x, k = state
        empty_distance = self.grid_distance(x, action)
        pickup_phase = (k + empty_distance) % self.horizon_steps
        g_value = g_values[(action, pickup_phase)]
        if reward_timing == "transition":
            reward_after_pickup, discounted_value_after_pickup = g_value  # type: ignore[misc]
            return (
                -self.move_cost_per_cell * empty_distance
                + reward_after_pickup
                + (discount ** empty_distance) * discounted_value_after_pickup
            )
        return (
            -self.move_cost_per_cell * empty_distance
            + (discount ** empty_distance) * float(g_value)
        )

    def _squared_distance(self, a: int, b: int) -> float:
        ax, ay = self.index_to_coord(a)
        bx, by = self.index_to_coord(b)
        dx = ax - bx
        dy = ay - by
        return float(dx * dx + dy * dy)


def run_policy_iteration(
    mdp: AnalyticTaxiMDP,
    *,
    discount: float,
    evaluation_sweeps: int = 80,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
    progress_callback: Callable[[LearningProgress], None] | None = None,
) -> LearnedPolicy:
    states = mdp.iter_states()
    policy = {(position, time_step): 0 for position, time_step in states}
    values = {(position, time_step): 0.0 for position, time_step in states}

    for iteration_idx in range(max_iterations):
        values, max_delta, _sweeps_used = _evaluate_policy_by_bellman_iteration(
            mdp=mdp,
            policy=policy,
            values=values,
            discount=discount,
            max_sweeps=evaluation_sweeps,
            tolerance=tolerance,
            progress_callback=progress_callback,
            outer_iteration_index=iteration_idx,
        )

        stable = True
        policy_changes = 0
        for state in states:
            best_action = policy[state]
            best_value = mdp.expected_action_value(values, state, best_action, discount)
            for action in mdp.iter_actions():
                candidate = mdp.expected_action_value(values, state, action, discount)
                if candidate > best_value + 1e-12:
                    best_value = candidate
                    best_action = action
            if best_action != policy[state]:
                policy[state] = best_action
                stable = False
                policy_changes += 1
        if progress_callback is not None:
            progress_callback(
                LearningProgress(
                    method="policy_iteration",
                    phase="improvement",
                    iteration=iteration_idx + 1,
                    loss=max_delta,
                    policy_changes=policy_changes,
                    sweeps_used=_sweeps_used,
                )
            )
        if stable:
            break

    return LearnedPolicy(horizon_steps=mdp.horizon_steps, actions=policy, values=values, method="policy_iteration")


def run_smdp_policy_iteration(
    mdp: AnalyticTaxiMDP,
    *,
    discount: float,
    action_radius: int | None = None,
    use_decomposition: bool = False,
    use_branch_and_bound: bool = False,
    reward_timing: str = "discounted_rewards",
    evaluation_sweeps: int = 80,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
    progress_callback: Callable[[LearningProgress], None] | None = None,
) -> PolicyIterationResult:
    states = mdp.iter_states()
    policy = {
        (position, time_step): mdp.action_candidates(position, action_radius)[0]
        for position, time_step in states
    }
    values = {(position, time_step): 0.0 for position, time_step in states}
    total_sweeps = 0
    possible_evaluations = 0
    exact_evaluations = 0
    pruned_evaluations = 0

    for iteration_idx in range(max_iterations):
        values, max_delta, sweeps_used = _evaluate_smdp_policy_by_bellman_iteration(
            mdp=mdp,
            policy=policy,
            values=values,
            discount=discount,
            use_decomposition=use_decomposition,
            reward_timing=reward_timing,
            max_sweeps=evaluation_sweeps,
            tolerance=tolerance,
            progress_callback=progress_callback,
            outer_iteration_index=iteration_idx,
        )
        total_sweeps += sweeps_used

        stable = True
        policy_changes = 0
        g_values = mdp.decomposed_g_values(values, discount, reward_timing=reward_timing) if use_decomposition else None
        upper_value = _smdp_upper_value_bound(mdp, discount)
        max_trip_gain = _smdp_max_trip_gain(mdp)

        for state in states:
            candidates = mdp.action_candidates(state[0], action_radius)
            possible_evaluations += len(candidates)
            candidates.sort(key=lambda action: (mdp.grid_distance(state[0], action), action))
            best_action = policy[state]
            best_value = float("-inf")
            for action in candidates:
                empty_distance = mdp.grid_distance(state[0], action)
                if use_branch_and_bound:
                    ub = mdp.move_cost_per_cell * (
                        max_trip_gain - empty_distance + (discount ** (empty_distance + 1)) * upper_value
                    )
                    if ub < best_value:
                        pruned_evaluations += 1
                        continue
                candidate = (
                    mdp.decomposed_action_value(g_values, state, action, discount, reward_timing=reward_timing)
                    if g_values is not None
                    else mdp.smdp_expected_action_value(values, state, action, discount, reward_timing=reward_timing)
                )
                exact_evaluations += 1
                if candidate > best_value + 1e-12:
                    best_value = candidate
                    best_action = action
            if best_action != policy[state]:
                policy[state] = best_action
                stable = False
                policy_changes += 1
        if progress_callback is not None:
            progress_callback(
                LearningProgress(
                    method="smdp_policy_iteration",
                    phase="improvement",
                    iteration=iteration_idx + 1,
                    loss=max_delta,
                    policy_changes=policy_changes,
                    sweeps_used=sweeps_used,
                )
            )
        if stable:
            break

    method = "smdp_policy_iteration"
    if action_radius is not None:
        method += f"_R{action_radius}"
    if use_decomposition:
        method += "_decomposed"
    if use_branch_and_bound:
        method += "_bnb"

    iterations = iteration_idx + 1 if "iteration_idx" in locals() else 0
    return PolicyIterationResult(
        policy=LearnedPolicy(horizon_steps=mdp.horizon_steps, actions=policy, values=values, method=method),
        stats=PolicyIterationStats(
            iterations=iterations,
            evaluation_sweeps=total_sweeps,
            possible_action_evaluations=possible_evaluations,
            exact_action_evaluations=exact_evaluations,
            pruned_action_evaluations=pruned_evaluations,
        ),
    )


def naive_bellman_update_smdp(
    mdp: AnalyticTaxiMDP,
    values: dict[tuple[int, int], float],
    *,
    discount: float,
    action_radius: int | None = None,
    reward_timing: str = "discounted_rewards",
) -> dict[tuple[int, int], float]:
    next_values: dict[tuple[int, int], float] = {}
    for state in mdp.iter_states():
        next_values[state] = max(
            mdp.smdp_expected_action_value(values, state, action, discount)
            if reward_timing == "discounted_rewards"
            else mdp.smdp_expected_action_value(values, state, action, discount, reward_timing=reward_timing)
            for action in mdp.action_candidates(state[0], action_radius)
        )
    return next_values


def decomposed_bellman_update_smdp(
    mdp: AnalyticTaxiMDP,
    values: dict[tuple[int, int], float],
    *,
    discount: float,
    action_radius: int | None = None,
    reward_timing: str = "discounted_rewards",
) -> dict[tuple[int, int], float]:
    g_values = mdp.decomposed_g_values(values, discount, reward_timing=reward_timing)
    next_values: dict[tuple[int, int], float] = {}
    for state in mdp.iter_states():
        next_values[state] = max(
            mdp.decomposed_action_value(g_values, state, action, discount, reward_timing=reward_timing)
            for action in mdp.action_candidates(state[0], action_radius)
        )
    return next_values


def radius_bound_parameters_1d(
    *,
    n_cells: int,
    discount: float,
    surge_max: float = 3.0,
    q: float = 1.5,
    destination_sigma: float = 1.5,
    move_cost_per_cell: float = 1.0,
) -> dict[str, float]:
    max_expected_trip_distance = 0.0
    sigma = max(1e-6, destination_sigma)
    for origin in range(n_cells):
        weights = []
        distances = []
        for destination in range(n_cells):
            if destination == origin:
                continue
            distance = abs(destination - origin)
            distances.append(float(distance))
            weights.append(math.exp(-(distance * distance) / (2.0 * sigma * sigma)))
        total = sum(weights)
        if total > 0.0:
            max_expected_trip_distance = max(
                max_expected_trip_distance,
                sum(distance * weight for distance, weight in zip(distances, weights)) / total,
            )
    c_value = max(0.0, (surge_max * q - 1.0) * max_expected_trip_distance)
    upper_value = c_value / max(1e-12, 1.0 - discount)
    lipschitz = move_cost_per_cell * (1.0 + upper_value * (1.0 - discount))
    r_cut = 0
    while r_cut < n_cells and move_cost_per_cell * (
        c_value - r_cut + (discount ** (r_cut + 1)) * upper_value
    ) >= 0.0:
        r_cut += 1
    r_star = min(n_cells - 1, r_cut)
    return {
        "C": c_value,
        "L": lipschitz,
        "U": upper_value,
        "r_cut": float(r_cut),
        "R_star": float(r_star),
        "action_fraction": min(2 * r_star + 1, n_cells) / n_cells,
    }


def _evaluate_policy_by_bellman_iteration(
    *,
    mdp: AnalyticTaxiMDP,
    policy: dict[tuple[int, int], int],
    values: dict[tuple[int, int], float],
    discount: float,
    max_sweeps: int,
    tolerance: float,
    progress_callback: Callable[[LearningProgress], None] | None,
    outer_iteration_index: int,
) -> tuple[dict[tuple[int, int], float], float, int]:
    states = mdp.iter_states()
    current_values = values
    last_delta = 0.0

    for sweep_idx in range(max_sweeps):
        max_delta = 0.0
        next_values: dict[tuple[int, int], float] = {}
        for state in states:
            action = policy[state]
            value = mdp.expected_action_value(current_values, state, action, discount)
            next_values[state] = value
            max_delta = max(max_delta, abs(value - current_values[state]))
        current_values = next_values
        last_delta = max_delta

        if progress_callback is not None:
            progress_callback(
                LearningProgress(
                    method="policy_iteration",
                    phase="evaluation",
                    iteration=outer_iteration_index * max_sweeps + sweep_idx + 1,
                    loss=max_delta,
                )
            )
        if max_delta < tolerance:
            return current_values, max_delta, sweep_idx + 1

    return current_values, last_delta, max_sweeps


def _evaluate_smdp_policy_by_bellman_iteration(
    *,
    mdp: AnalyticTaxiMDP,
    policy: dict[tuple[int, int], int],
    values: dict[tuple[int, int], float],
    discount: float,
    use_decomposition: bool,
    reward_timing: str,
    max_sweeps: int,
    tolerance: float,
    progress_callback: Callable[[LearningProgress], None] | None,
    outer_iteration_index: int,
) -> tuple[dict[tuple[int, int], float], float, int]:
    states = mdp.iter_states()
    current_values = values
    last_delta = 0.0

    for sweep_idx in range(max_sweeps):
        max_delta = 0.0
        next_values: dict[tuple[int, int], float] = {}
        g_values = mdp.decomposed_g_values(current_values, discount, reward_timing=reward_timing) if use_decomposition else None
        for state in states:
            action = policy[state]
            value = (
                mdp.decomposed_action_value(g_values, state, action, discount, reward_timing=reward_timing)
                if g_values is not None
                else mdp.smdp_expected_action_value(current_values, state, action, discount, reward_timing=reward_timing)
            )
            next_values[state] = value
            max_delta = max(max_delta, abs(value - current_values[state]))
        current_values = next_values
        last_delta = max_delta

        if progress_callback is not None:
            progress_callback(
                LearningProgress(
                    method="smdp_policy_iteration",
                    phase="evaluation",
                    iteration=outer_iteration_index * max_sweeps + sweep_idx + 1,
                    loss=max_delta,
                )
            )
        if max_delta < tolerance:
            return current_values, max_delta, sweep_idx + 1

    return current_values, last_delta, max_sweeps


def _smdp_max_trip_gain(mdp: AnalyticTaxiMDP) -> float:
    max_expected_distance = 0.0
    for action in mdp.iter_actions():
        expected_distance = 0.0
        for destination, _probability in mdp.destination_distribution(action):
            expected_distance += _probability * mdp.grid_distance(action, destination)
        max_expected_distance = max(max_expected_distance, expected_distance)
    max_surge = 3.0
    return max(0.0, (max_surge * mdp.tariff_per_cell / mdp.move_cost_per_cell - 1.0) * max_expected_distance)


def _smdp_upper_value_bound(mdp: AnalyticTaxiMDP, discount: float) -> float:
    return mdp.move_cost_per_cell * _smdp_max_trip_gain(mdp) / max(1e-12, 1.0 - discount)


def run_q_learning(
    mdp: AnalyticTaxiMDP,
    *,
    discount: float,
    episodes: int = 4000,
    alpha: float = 0.15,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.02,
    episode_decisions: int | None = None,
    seed: int | None = None,
    progress_callback: Callable[[LearningProgress], None] | None = None,
    log_every: int = 100,
) -> LearnedPolicy:
    rng = random.Random(seed)
    states = mdp.iter_states()
    actions = list(mdp.iter_actions())
    q_values = {(state[0], state[1], action): 0.0 for state in states for action in actions}
    decisions_per_episode = episode_decisions or mdp.horizon_steps
    current_epsilon = epsilon

    running_abs_td = 0.0
    running_updates = 0
    for episode_idx in range(episodes):
        state = rng.choice(states)
        for _ in range(decisions_per_episode):
            if rng.random() < current_epsilon:
                action = rng.choice(actions)
            else:
                action = _best_action_from_q(q_values, state, actions)

            next_states, weights, rewards = _sampleable_outcomes(mdp.transitions(state[0], state[1], action))
            choice_idx = rng.choices(range(len(next_states)), weights=weights, k=1)[0]
            next_state = next_states[choice_idx]
            reward = rewards[choice_idx]

            best_next = max(q_values[(next_state[0], next_state[1], next_action)] for next_action in actions)
            key = (state[0], state[1], action)
            td_error = reward + discount * best_next - q_values[key]
            q_values[key] += alpha * td_error
            running_abs_td += abs(td_error)
            running_updates += 1
            state = next_state

        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        if progress_callback is not None and ((episode_idx + 1) % max(1, log_every) == 0 or episode_idx + 1 == episodes):
            avg_loss = 0.0 if running_updates == 0 else running_abs_td / running_updates
            progress_callback(
                LearningProgress(
                    method="q_learning",
                    phase="training",
                    iteration=episode_idx + 1,
                    loss=avg_loss,
                    epsilon=current_epsilon,
                )
            )
            running_abs_td = 0.0
            running_updates = 0

    policy_actions: dict[tuple[int, int], int] = {}
    policy_values: dict[tuple[int, int], float] = {}
    for state in states:
        best_action = _best_action_from_q(q_values, state, actions)
        policy_actions[state] = best_action
        policy_values[state] = q_values[(state[0], state[1], best_action)]

    return LearnedPolicy(horizon_steps=mdp.horizon_steps, actions=policy_actions, values=policy_values, method="q_learning")


def _best_action_from_q(
    q_values: dict[tuple[int, int, int], float],
    state: tuple[int, int],
    actions: list[int],
) -> int:
    best_action = actions[0]
    best_value = q_values[(state[0], state[1], best_action)]
    for action in actions[1:]:
        candidate = q_values[(state[0], state[1], action)]
        if candidate > best_value:
            best_value = candidate
            best_action = action
    return best_action


def _sampleable_outcomes(
    outcomes: list[tuple[tuple[int, int], float, float]]
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    next_states = [state for state, _, _ in outcomes]
    weights = [probability for _, probability, _ in outcomes]
    rewards = [reward for _, _, reward in outcomes]
    return next_states, weights, rewards
