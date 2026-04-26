from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Protocol

from .learning import (
    AnalyticTaxiMDP,
    LearnedPolicy,
    load_learned_policy,
    run_policy_iteration,
    run_q_learning,
)
from .models import (
    AnalyticWaveModel,
    CellState,
    CommuteCycleModel,
    HotspotShiftModel,
    RandomWalkModel,
    RandomizationModel,
    UniformResetModel,
)


_UNSET = object()
_LEARNED_POLICY_CACHE: dict[tuple[object, ...], LearnedPolicy] = {}


class Strategy(Protocol):
    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        ...

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        ...


class GreedyStrategy:
    """Wait near the highest-surge cell and accept offers close to it."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        best_idx = 0
        best_surge = sim.cells[0].surge
        for idx, cell in enumerate(sim.cells[1:], start=1):
            if cell.surge > best_surge:
                best_surge = cell.surge
                best_idx = idx
        return best_idx

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_wait_cell(sim)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        return sim.grid_distance(offer.pickup_cell, target_cell) <= sim.config.offer_acceptance_radius


class RandomStrategy:
    """Move randomly and accept random nearby offers."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        return sim.rng.randrange(sim.config.n_cells)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_wait_cell(sim)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        return sim.rng.random() < 0.65


class NearestMaxSurgeStrategy:
    """Pick the closest cell among cells with maximum surge."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        surges = sim.get_surges()
        max_surge = max(surges)
        candidates = [idx for idx, surge in enumerate(surges) if abs(surge - max_surge) < 1e-9]
        return min(candidates, key=lambda idx: (sim.grid_distance(sim.driver_position, idx), idx))

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_wait_cell(sim)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        return sim.grid_distance(offer.pickup_cell, target_cell) <= max(1, sim.config.offer_acceptance_radius)


class CostAwareStrategy:
    """
    Pick a cell by balancing surge and travel distance:
    score = surge - penalty * distance_to_cell
    """

    def __init__(self, distance_penalty: float = 0.25) -> None:
        self.distance_penalty = max(0.0, distance_penalty)

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        best_idx = 0
        best_score = float("-inf")
        for idx, cell in enumerate(sim.cells):
            distance = sim.grid_distance(sim.driver_position, idx)
            score = cell.surge - self.distance_penalty * distance
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_wait_cell(sim)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        offered_score = sim.cells[offer.pickup_cell].surge - self.distance_penalty * sim.grid_distance(sim.driver_position, offer.pickup_cell)
        target_score = sim.cells[target_cell].surge - self.distance_penalty * sim.grid_distance(sim.driver_position, target_cell)
        return offered_score >= target_score - sim.config.offer_score_slack


class EpsilonGreedyStrategy:
    """Greedy most of the time, random exploration with probability epsilon."""

    def __init__(self, epsilon: float = 0.15) -> None:
        self.epsilon = min(max(epsilon, 0.0), 1.0)
        self._greedy = GreedyStrategy()
        self._random = RandomStrategy()

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        if sim.rng.random() < self.epsilon:
            return self._random.choose_wait_cell(sim)
        return self._greedy.choose_wait_cell(sim)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_wait_cell(sim)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        if sim.rng.random() < self.epsilon:
            return self._random.should_accept_offer(sim, offer, target_cell)
        return self._greedy.should_accept_offer(sim, offer, target_cell)


class SmartStrategy:
    """
    Analytic-only strategy.

    For each candidate cell x, compute:
    expected_trip_value(x) - cost_to_reach(x)
    and choose the maximum.
    """

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_pickup_cell(sim)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        sim.ensure_smart_strategy_supported()
        best_idx = 0
        best_score = float("-inf")
        for idx in range(sim.config.n_cells):
            score = sim.expected_trip_value_from_cell(idx) - sim.config.move_cost_per_cell * sim.grid_distance(sim.driver_position, idx)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        sim.ensure_smart_strategy_supported()
        offered_score = (
            sim.expected_trip_value_from_cell(offer.pickup_cell)
            - sim.config.move_cost_per_cell * sim.grid_distance(sim.driver_position, offer.pickup_cell)
        )
        target_score = (
            sim.expected_trip_value_from_cell(target_cell)
            - sim.config.move_cost_per_cell * sim.grid_distance(sim.driver_position, target_cell)
        )
        return offered_score >= target_score - sim.config.offer_score_slack


class PolicyIterationStrategy:
    """Use policy iteration on the analytic control MDP."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_pickup_cell(sim)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        policy = sim.get_learned_policy("policy_iteration")
        return policy.action_for(sim.driver_position, sim.stats.time_steps)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        policy = sim.get_learned_policy("policy_iteration")
        target_action = policy.action_for(sim.driver_position, sim.stats.time_steps)
        return offer.pickup_cell == target_action


class QLearningStrategy:
    """Use a Q-learning policy trained on the analytic control MDP."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_pickup_cell(sim)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        policy = sim.get_learned_policy("q_learning")
        return policy.action_for(sim.driver_position, sim.stats.time_steps)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        policy = sim.get_learned_policy("q_learning")
        target_action = policy.action_for(sim.driver_position, sim.stats.time_steps)
        return offer.pickup_cell == target_action


class SavedPolicyStrategy:
    """Replay a previously saved learned policy."""

    def choose_wait_cell(self, sim: "TaxiSimulation") -> int:
        return self.choose_pickup_cell(sim)

    def choose_pickup_cell(self, sim: "TaxiSimulation") -> int:
        policy = sim.get_learned_policy("saved_policy")
        return policy.action_for(sim.driver_position, sim.stats.time_steps)

    def should_accept_offer(self, sim: "TaxiSimulation", offer: "RideOffer", target_cell: int) -> bool:
        policy = sim.get_learned_policy("saved_policy")
        target_action = policy.action_for(sim.driver_position, sim.stats.time_steps)
        return offer.pickup_cell == target_action


@dataclass
class RideOffer:
    pickup_cell: int
    destination: int
    distance_to_pickup: int
    surge: float


@dataclass
class SimulationConfig:
    n_cells: int = 10
    field_dimension: int = 1
    grid_width: int = 10
    grid_height: int = 1
    surge_update_every_k_steps: int = 3
    move_cost_per_cell: float = 1.0
    tariff_per_cell: float = 4.0
    initial_balance: float = 0.0
    randomization_model: str = "uniform_reset"
    dispatch_mode: str = "nearby_offer"
    strategy_name: str = "greedy"
    strategy_distance_penalty: float = 0.25
    strategy_epsilon: float = 0.15
    offer_radius: int = 2
    offer_acceptance_radius: int = 1
    offer_score_slack: float = 0.15
    offer_probability_scale: float = 0.45
    commute_mode: str = "cycle"
    commute_period_ticks: int = 60
    analytic_dt: float = 0.05
    analytic_destination_sigma: float = 1.5
    rl_horizon_steps: int = 40
    rl_discount: float = 0.97
    policy_iteration_evaluation_sweeps: int = 80
    policy_iteration_max_iterations: int = 80
    policy_iteration_tolerance: float = 1e-9
    q_learning_episodes: int = 4000
    q_learning_alpha: float = 0.15
    q_learning_epsilon: float = 0.25
    q_learning_epsilon_decay: float = 0.995
    q_learning_min_epsilon: float = 0.02
    q_learning_episode_decisions: int = 40
    learning_verbose: bool = False
    learning_log_every: int = 50
    load_policy_path: str | None = None
    seed: int | None = None


@dataclass
class SimulationStats:
    balance: float = 0.0
    time_steps: int = 0
    trips_completed: int = 0
    empty_distance: int = 0
    paid_distance: int = 0
    total_revenue: float = 0.0
    total_move_cost: float = 0.0
    surge_updates: int = 0
    offers_seen: int = 0
    offers_accepted: int = 0
    offers_skipped: int = 0
    waiting_ticks: int = 0
    reposition_distance: int = 0


@dataclass
class Event:
    kind: str
    message: str
    data: dict = field(default_factory=dict)


class _TimeLimitReached(Exception):
    def __init__(self, event: Event) -> None:
        super().__init__(event.message)
        self.event = event


class TaxiSimulation:
    def __init__(self, config: SimulationConfig, strategy: Strategy | None = None) -> None:
        if config.grid_width <= 0 or config.grid_height <= 0:
            raise ValueError("grid_width and grid_height must be > 0")
        expected_dimension = 1 if config.grid_height == 1 else 2
        if config.field_dimension not in {1, 2}:
            raise ValueError("field_dimension must be 1 or 2")
        if config.field_dimension != expected_dimension:
            raise ValueError("field_dimension must match grid shape")
        config.n_cells = config.grid_width * config.grid_height
        if config.n_cells < 2:
            raise ValueError("n_cells must be >= 2")
        if config.surge_update_every_k_steps <= 0:
            raise ValueError("surge_update_every_k_steps must be > 0")

        self.config = config
        self.rng = random.Random(config.seed)
        self.strategy = strategy or self._build_strategy(config.strategy_name)
        self.model = self._build_model(config.randomization_model)
        self.cells: list[CellState] = self.model.initialize(self.rng, config.n_cells)

        self.driver_position = 0
        self.stats = SimulationStats(balance=config.initial_balance)
        self.last_event: Event | None = None
        self.trip_context: dict[str, object | None] = {
            "phase": "idle",
            "chosen_cell": None,
            "passenger_cell": None,
            "destination_cell": None,
            "passenger_onboard": False,
            "offered_pickup_cell": None,
        }
        self._max_time_steps: int | None = None

    def _build_model(self, model_name: str) -> RandomizationModel:
        models: dict[str, RandomizationModel] = {
            "uniform_reset": UniformResetModel(),
            "random_walk": RandomWalkModel(),
            "hotspot_shift": HotspotShiftModel(),
            "commute_cycle": CommuteCycleModel(
                mode=self.config.commute_mode,
                period_ticks=self.config.commute_period_ticks,
            ),
            "analytic_wave": AnalyticWaveModel(
                dt=self.config.analytic_dt,
                grid_width=self.config.grid_width,
                grid_height=self.config.grid_height,
            ),
        }
        if model_name not in models:
            raise ValueError(f"Unknown randomization model: {model_name}")
        return models[model_name]

    def _build_strategy(self, strategy_name: str) -> Strategy:
        strategies: dict[str, Strategy] = {
            "greedy": GreedyStrategy(),
            "random": RandomStrategy(),
            "nearest_max": NearestMaxSurgeStrategy(),
            "cost_aware": CostAwareStrategy(distance_penalty=self.config.strategy_distance_penalty),
            "epsilon_greedy": EpsilonGreedyStrategy(epsilon=self.config.strategy_epsilon),
            "smart": SmartStrategy(),
            "policy_iteration": PolicyIterationStrategy(),
            "q_learning": QLearningStrategy(),
            "saved_policy": SavedPolicyStrategy(),
        }
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        if strategy_name in {"smart", "policy_iteration", "q_learning", "saved_policy"}:
            self.ensure_smart_strategy_supported()
        return strategies[strategy_name]

    def get_surges(self) -> list[float]:
        return [cell.surge for cell in self.cells]

    def get_model_label(self) -> str:
        return self.config.randomization_model

    def get_model_cycle_label(self) -> str:
        return self.model.describe(self.stats.time_steps)

    def index_to_coord(self, idx: int) -> tuple[int, int]:
        return idx % self.config.grid_width, idx // self.config.grid_width

    def coord_to_index(self, x: int, y: int) -> int:
        return y * self.config.grid_width + x

    def format_position(self, idx: int | None) -> str:
        if idx is None:
            return "None"
        x, y = self.index_to_coord(idx)
        if self.config.field_dimension == 1:
            return str(x)
        return f"({x},{y})"

    def grid_distance(self, a: int, b: int) -> int:
        ax, ay = self.index_to_coord(a)
        bx, by = self.index_to_coord(b)
        return abs(ax - bx) + abs(ay - by)

    def ensure_smart_strategy_supported(self) -> None:
        if self.config.randomization_model != "analytic_wave" or self.config.dispatch_mode != "direct_cell":
            raise ValueError(
                "smart/policy_iteration/q_learning/saved_policy strategies are supported only with "
                "randomization_model=analytic_wave and dispatch_mode=direct_cell"
            )

    def get_learned_policy(self, method: str) -> LearnedPolicy:
        self.ensure_smart_strategy_supported()
        cache_key = (
            method,
            self.config.n_cells,
            self.config.grid_width,
            self.config.grid_height,
            self.config.analytic_dt,
            self.config.analytic_destination_sigma,
            self.config.move_cost_per_cell,
            self.config.tariff_per_cell,
            self.config.rl_horizon_steps,
            self.config.rl_discount,
            self.config.policy_iteration_evaluation_sweeps,
            self.config.policy_iteration_max_iterations,
            self.config.policy_iteration_tolerance,
            self.config.q_learning_episodes,
            self.config.q_learning_alpha,
            self.config.q_learning_epsilon,
            self.config.q_learning_epsilon_decay,
            self.config.q_learning_min_epsilon,
            self.config.q_learning_episode_decisions,
            self.config.learning_verbose,
            self.config.learning_log_every,
            self.config.load_policy_path,
            self.config.seed,
        )
        if cache_key in _LEARNED_POLICY_CACHE:
            return _LEARNED_POLICY_CACHE[cache_key]

        if method == "saved_policy":
            if not self.config.load_policy_path:
                raise ValueError("saved_policy strategy requires load_policy_path")
            policy = load_learned_policy(self.config.load_policy_path)
            _LEARNED_POLICY_CACHE[cache_key] = policy
            return policy

        mdp = AnalyticTaxiMDP(
            n_cells=self.config.n_cells,
            grid_width=self.config.grid_width,
            grid_height=self.config.grid_height,
            horizon_steps=self.config.rl_horizon_steps,
            analytic_dt=self.config.analytic_dt,
            destination_sigma=self.config.analytic_destination_sigma,
            move_cost_per_cell=self.config.move_cost_per_cell,
            tariff_per_cell=self.config.tariff_per_cell,
        )
        if method == "policy_iteration":
            policy = run_policy_iteration(
                mdp,
                discount=self.config.rl_discount,
                evaluation_sweeps=self.config.policy_iteration_evaluation_sweeps,
                max_iterations=self.config.policy_iteration_max_iterations,
                tolerance=self.config.policy_iteration_tolerance,
                progress_callback=self._learning_progress_callback if self.config.learning_verbose else None,
            )
        elif method == "q_learning":
            policy = run_q_learning(
                mdp,
                discount=self.config.rl_discount,
                episodes=self.config.q_learning_episodes,
                alpha=self.config.q_learning_alpha,
                epsilon=self.config.q_learning_epsilon,
                epsilon_decay=self.config.q_learning_epsilon_decay,
                min_epsilon=self.config.q_learning_min_epsilon,
                episode_decisions=self.config.q_learning_episode_decisions,
                seed=self.config.seed,
                progress_callback=self._learning_progress_callback if self.config.learning_verbose else None,
                log_every=self.config.learning_log_every,
            )
        else:
            raise ValueError(f"Unknown learning method: {method}")
        _LEARNED_POLICY_CACHE[cache_key] = policy
        return policy

    def _learning_progress_callback(self, progress) -> None:
        if progress.method == "policy_iteration":
            if progress.phase == "evaluation":
                if progress.iteration % max(1, self.config.learning_log_every) == 0:
                    print(
                        f"[policy evaluation] sweep={progress.iteration} "
                        f"loss={progress.loss:.8f}"
                    )
            else:
                print(
                    f"[policy improvement] iteration={progress.iteration} "
                    f"evaluation_loss={progress.loss:.8f} "
                    f"sweeps_used={progress.sweeps_used} "
                    f"policy_changes={progress.policy_changes}"
                )
        elif progress.method == "q_learning":
            print(
                f"[q_learning] episode={progress.iteration} "
                f"loss={progress.loss:.8f} epsilon={progress.epsilon:.4f}"
            )

    def expected_trip_value_from_cell(self, origin: int) -> float:
        """
        E[revenue | pickup=origin] under the analytic destination distribution.
        """
        expected_distance = self.expected_destination_distance_from_cell(origin)
        return self.cells[origin].surge * self.config.tariff_per_cell * expected_distance

    def expected_destination_distance_from_cell(self, origin: int) -> float:
        candidates = [idx for idx in range(self.config.n_cells) if idx != origin]
        if not candidates:
            return 0.0

        sigma = max(1e-6, self.config.analytic_destination_sigma)
        weights = []
        distances = []
        for idx in candidates:
            distance = self.grid_distance(idx, origin)
            distances.append(float(distance))
            ox, oy = self.index_to_coord(origin)
            dx, dy = self.index_to_coord(idx)
            sq_distance = (dx - ox) * (dx - ox) + (dy - oy) * (dy - oy)
            weights.append(math.exp(-(sq_distance) / (2.0 * sigma * sigma)))

        total_weight = sum(weights)
        if total_weight <= 0.0:
            return 0.0
        return sum(distance * weight for distance, weight in zip(distances, weights)) / total_weight

    def step(self) -> Event:
        if self.config.dispatch_mode == "direct_cell":
            return self._step_direct_dispatch()
        while True:
            preferred_cell = self.strategy.choose_wait_cell(self)
            self._set_trip_context(
                phase="waiting_offer",
                chosen_cell=preferred_cell,
                passenger_cell=None,
                destination_cell=None,
                passenger_onboard=False,
                offered_pickup_cell=None,
            )
            offer = self._generate_offer()
            if offer is not None:
                self.stats.offers_seen += 1
                self._set_trip_context(offered_pickup_cell=offer.pickup_cell)
                if self.strategy.should_accept_offer(self, offer, preferred_cell):
                    self.stats.offers_accepted += 1
                    return self._complete_offer(offer, preferred_cell)
                self.stats.offers_skipped += 1
                self._advance_toward_preferred_or_wait(preferred_cell)
                continue

            self._advance_toward_preferred_or_wait(preferred_cell)

    def run(self, trips: int | None = None, max_time_steps: int | None = None, renderer=None, render_each_tick: bool = False) -> None:
        if trips is None and max_time_steps is None:
            raise ValueError("At least one stopping condition must be provided: trips or max_time_steps")
        if trips is not None and trips < 0:
            raise ValueError("trips must be >= 0")
        if max_time_steps is not None and max_time_steps < 0:
            raise ValueError("max_time_steps must be >= 0")

        self._max_time_steps = max_time_steps
        if renderer is not None:
            renderer.render(self, Event(kind="init", message="Simulation started"))
        try:
            while not self._should_stop(trips):
                if renderer is not None and getattr(renderer, "is_closed", False):
                    break
                try:
                    if self.config.dispatch_mode == "direct_cell" and renderer is not None and render_each_tick:
                        self._run_direct_dispatch_with_tick_renders(renderer)
                    elif renderer is None or not render_each_tick:
                        event = self.step()
                        if renderer is not None:
                            renderer.render(self, event)
                    else:
                        self._run_trip_with_tick_renders(renderer)
                except _TimeLimitReached as exc:
                    self.last_event = exc.event
                    if renderer is not None:
                        renderer.render(self, exc.event)
                    break
                if renderer is not None and getattr(renderer, "is_closed", False):
                    break
        finally:
            self._max_time_steps = None
            if renderer is not None:
                renderer.close()

    def _should_stop(self, trips: int | None) -> bool:
        if self._time_limit_reached():
            return True
        if trips is not None and self.stats.trips_completed >= trips:
            return True
        return False

    def _travel(self, distance: int, paid: bool, move_target: int, repositioning: bool = False) -> None:
        for next_idx in self._path_to_target(move_target)[:distance]:
            self.driver_position = next_idx
            self.stats.time_steps += 1
            self.stats.balance -= self.config.move_cost_per_cell
            self.stats.total_move_cost += self.config.move_cost_per_cell
            if paid:
                self.stats.paid_distance += 1
            else:
                self.stats.empty_distance += 1
                if repositioning:
                    self.stats.reposition_distance += 1
            self._maybe_update_surge()
            self._check_time_limit()

    def _run_trip_with_tick_renders(self, renderer) -> None:
        if self.config.dispatch_mode == "direct_cell":
            self._run_direct_dispatch_with_tick_renders(renderer)
            return
        while True:
            preferred_cell = self.strategy.choose_wait_cell(self)
            self._set_trip_context(
                phase="waiting_offer",
                chosen_cell=preferred_cell,
                passenger_cell=None,
                destination_cell=None,
                passenger_onboard=False,
                offered_pickup_cell=None,
            )
            renderer.render(
                self,
                Event(
                    kind="target_selected",
                    message=f"Driver waits near preferred cell {preferred_cell}",
                    data={"chosen_cell": preferred_cell},
                ),
            )
            offer = self._generate_offer()
            if offer is not None:
                self.stats.offers_seen += 1
                self._set_trip_context(offered_pickup_cell=offer.pickup_cell)
                renderer.render(
                    self,
                    Event(
                        kind="offer_received",
                        message=(
                            f"Offer nearby: pickup={offer.pickup_cell}, dest={offer.destination}, "
                            f"surge={offer.surge:.2f}"
                        ),
                        data={"pickup_cell": offer.pickup_cell, "destination": offer.destination},
                    ),
                )
                if self.strategy.should_accept_offer(self, offer, preferred_cell):
                    self.stats.offers_accepted += 1
                    self._complete_offer_with_tick_renders(renderer, offer, preferred_cell)
                    return
                self.stats.offers_skipped += 1
                self._advance_toward_preferred_or_wait_with_tick_render(renderer, preferred_cell)
                renderer.render(
                    self,
                    Event(
                        kind="offer_skipped",
                        message=f"Skipped offer at cell {offer.pickup_cell}",
                        data={"pickup_cell": offer.pickup_cell},
                    ),
                )
                continue

            self._advance_toward_preferred_or_wait_with_tick_render(renderer, preferred_cell)

    def _travel_with_tick_renders(
        self, renderer, target: int, paid: bool, phase: str, repositioning: bool = False
    ) -> None:
        path = self._path_to_target(target)
        distance = len(path)
        if distance == 0:
            return
        for next_idx in path:
            prev_pos = self.driver_position
            self.driver_position = next_idx
            self.stats.time_steps += 1
            self.stats.balance -= self.config.move_cost_per_cell
            self.stats.total_move_cost += self.config.move_cost_per_cell
            if paid:
                self.stats.paid_distance += 1
            else:
                self.stats.empty_distance += 1
                if repositioning:
                    self.stats.reposition_distance += 1
            self._maybe_update_surge()
            self._set_trip_context(phase=phase)
            renderer.render(
                self,
                Event(
                    kind="tick_move",
                    message=(
                        f"Move {self.format_position(prev_pos)}->{self.format_position(self.driver_position)} "
                        f"({'paid' if paid else 'empty'})"
                    ),
                    data={
                        "from": prev_pos,
                        "to": self.driver_position,
                        "paid": paid,
                        "phase": phase,
                        "repositioning": repositioning,
                    },
                ),
            )
            self._check_time_limit()

    def _set_trip_context(
        self,
        *,
        phase: str | object = _UNSET,
        chosen_cell: int | None | object = _UNSET,
        passenger_cell: int | None | object = _UNSET,
        destination_cell: int | None | object = _UNSET,
        passenger_onboard: bool | object = _UNSET,
        offered_pickup_cell: int | None | object = _UNSET,
    ) -> None:
        if phase is not _UNSET:
            self.trip_context["phase"] = phase
        if chosen_cell is not _UNSET:
            self.trip_context["chosen_cell"] = chosen_cell
        if passenger_cell is not _UNSET:
            self.trip_context["passenger_cell"] = passenger_cell
        if destination_cell is not _UNSET:
            self.trip_context["destination_cell"] = destination_cell
        if passenger_onboard is not _UNSET:
            self.trip_context["passenger_onboard"] = passenger_onboard
        if offered_pickup_cell is not _UNSET:
            self.trip_context["offered_pickup_cell"] = offered_pickup_cell

    def _generate_offer(self) -> RideOffer | None:
        nearby_cells = [
            idx for idx in range(self.config.n_cells)
            if self.grid_distance(idx, self.driver_position) <= self.config.offer_radius
        ]
        if not nearby_cells:
            return None

        weights = []
        for idx in nearby_cells:
            distance = self.grid_distance(idx, self.driver_position)
            demand_weight = max(self.cells[idx].surge, 0.1)
            proximity_weight = max(0.4, self.config.offer_radius + 1 - distance)
            weights.append(demand_weight * proximity_weight)

        local_offer_pressure = sum(weights) / len(weights)
        offer_probability = min(0.95, self.config.offer_probability_scale * local_offer_pressure)
        if self.rng.random() > offer_probability:
            return None

        pickup = self.rng.choices(nearby_cells, weights=weights, k=1)[0]
        destination = self._random_destination(excluding=pickup)
        return RideOffer(
            pickup_cell=pickup,
            destination=destination,
            distance_to_pickup=self.grid_distance(self.driver_position, pickup),
            surge=self.cells[pickup].surge,
        )

    def _step_direct_dispatch(self) -> Event:
        pickup_cell = self.strategy.choose_pickup_cell(self)
        empty_distance = self.grid_distance(self.driver_position, pickup_cell)
        self._set_trip_context(
            phase="to_pickup",
            chosen_cell=pickup_cell,
            passenger_cell=pickup_cell,
            destination_cell=None,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        if empty_distance:
            self._travel(distance=empty_distance, paid=False, move_target=pickup_cell)

        destination = self._sample_destination_distribution(pickup_cell)
        trip_distance = self.grid_distance(destination, pickup_cell)
        trip_surge = self.cells[pickup_cell].surge
        revenue = trip_surge * self.config.tariff_per_cell * trip_distance

        self._set_trip_context(
            phase="to_destination",
            chosen_cell=pickup_cell,
            passenger_cell=None,
            destination_cell=destination,
            passenger_onboard=True,
            offered_pickup_cell=None,
        )
        self._travel(distance=trip_distance, paid=True, move_target=destination)
        self.stats.balance += revenue
        self.stats.total_revenue += revenue
        self.stats.trips_completed += 1

        event = Event(
            kind="trip_completed",
            message=(
                f"Trip #{self.stats.trips_completed}: pickup={pickup_cell}, dest={destination}, "
                f"empty={empty_distance}, trip={trip_distance}, surge={trip_surge:.2f}, "
                f"revenue=${revenue:.2f}, balance=${self.stats.balance:.2f}"
            ),
            data={
                "pickup_cell": pickup_cell,
                "destination": destination,
                "empty_distance": empty_distance,
                "trip_distance": trip_distance,
                "trip_surge": trip_surge,
                "revenue": revenue,
                "balance": self.stats.balance,
            },
        )
        self._set_trip_context(
            phase="idle",
            chosen_cell=pickup_cell,
            passenger_cell=None,
            destination_cell=destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        self.last_event = event
        return event

    def _run_direct_dispatch_with_tick_renders(self, renderer) -> None:
        pickup_cell = self.strategy.choose_pickup_cell(self)
        empty_distance = self.grid_distance(self.driver_position, pickup_cell)
        self._set_trip_context(
            phase="to_pickup",
            chosen_cell=pickup_cell,
            passenger_cell=pickup_cell,
            destination_cell=None,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        renderer.render(
            self,
            Event(
                kind="pickup_selected",
                message=f"Direct dispatch: driver chooses pickup cell {pickup_cell}",
                data={"pickup_cell": pickup_cell},
            ),
        )
        if empty_distance:
            self._travel_with_tick_renders(renderer, target=pickup_cell, paid=False, phase="to_pickup")

        destination = self._sample_destination_distribution(pickup_cell)
        trip_distance = self.grid_distance(destination, pickup_cell)
        trip_surge = self.cells[pickup_cell].surge
        revenue = trip_surge * self.config.tariff_per_cell * trip_distance
        self._set_trip_context(
            phase="to_destination",
            chosen_cell=pickup_cell,
            passenger_cell=None,
            destination_cell=destination,
            passenger_onboard=True,
            offered_pickup_cell=None,
        )
        renderer.render(
            self,
            Event(
                kind="destination_sampled",
                message=f"Destination sampled: {destination} from pickup {pickup_cell}",
                data={"pickup_cell": pickup_cell, "destination": destination},
            ),
        )
        self._travel_with_tick_renders(renderer, target=destination, paid=True, phase="to_destination")
        self.stats.balance += revenue
        self.stats.total_revenue += revenue
        self.stats.trips_completed += 1

        event = Event(
            kind="trip_completed",
            message=(
                f"Trip #{self.stats.trips_completed}: pickup={pickup_cell}, dest={destination}, "
                f"empty={empty_distance}, trip={trip_distance}, surge={trip_surge:.2f}, "
                f"revenue=${revenue:.2f}, balance=${self.stats.balance:.2f}"
            ),
            data={
                "pickup_cell": pickup_cell,
                "destination": destination,
                "empty_distance": empty_distance,
                "trip_distance": trip_distance,
                "trip_surge": trip_surge,
                "revenue": revenue,
                "balance": self.stats.balance,
            },
        )
        self._set_trip_context(
            phase="idle",
            chosen_cell=pickup_cell,
            passenger_cell=None,
            destination_cell=destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        self.last_event = event
        renderer.render(self, event)

    def _sample_destination_distribution(self, origin: int) -> int:
        candidates = [idx for idx in range(self.config.n_cells) if idx != origin]
        if not candidates:
            return origin

        sigma = max(1e-6, self.config.analytic_destination_sigma)
        weights = []
        for idx in candidates:
            ox, oy = self.index_to_coord(origin)
            dx, dy = self.index_to_coord(idx)
            sq_distance = (dx - ox) * (dx - ox) + (dy - oy) * (dy - oy)
            weights.append(math.exp(-(sq_distance) / (2.0 * sigma * sigma)))
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _complete_offer(self, offer: RideOffer, preferred_cell: int) -> Event:
        empty_distance = self.grid_distance(self.driver_position, offer.pickup_cell)
        self._set_trip_context(
            phase="to_pickup",
            chosen_cell=preferred_cell,
            passenger_cell=offer.pickup_cell,
            destination_cell=None,
            passenger_onboard=False,
            offered_pickup_cell=offer.pickup_cell,
        )
        if empty_distance:
            self._travel(distance=empty_distance, paid=False, move_target=offer.pickup_cell)

        self._set_trip_context(
            phase="to_destination",
            chosen_cell=preferred_cell,
            passenger_cell=offer.pickup_cell,
            destination_cell=offer.destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        trip_distance = self.grid_distance(offer.destination, offer.pickup_cell)
        trip_surge = self.cells[offer.pickup_cell].surge
        revenue = trip_surge * self.config.tariff_per_cell * trip_distance
        self._set_trip_context(
            phase="to_destination",
            chosen_cell=preferred_cell,
            passenger_cell=None,
            destination_cell=offer.destination,
            passenger_onboard=True,
            offered_pickup_cell=None,
        )
        self._travel(distance=trip_distance, paid=True, move_target=offer.destination)
        self.stats.balance += revenue
        self.stats.total_revenue += revenue
        self.stats.trips_completed += 1

        event = Event(
            kind="trip_completed",
            message=(
                f"Trip #{self.stats.trips_completed}: offer_pickup={offer.pickup_cell}, "
                f"dest={offer.destination}, empty={empty_distance}, trip={trip_distance}, "
                f"surge={trip_surge:.2f}, revenue=${revenue:.2f}, "
                f"balance=${self.stats.balance:.2f}"
            ),
            data={
                "pickup_cell": offer.pickup_cell,
                "destination": offer.destination,
                "empty_distance": empty_distance,
                "trip_distance": trip_distance,
                "trip_surge": trip_surge,
                "revenue": revenue,
                "balance": self.stats.balance,
                "preferred_cell": preferred_cell,
            },
        )
        self._set_trip_context(
            phase="idle",
            chosen_cell=preferred_cell,
            passenger_cell=None,
            destination_cell=offer.destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        self.last_event = event
        return event

    def _complete_offer_with_tick_renders(self, renderer, offer: RideOffer, preferred_cell: int) -> None:
        empty_distance = self.grid_distance(self.driver_position, offer.pickup_cell)
        self._set_trip_context(
            phase="to_pickup",
            chosen_cell=preferred_cell,
            passenger_cell=offer.pickup_cell,
            destination_cell=None,
            passenger_onboard=False,
            offered_pickup_cell=offer.pickup_cell,
        )
        self._travel_with_tick_renders(renderer, target=offer.pickup_cell, paid=False, phase="to_pickup")
        self._set_trip_context(
            phase="to_destination",
            chosen_cell=preferred_cell,
            passenger_cell=offer.pickup_cell,
            destination_cell=offer.destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        renderer.render(
            self,
            Event(
                kind="passenger_picked",
                message=f"Accepted nearby client at {offer.pickup_cell}; destination {offer.destination}",
                data={"pickup_cell": offer.pickup_cell, "destination": offer.destination},
            ),
        )

        trip_distance = self.grid_distance(offer.destination, offer.pickup_cell)
        trip_surge = self.cells[offer.pickup_cell].surge
        revenue = trip_surge * self.config.tariff_per_cell * trip_distance
        self._set_trip_context(
            phase="to_destination",
            chosen_cell=preferred_cell,
            passenger_cell=None,
            destination_cell=offer.destination,
            passenger_onboard=True,
            offered_pickup_cell=None,
        )
        self._travel_with_tick_renders(renderer, target=offer.destination, paid=True, phase="to_destination")
        self.stats.balance += revenue
        self.stats.total_revenue += revenue
        self.stats.trips_completed += 1

        event = Event(
            kind="trip_completed",
            message=(
                f"Trip #{self.stats.trips_completed}: offer_pickup={offer.pickup_cell}, "
                f"dest={offer.destination}, empty={empty_distance}, trip={trip_distance}, "
                f"surge={trip_surge:.2f}, revenue=${revenue:.2f}, "
                f"balance=${self.stats.balance:.2f}"
            ),
            data={
                "pickup_cell": offer.pickup_cell,
                "destination": offer.destination,
                "empty_distance": empty_distance,
                "trip_distance": trip_distance,
                "trip_surge": trip_surge,
                "revenue": revenue,
                "balance": self.stats.balance,
                "preferred_cell": preferred_cell,
            },
        )
        self._set_trip_context(
            phase="idle",
            chosen_cell=preferred_cell,
            passenger_cell=None,
            destination_cell=offer.destination,
            passenger_onboard=False,
            offered_pickup_cell=None,
        )
        self.last_event = event
        renderer.render(self, event)

    def _wait_tick(self) -> None:
        self.stats.time_steps += 1
        self.stats.waiting_ticks += 1
        self._maybe_update_surge()
        self._check_time_limit()

    def _advance_toward_preferred_or_wait(self, preferred_cell: int) -> None:
        if self.driver_position != preferred_cell:
            self._travel(distance=1, paid=False, move_target=preferred_cell, repositioning=True)
        else:
            self._wait_tick()

    def _advance_toward_preferred_or_wait_with_tick_render(self, renderer, preferred_cell: int) -> None:
        if self.driver_position != preferred_cell:
            self._travel_with_tick_renders(
                renderer, target=preferred_cell, paid=False, phase="repositioning", repositioning=True
            )
        else:
            self._wait_tick()
            renderer.render(
                self,
                Event(
                    kind="waiting",
                    message=f"Waiting at cell {self.driver_position} for next nearby offer",
                ),
            )

    def _maybe_update_surge(self) -> None:
        if self.stats.time_steps % self.config.surge_update_every_k_steps == 0:
            self.model.update(self.rng, self.cells, self.stats.time_steps)
            self.stats.surge_updates += 1

    def _time_limit_reached(self) -> bool:
        return self._max_time_steps is not None and self.stats.time_steps >= self._max_time_steps

    def _check_time_limit(self) -> None:
        if not self._time_limit_reached():
            return
        raise _TimeLimitReached(
            Event(
                kind="time_limit_reached",
                message=f"Time limit reached at tick {self.stats.time_steps}",
                data={"time_steps": self.stats.time_steps},
            )
        )

    def _random_destination(self, excluding: int) -> int:
        destination = excluding
        while destination == excluding:
            destination = self.rng.randrange(self.config.n_cells)
        return destination

    def _path_to_target(self, target: int) -> list[int]:
        if target == self.driver_position:
            return []
        x, y = self.index_to_coord(self.driver_position)
        tx, ty = self.index_to_coord(target)
        path: list[int] = []
        while x != tx:
            x += 1 if tx > x else -1
            path.append(self.coord_to_index(x, y))
        while y != ty:
            y += 1 if ty > y else -1
            path.append(self.coord_to_index(x, y))
        return path


def available_models() -> list[str]:
    return ["uniform_reset", "random_walk", "hotspot_shift", "commute_cycle", "analytic_wave"]


def available_strategies() -> list[str]:
    return ["greedy", "random", "nearest_max", "cost_aware", "epsilon_greedy", "smart", "policy_iteration", "q_learning", "saved_policy"]
