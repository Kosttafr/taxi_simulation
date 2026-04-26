import random
import tempfile
import unittest

from taxi_sim.learning import load_learned_policy, save_learned_policy
from taxi_sim.models import AnalyticWaveModel, CommuteCycleModel
from taxi_sim.models import CellState
from taxi_sim.simulation import (
    CostAwareStrategy,
    GreedyStrategy,
    NearestMaxSurgeStrategy,
    RideOffer,
    SmartStrategy,
    SimulationConfig,
    TaxiSimulation,
)


class TaxiSimulationTests(unittest.TestCase):
    class _ChaseStrategy:
        def __init__(self) -> None:
            self._accept_on_second_offer = False

        def choose_wait_cell(self, sim):
            return sim.config.n_cells - 1

        def choose_pickup_cell(self, sim):
            return self.choose_wait_cell(sim)

        def should_accept_offer(self, sim, offer, target_cell):
            if self._accept_on_second_offer:
                return True
            self._accept_on_second_offer = True
            return False

    def test_greedy_strategy_picks_biggest_surge(self):
        sim = TaxiSimulation(SimulationConfig(n_cells=3, seed=1, randomization_model="uniform_reset"))
        sim.cells = [
            CellState(supply=1.0, demand=2.0),  # surge 0.5
            CellState(supply=3.0, demand=1.0),  # surge 3.0
            CellState(supply=2.0, demand=2.0),  # surge 1.0
        ]
        self.assertEqual(GreedyStrategy().choose_pickup_cell(sim), 1)

    def test_nearest_max_surge_prefers_closest_peak(self):
        sim = TaxiSimulation(SimulationConfig(n_cells=5, seed=1, randomization_model="uniform_reset"))
        sim.driver_position = 2
        sim.cells = [
            CellState(supply=4.0, demand=1.0),  # surge 4.0
            CellState(supply=2.0, demand=2.0),  # surge 1.0
            CellState(supply=1.0, demand=2.0),  # surge 0.5
            CellState(supply=1.0, demand=2.0),  # surge 0.5
            CellState(supply=4.0, demand=1.0),  # surge 4.0
        ]
        # Both edges have equal max surge; cell 0 is closer from position 2.
        self.assertEqual(NearestMaxSurgeStrategy().choose_pickup_cell(sim), 0)

    def test_cost_aware_can_prefer_near_cell(self):
        sim = TaxiSimulation(SimulationConfig(n_cells=3, seed=1, randomization_model="uniform_reset"))
        sim.driver_position = 1
        sim.cells = [
            CellState(supply=6.0, demand=1.0),   # surge 6.0, distance 1
            CellState(supply=5.2, demand=1.0),   # surge 5.2, distance 0
            CellState(supply=2.0, demand=1.0),   # surge 2.0, distance 1
        ]
        # With high penalty, staying in place is better than moving for slightly higher surge.
        self.assertEqual(CostAwareStrategy(distance_penalty=1.5).choose_pickup_cell(sim), 1)

    def test_simulation_builds_strategy_from_config(self):
        sim = TaxiSimulation(SimulationConfig(n_cells=8, strategy_name="random", seed=5))
        self.assertEqual(type(sim.strategy).__name__, "RandomStrategy")

    def test_smart_strategy_chooses_cell_with_best_expected_value_minus_approach_cost(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_dt=0.5,
                analytic_destination_sigma=1.2,
                tariff_per_cell=4.0,
                move_cost_per_cell=1.0,
                seed=5,
            )
        )
        sim.driver_position = 4

        scores = [
            sim.expected_trip_value_from_cell(idx) - sim.config.move_cost_per_cell * abs(sim.driver_position - idx)
            for idx in range(sim.config.n_cells)
        ]
        self.assertEqual(SmartStrategy().choose_pickup_cell(sim), max(range(sim.config.n_cells), key=lambda idx: scores[idx]))

    def test_smart_strategy_requires_analytic_wave_model(self):
        for strategy_name in ("smart", "policy_iteration", "q_learning"):
            with self.assertRaises(ValueError):
                TaxiSimulation(
                    SimulationConfig(
                        n_cells=5,
                        randomization_model="uniform_reset",
                        strategy_name=strategy_name,
                        seed=1,
                    )
                )

    def test_policy_iteration_matches_one_step_optimum_when_discount_is_zero(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_dt=0.25,
                analytic_destination_sigma=1.0,
                move_cost_per_cell=1.0,
                tariff_per_cell=4.0,
                rl_horizon_steps=8,
                rl_discount=0.0,
                policy_iteration_evaluation_sweeps=20,
                policy_iteration_max_iterations=20,
                seed=11,
            )
        )
        sim.driver_position = 3
        sim.stats.time_steps = 2
        sim.model.update(sim.rng, sim.cells, sim.stats.time_steps)

        expected_scores = []
        for idx in range(sim.config.n_cells):
            expected_trip_distance = sim.expected_destination_distance_from_cell(idx)
            score = (
                sim.cells[idx].surge * sim.config.tariff_per_cell * expected_trip_distance
                - sim.config.move_cost_per_cell * abs(sim.driver_position - idx)
                - sim.config.move_cost_per_cell * expected_trip_distance
            )
            expected_scores.append(score)

        policy = sim.get_learned_policy("policy_iteration")
        chosen = policy.action_for(sim.driver_position, sim.stats.time_steps)
        self.assertEqual(chosen, max(range(sim.config.n_cells), key=lambda idx: expected_scores[idx]))

    def test_q_learning_learns_same_greedy_action_in_simple_discount_zero_case(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_dt=0.25,
                analytic_destination_sigma=1.0,
                move_cost_per_cell=1.0,
                tariff_per_cell=4.0,
                rl_horizon_steps=8,
                rl_discount=0.0,
                q_learning_episodes=2500,
                q_learning_alpha=0.2,
                q_learning_epsilon=0.3,
                q_learning_epsilon_decay=0.997,
                q_learning_min_epsilon=0.02,
                q_learning_episode_decisions=10,
                seed=17,
            )
        )
        sim.driver_position = 1
        sim.stats.time_steps = 0
        sim.model.update(sim.rng, sim.cells, sim.stats.time_steps)

        expected_scores = []
        for idx in range(sim.config.n_cells):
            expected_trip_distance = sim.expected_destination_distance_from_cell(idx)
            score = (
                sim.cells[idx].surge * sim.config.tariff_per_cell * expected_trip_distance
                - sim.config.move_cost_per_cell * abs(sim.driver_position - idx)
                - sim.config.move_cost_per_cell * expected_trip_distance
            )
            expected_scores.append(score)

        policy = sim.get_learned_policy("q_learning")
        chosen = policy.action_for(sim.driver_position, sim.stats.time_steps)
        self.assertEqual(chosen, max(range(sim.config.n_cells), key=lambda idx: expected_scores[idx]))

    def test_saved_policy_roundtrip_and_replay_strategy(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                strategy_name="policy_iteration",
                analytic_dt=0.25,
                analytic_destination_sigma=1.0,
                rl_horizon_steps=8,
                policy_iteration_evaluation_sweeps=20,
                policy_iteration_max_iterations=20,
                seed=19,
            )
        )
        learned = sim.get_learned_policy("policy_iteration")

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
            save_learned_policy(learned, tmp.name)
            restored = load_learned_policy(tmp.name)

            self.assertEqual(restored.horizon_steps, learned.horizon_steps)
            self.assertEqual(restored.actions, learned.actions)

            replay_sim = TaxiSimulation(
                SimulationConfig(
                    n_cells=5,
                    randomization_model="analytic_wave",
                    dispatch_mode="direct_cell",
                    strategy_name="saved_policy",
                    analytic_dt=0.25,
                    analytic_destination_sigma=1.0,
                    rl_horizon_steps=8,
                    load_policy_path=tmp.name,
                    seed=19,
                )
            )
            replay_sim.driver_position = 2
            replay_sim.stats.time_steps = 3
            self.assertEqual(
                replay_sim.strategy.choose_pickup_cell(replay_sim),
                learned.action_for(replay_sim.driver_position, replay_sim.stats.time_steps),
            )

    def test_commute_cycle_prefers_outer_then_center(self):
        model = CommuteCycleModel(period_ticks=10, noise=0.0, mode="cycle")
        rng = random.Random(1)
        cells = model.initialize(rng, 5)
        morning_surges = [cell.surge for cell in cells]
        model.update(rng, cells, tick=20)
        evening_surges = [cell.surge for cell in cells]

        self.assertGreater(morning_surges[0], morning_surges[2])
        self.assertGreater(morning_surges[4], morning_surges[2])
        self.assertGreater(evening_surges[2], evening_surges[0])
        self.assertGreater(evening_surges[2], evening_surges[4])

    def test_commute_cycle_supports_fixed_period_mode(self):
        model = CommuteCycleModel(noise=0.0, mode="night")
        rng = random.Random(1)
        cells = model.initialize(rng, 7)
        surges = [cell.surge for cell in cells]

        self.assertLess(max(surges) - min(surges), 0.15)
        self.assertLess(max(surges), 0.60)

    def test_analytic_wave_matches_formula(self):
        model = AnalyticWaveModel(dt=0.5, grid_width=3, grid_height=1)
        rng = random.Random(1)
        cells = model.initialize(rng, 3)
        surges = [cell.surge for cell in cells]

        self.assertAlmostEqual(surges[0], 3.0)
        self.assertAlmostEqual(surges[1], 2.0)
        self.assertAlmostEqual(surges[2], 1.0)

        model.update(rng, cells, tick=1)
        shifted_surges = [cell.surge for cell in cells]
        self.assertAlmostEqual(shifted_surges[0], 2.0)
        self.assertAlmostEqual(shifted_surges[1], 1.0)
        self.assertAlmostEqual(shifted_surges[2], 2.0)

    def test_analytic_wave_supports_2d_grid_formula(self):
        model = AnalyticWaveModel(dt=0.0, grid_width=2, grid_height=2)
        rng = random.Random(1)
        cells = model.initialize(rng, 4)
        surges = [cell.surge for cell in cells]

        self.assertAlmostEqual(surges[0], 3.0)
        self.assertAlmostEqual(surges[1], 3.0)
        self.assertAlmostEqual(surges[2], 3.0)
        self.assertAlmostEqual(surges[3], 1.0)

    def test_offer_based_loop_tracks_skipped_offers_and_waiting(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=6,
                randomization_model="commute_cycle",
                strategy_name="greedy",
                offer_radius=1,
                offer_acceptance_radius=0,
                offer_probability_scale=0.25,
                seed=4,
            )
        )
        event = sim.step()
        self.assertEqual(event.kind, "trip_completed")
        self.assertGreaterEqual(sim.stats.offers_seen, 1)
        self.assertGreaterEqual(sim.stats.offers_accepted, 1)
        self.assertGreaterEqual(sim.stats.waiting_ticks + sim.stats.reposition_distance, 0)

    def test_skipped_offer_does_not_stop_repositioning(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="uniform_reset",
                seed=1,
            ),
            strategy=self._ChaseStrategy(),
        )

        offers = iter(
            [
                RideOffer(pickup_cell=0, destination=2, distance_to_pickup=0, surge=0.5),
                RideOffer(pickup_cell=1, destination=3, distance_to_pickup=0, surge=1.0),
            ]
        )
        sim._generate_offer = lambda: next(offers)

        event = sim.step()

        self.assertEqual(event.kind, "trip_completed")
        self.assertEqual(sim.stats.offers_skipped, 1)
        self.assertEqual(sim.stats.reposition_distance, 1)

    def test_simulation_runs_headless_and_updates_stats(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=10,
                surge_update_every_k_steps=3,
                move_cost_per_cell=1.0,
                tariff_per_cell=4.0,
                randomization_model="random_walk",
                seed=123,
            )
        )
        sim.run(trips=5, renderer=None)

        self.assertEqual(sim.stats.trips_completed, 5)
        self.assertEqual(
            sim.stats.time_steps,
            sim.stats.empty_distance + sim.stats.paid_distance + sim.stats.waiting_ticks,
        )
        self.assertEqual(
            sim.stats.total_move_cost,
            (sim.stats.empty_distance + sim.stats.paid_distance) * sim.config.move_cost_per_cell,
        )
        self.assertEqual(
            sim.stats.surge_updates,
            sim.stats.time_steps // sim.config.surge_update_every_k_steps,
        )
        self.assertGreaterEqual(sim.stats.offers_seen, sim.stats.offers_accepted)

    def test_simulation_can_stop_on_fixed_time_limit(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=7,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_dt=0.05,
                seed=7,
            )
        )

        sim.run(trips=None, max_time_steps=3, renderer=None)

        self.assertEqual(sim.stats.time_steps, 3)
        self.assertLessEqual(sim.stats.trips_completed, 1)
        self.assertEqual(sim.stats.time_steps, sim.stats.empty_distance + sim.stats.paid_distance + sim.stats.waiting_ticks)

    def test_direct_dispatch_mode_uses_chosen_cell_without_offers(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=5,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_dt=0.5,
                seed=2,
            )
        )
        expected_pickup = GreedyStrategy().choose_pickup_cell(sim)
        event = sim.step()

        self.assertEqual(event.kind, "trip_completed")
        self.assertEqual(sim.stats.offers_seen, 0)
        self.assertEqual(event.data["pickup_cell"], expected_pickup)

    def test_destination_distribution_is_centered_near_origin(self):
        sim = TaxiSimulation(
            SimulationConfig(
                n_cells=9,
                randomization_model="uniform_reset",
                dispatch_mode="direct_cell",
                analytic_destination_sigma=1.2,
                seed=3,
            )
        )
        origin = 4
        samples = [sim._sample_destination_distribution(origin=origin) for _ in range(500)]
        mean_destination = sum(samples) / len(samples)
        mean_abs_distance = sum(abs(x - origin) for x in samples) / len(samples)

        self.assertGreater(mean_destination, 3.3)
        self.assertLess(mean_destination, 4.7)
        self.assertLess(mean_abs_distance, 1.5)

    def test_destination_distribution_is_2d_centered(self):
        sim = TaxiSimulation(
            SimulationConfig(
                field_dimension=2,
                grid_width=3,
                grid_height=3,
                randomization_model="analytic_wave",
                dispatch_mode="direct_cell",
                analytic_destination_sigma=1.0,
                seed=5,
            )
        )
        origin = sim.coord_to_index(1, 1)
        samples = [sim._sample_destination_distribution(origin=origin) for _ in range(400)]
        mean_distance = sum(sim.grid_distance(origin, idx) for idx in samples) / len(samples)
        self.assertLess(mean_distance, 1.8)

    def test_destination_is_never_same_as_pickup(self):
        sim = TaxiSimulation(SimulationConfig(n_cells=2, seed=42))
        for _ in range(30):
            event = sim.step()
            self.assertNotEqual(event.data["pickup_cell"], event.data["destination"])


if __name__ == "__main__":
    unittest.main()
