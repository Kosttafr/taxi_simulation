import unittest

from taxi_sim.simulation import SimulationConfig
from taxi_sim.stats import histogram, run_batch, summarize


class BatchStatsTests(unittest.TestCase):
    def test_run_batch_produces_requested_number_of_runs(self):
        cfg = SimulationConfig(n_cells=8, strategy_name="greedy", randomization_model="uniform_reset", seed=10)
        results = run_batch(config=cfg, trips=4, runs=5, seed=10)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.trips == 4 for r in results))

    def test_run_batch_can_stop_on_fixed_time_limit(self):
        cfg = SimulationConfig(
            n_cells=8,
            strategy_name="greedy",
            randomization_model="analytic_wave",
            dispatch_mode="direct_cell",
            seed=10,
        )
        results = run_batch(config=cfg, trips=None, max_time_steps=5, runs=3, seed=10)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.time_steps == 5 for r in results))

    def test_summarize_basic_metrics(self):
        s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 5.0)
        self.assertEqual(s["p50"], 3.0)
        self.assertAlmostEqual(s["mean"], 3.0)

    def test_histogram_counts_match_input_size(self):
        values = [1.0, 2.0, 2.5, 4.0, 8.0]
        hist = histogram(values, bins=3)
        self.assertEqual(sum(count for _, _, count in hist), len(values))


if __name__ == "__main__":
    unittest.main()
