import unittest

from taxi_sim.learning import (
    AnalyticTaxiMDP,
    decomposed_bellman_update_smdp,
    naive_bellman_update_smdp,
    run_smdp_policy_iteration,
)


class SMDPLearningTests(unittest.TestCase):
    def test_decomposed_bellman_update_matches_naive_update(self):
        mdp = AnalyticTaxiMDP(
            n_cells=12,
            grid_width=12,
            grid_height=1,
            horizon_steps=8,
            analytic_dt=0.05,
            destination_sigma=1.5,
            move_cost_per_cell=1.0,
            tariff_per_cell=1.5,
        )
        values = {
            state: 0.01 * ((state[0] + 3 * state[1]) % 11)
            for state in mdp.iter_states()
        }

        naive = naive_bellman_update_smdp(mdp, values, discount=0.9)
        decomposed = decomposed_bellman_update_smdp(mdp, values, discount=0.9)

        self.assertLess(
            max(abs(naive[state] - decomposed[state]) for state in naive),
            1e-12,
        )

    def test_transition_reward_decomposition_matches_naive_update(self):
        mdp = AnalyticTaxiMDP(
            n_cells=12,
            grid_width=12,
            grid_height=1,
            horizon_steps=8,
            analytic_dt=0.05,
            destination_sigma=1.5,
            move_cost_per_cell=1.0,
            tariff_per_cell=1.5,
        )
        values = {
            state: 0.01 * ((state[0] + 3 * state[1]) % 11)
            for state in mdp.iter_states()
        }

        naive = naive_bellman_update_smdp(mdp, values, discount=0.9, reward_timing="transition")
        decomposed = decomposed_bellman_update_smdp(mdp, values, discount=0.9, reward_timing="transition")

        self.assertLess(
            max(abs(naive[state] - decomposed[state]) for state in naive),
            1e-12,
        )

    def test_truncated_policy_iteration_limits_actions(self):
        mdp = AnalyticTaxiMDP(
            n_cells=10,
            grid_width=10,
            grid_height=1,
            horizon_steps=6,
            analytic_dt=0.05,
            destination_sigma=1.5,
            move_cost_per_cell=1.0,
            tariff_per_cell=1.5,
        )

        result = run_smdp_policy_iteration(
            mdp,
            discount=0.9,
            action_radius=2,
            evaluation_sweeps=5,
            max_iterations=3,
        )

        for (position, _time_step), action in result.policy.actions.items():
            self.assertLessEqual(mdp.grid_distance(position, action), 2)


if __name__ == "__main__":
    unittest.main()
