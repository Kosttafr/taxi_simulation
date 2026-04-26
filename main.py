from __future__ import annotations

import argparse

from taxi_sim.renderers import ConsoleRenderer, TkRenderer
from taxi_sim.learning import save_learned_policy
from taxi_sim.simulation import SimulationConfig, TaxiSimulation, available_models, available_strategies
from taxi_sim.stats import format_histogram, format_summary_table, histogram, run_batch, summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Taxi game simulation (1D demo field)")
    p.add_argument("--mode", choices=["headless", "console", "graphics"], default="graphics")
    p.add_argument("--trips", type=int, default=30, help="Number of completed trips to simulate")
    p.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help="Stop after this many simulation ticks instead of trip count",
    )
    p.add_argument("--cells", type=int, default=15, help="Field size (1D cells)")
    p.add_argument("--rows", type=int, default=1, help="Grid height; >1 enables 2D mode")
    p.add_argument("--k", type=int, default=3, help="Randomize supply/demand every k movement steps")
    p.add_argument("--move-cost", type=float, default=1.0, help="Cost per moved cell")
    p.add_argument("--tariff", type=float, default=4.0, help="Tariff per passenger cell")
    p.add_argument("--model", choices=available_models(), default="uniform_reset", help="Supply/demand randomization model")
    p.add_argument(
        "--dispatch-mode",
        choices=["nearby_offer", "direct_cell"],
        default="nearby_offer",
        help="How the driver gets passengers: nearby service offers or direct pickup from chosen cell",
    )
    p.add_argument(
        "--strategy",
        choices=available_strategies(),
        default="greedy",
        help="Driver strategy for choosing pickup cell",
    )
    p.add_argument(
        "--strategy-epsilon",
        type=float,
        default=0.15,
        help="Exploration rate for epsilon_greedy strategy (0..1)",
    )
    p.add_argument(
        "--strategy-distance-penalty",
        type=float,
        default=0.25,
        help="Distance penalty for cost_aware strategy",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    p.add_argument("--offer-radius", type=int, default=2, help="Nearby radius where service can offer a client")
    p.add_argument(
        "--offer-acceptance-radius",
        type=int,
        default=1,
        help="Accept an offered client if it is this close to the preferred cell for greedy-like strategies",
    )
    p.add_argument(
        "--offer-score-slack",
        type=float,
        default=0.15,
        help="How much worse an offer may be than the ideal target for cost_aware strategy",
    )
    p.add_argument(
        "--offer-probability-scale",
        type=float,
        default=0.45,
        help="Controls how often nearby offers appear",
    )
    p.add_argument(
        "--commute-mode",
        choices=["cycle", "morning", "day", "evening", "night"],
        default="cycle",
        help="For commute_cycle model: full daily loop or one fixed city period",
    )
    p.add_argument(
        "--commute-period-ticks",
        type=int,
        default=60,
        help="For commute_cycle model: ticks spent in each period before blending to the next one",
    )
    p.add_argument(
        "--analytic-dt",
        type=float,
        default=0.05,
        help="For analytic_wave model: mathematical time increment per simulation tick",
    )
    p.add_argument(
        "--analytic-destination-sigma",
        type=float,
        default=1.5,
        help="For direct analytic mode: stddev of destination normal distribution centered at pickup cell",
    )
    p.add_argument(
        "--rl-horizon-steps",
        type=int,
        default=40,
        help="For policy_iteration/q_learning: time horizon modulo used in the analytic control MDP",
    )
    p.add_argument(
        "--rl-discount",
        type=float,
        default=0.97,
        help="Discount factor for policy_iteration/q_learning",
    )
    p.add_argument(
        "--policy-iteration-evaluation-sweeps",
        type=int,
        default=80,
        help="Policy evaluation sweeps per policy iteration step",
    )
    p.add_argument(
        "--policy-iteration-max-iterations",
        type=int,
        default=80,
        help="Maximum policy improvement rounds",
    )
    p.add_argument(
        "--policy-iteration-tolerance",
        type=float,
        default=1e-9,
        help="Bellman iteration tolerance for policy evaluation in policy_iteration",
    )
    p.add_argument(
        "--q-learning-episodes",
        type=int,
        default=4000,
        help="Training episodes for q_learning strategy",
    )
    p.add_argument(
        "--q-learning-alpha",
        type=float,
        default=0.15,
        help="Learning rate for q_learning",
    )
    p.add_argument(
        "--q-learning-epsilon",
        type=float,
        default=0.25,
        help="Initial exploration rate for q_learning",
    )
    p.add_argument(
        "--q-learning-epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay for q_learning",
    )
    p.add_argument(
        "--q-learning-min-epsilon",
        type=float,
        default=0.02,
        help="Minimum exploration rate for q_learning",
    )
    p.add_argument(
        "--q-learning-episode-decisions",
        type=int,
        default=40,
        help="Decision count per q_learning training episode",
    )
    p.add_argument(
        "--learning-verbose",
        action="store_true",
        help="Print policy_iteration/q_learning training progress to console",
    )
    p.add_argument(
        "--learning-log-every",
        type=int,
        default=50,
        help="How often to print learning progress (eval sweeps for policy_iteration, episodes for q_learning)",
    )
    p.add_argument(
        "--save-policy",
        type=str,
        default=None,
        help="Save learned policy_iteration/q_learning policy to this JSON file",
    )
    p.add_argument(
        "--load-policy",
        type=str,
        default=None,
        help="Load a previously saved policy JSON file for strategy=saved_policy",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0.15,
        help="Delay between frames to slow game speed (seconds; headless ignores it)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Deprecated alias for --timeout",
    )
    p.add_argument(
        "--every-tick",
        action="store_true",
        help="Render each move by one cell (tick) instead of only after completed trips",
    )
    p.add_argument(
        "--batch-runs",
        type=int,
        default=1,
        help="Run many simulations in background and print aggregate statistics",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=12,
        help="Number of bins for printed distribution histogram in batch mode",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    frame_delay = args.timeout if args.delay is None else args.delay

    cfg = SimulationConfig(
        n_cells=args.cells * args.rows,
        field_dimension=1 if args.rows == 1 else 2,
        grid_width=args.cells,
        grid_height=args.rows,
        surge_update_every_k_steps=args.k,
        move_cost_per_cell=args.move_cost,
        tariff_per_cell=args.tariff,
        randomization_model=args.model,
        dispatch_mode=args.dispatch_mode,
        strategy_name=args.strategy,
        strategy_distance_penalty=args.strategy_distance_penalty,
        strategy_epsilon=args.strategy_epsilon,
        offer_radius=args.offer_radius,
        offer_acceptance_radius=args.offer_acceptance_radius,
        offer_score_slack=args.offer_score_slack,
        offer_probability_scale=args.offer_probability_scale,
        commute_mode=args.commute_mode,
        commute_period_ticks=args.commute_period_ticks,
        analytic_dt=args.analytic_dt,
        analytic_destination_sigma=args.analytic_destination_sigma,
        rl_horizon_steps=args.rl_horizon_steps,
        rl_discount=args.rl_discount,
        policy_iteration_evaluation_sweeps=args.policy_iteration_evaluation_sweeps,
        policy_iteration_max_iterations=args.policy_iteration_max_iterations,
        policy_iteration_tolerance=args.policy_iteration_tolerance,
        q_learning_episodes=args.q_learning_episodes,
        q_learning_alpha=args.q_learning_alpha,
        q_learning_epsilon=args.q_learning_epsilon,
        q_learning_epsilon_decay=args.q_learning_epsilon_decay,
        q_learning_min_epsilon=args.q_learning_min_epsilon,
        q_learning_episode_decisions=args.q_learning_episode_decisions,
        learning_verbose=args.learning_verbose,
        learning_log_every=args.learning_log_every,
        load_policy_path=args.load_policy,
        seed=args.seed,
    )

    if args.batch_runs > 1:
        trip_limit = None if args.max_time_steps is not None else args.trips
        results = run_batch(
            config=cfg,
            trips=trip_limit,
            max_time_steps=args.max_time_steps,
            runs=args.batch_runs,
            seed=args.seed,
        )
        balances = [r.balance for r in results]
        revenues = [r.revenue for r in results]
        time_steps = [float(r.time_steps) for r in results]
        empty_ratio = [
            0.0 if (r.empty_distance + r.paid_distance) == 0 else r.empty_distance / (r.empty_distance + r.paid_distance)
            for r in results
        ]

        if args.max_time_steps is not None:
            print(f"Batch mode: runs={args.batch_runs}, time_steps_per_run={args.max_time_steps}")
        else:
            print(f"Batch mode: runs={args.batch_runs}, trips_per_run={args.trips}")
        print(format_summary_table("Final balance", summarize(balances)))
        print(format_summary_table("Total revenue", summarize(revenues)))
        print(format_summary_table("Time steps", summarize(time_steps)))
        print(format_summary_table("Empty distance ratio", summarize(empty_ratio)))
        print(format_histogram("Balance", histogram(balances, bins=args.hist_bins)))
        return

    sim = TaxiSimulation(cfg)

    renderer = None
    if args.mode == "console":
        renderer = ConsoleRenderer(delay_seconds=max(0.0, frame_delay))
    elif args.mode == "graphics":
        renderer = TkRenderer(delay_ms=int(max(0.0, frame_delay) * 1000))

    trip_limit = None if args.max_time_steps is not None else args.trips
    sim.run(
        trips=trip_limit,
        max_time_steps=args.max_time_steps,
        renderer=renderer,
        render_each_tick=args.every_tick,
    )

    if args.save_policy is not None:
        if args.strategy not in {"policy_iteration", "q_learning"}:
            raise ValueError("--save-policy is supported only with --strategy policy_iteration or q_learning")
        policy = sim.get_learned_policy(args.strategy)
        save_learned_policy(policy, args.save_policy)
        print(f"Saved policy to {args.save_policy}")

    print("Final stats:")
    print(
        f"balance=${sim.stats.balance:.2f}, trips={sim.stats.trips_completed}, time_steps={sim.stats.time_steps}, "
        f"empty_distance={sim.stats.empty_distance}, paid_distance={sim.stats.paid_distance}, "
        f"revenue=${sim.stats.total_revenue:.2f}, move_cost=${sim.stats.total_move_cost:.2f}, "
        f"surge_updates={sim.stats.surge_updates}, offers_seen={sim.stats.offers_seen}, "
        f"offers_skipped={sim.stats.offers_skipped}, waiting_ticks={sim.stats.waiting_ticks}"
    )


if __name__ == "__main__":
    main()
