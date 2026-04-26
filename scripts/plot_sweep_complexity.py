from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot exact loop-count complexity for one PI improvement sweep."
    )
    parser.add_argument("--min-n", type=int, default=10, help="Smallest 1D length / 2D side.")
    parser.add_argument("--max-n", type=int, default=100, help="Largest 1D length / 2D side.")
    parser.add_argument("--step", type=int, default=10, help="Step for N.")
    parser.add_argument(
        "--extended-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use N=10..100 step 10 and N=200..1000 step 100.",
    )
    parser.add_argument("--radius", type=int, default=15, help="Truncated action radius R.")
    parser.add_argument("--horizon", type=int, default=40, help="Periodic horizon H.")
    parser.add_argument(
        "--destination-radius",
        type=int,
        default=-1,
        help="Destination support radius. Default -1 means full destination support.",
    )
    parser.add_argument(
        "--output-dir",
        default="saved_policies/smdp_experiments",
        help="Where to write PNG and CSV outputs.",
    )
    parser.add_argument(
        "--dimension",
        choices=["both", "1d", "2d"],
        default="both",
        help="Which complexity graph(s) to compute.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress for every N and dimension.")
    return parser


def action_count_1d(position: int, n_cells: int, radius: int | None) -> int:
    if radius is None:
        return n_cells
    lo = max(0, position - radius)
    hi = min(n_cells - 1, position + radius)
    return hi - lo + 1


def action_count_2d(index: int, side: int, radius: int | None) -> int:
    n_cells = side * side
    if radius is None:
        return n_cells
    x = index % side
    y = index // side
    count = 0
    for yy in range(max(0, y - radius), min(side - 1, y + radius) + 1):
        remaining = radius - abs(yy - y)
        count += min(side - 1, x + remaining) - max(0, x - remaining) + 1
    return count


def destination_count_1d(origin: int, n_cells: int, destination_radius: int | None) -> int:
    if destination_radius is None:
        return n_cells - 1
    return action_count_1d(origin, n_cells, destination_radius) - 1


def destination_count_2d(origin: int, side: int, destination_radius: int | None) -> int:
    if destination_radius is None:
        return side * side - 1
    return action_count_2d(origin, side, destination_radius) - 1


def counts_1d(n: int, *, horizon: int, radius: int, destination_radius: int | None) -> dict[str, int]:
    n_cells = n
    full_actions = n_cells
    truncated_actions = sum(action_count_1d(x, n_cells, radius) for x in range(n_cells))
    full_destinations = sum(destination_count_1d(a, n_cells, destination_radius) for a in range(n_cells))
    destination_by_action = [
        destination_count_1d(a, n_cells, destination_radius)
        for a in range(n_cells)
    ]
    truncated_action_destination_pairs = sum(
        destination_by_action[a]
        for x in range(n_cells)
        for a in range(n_cells)
        if abs(x - a) <= radius
    )

    return {
        "classic_pi": horizon * n_cells * full_destinations,
        "truncated_pi": horizon * truncated_action_destination_pairs,
        "truncated_decomposition": horizon * full_destinations + horizon * truncated_actions,
    }


def counts_2d(side: int, *, horizon: int, radius: int, destination_radius: int | None) -> dict[str, int]:
    n_cells = side * side
    destination_by_action = [
        destination_count_2d(a, side, destination_radius)
        for a in range(n_cells)
    ]
    full_destinations = sum(destination_by_action)
    truncated_actions = sum(action_count_2d(x, side, radius) for x in range(n_cells))
    truncated_action_destination_pairs = 0
    for x in range(n_cells):
        xx = x % side
        xy = x // side
        for a in range(n_cells):
            ax = a % side
            ay = a // side
            if abs(xx - ax) + abs(xy - ay) <= radius:
                truncated_action_destination_pairs += destination_by_action[a]

    return {
        "classic_pi": horizon * n_cells * full_destinations,
        "truncated_pi": horizon * truncated_action_destination_pairs,
        "truncated_decomposition": horizon * full_destinations + horizon * truncated_actions,
    }


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, int | str]], *, dimension: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(row["N"]) for row in rows]
    series = [
        ("classic_pi", "classic PI"),
        ("truncated_pi", "truncated PI"),
        ("truncated_decomposition", "truncated PI + decomposition"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label in series:
        ax.plot(xs, [int(row[key]) for row in rows], marker="o", linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("N" if dimension == "1d" else "side length N for NxN grid")
    ax.set_ylabel("primitive loop count per improvement sweep, log scale")
    ax.set_title(f"One-sweep PI complexity, {dimension.upper()}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    destination_radius = None if args.destination_radius < 0 else args.destination_radius
    output_dir = Path(args.output_dir)
    if args.extended_grid:
        ns = list(range(10, 101, 10)) + list(range(200, 1001, 100))
    else:
        ns = list(range(args.min_n, args.max_n + 1, args.step))

    rows_1d = []
    rows_2d = []
    for n in ns:
        if args.dimension in {"both", "1d"} and args.verbose:
            print(f"[progress] 1D N={n}: counting loops", flush=True)
        if args.dimension in {"both", "1d"}:
            row_1d = {
                "dimension": "1d",
                "N": n,
                **counts_1d(n, horizon=args.horizon, radius=args.radius, destination_radius=destination_radius),
            }
            rows_1d.append(row_1d)
        if args.dimension in {"both", "2d"} and args.verbose:
            print(f"[progress] 2D {n}x{n}: counting loops", flush=True)
        if args.dimension in {"both", "2d"}:
            row_2d = {
                "dimension": "2d",
                "N": n,
                **counts_2d(n, horizon=args.horizon, radius=args.radius, destination_radius=destination_radius),
            }
            rows_2d.append(row_2d)

    if args.verbose:
        print("[progress] writing CSV files", flush=True)
    if rows_1d:
        write_csv(output_dir / "sweep_complexity_1d.csv", rows_1d)
    if rows_2d:
        write_csv(output_dir / "sweep_complexity_2d.csv", rows_2d)
    if rows_1d and args.verbose:
        print("[progress] plotting 1D graph", flush=True)
    if rows_1d:
        plot(rows_1d, dimension="1d", output_path=output_dir / "sweep_complexity_1d.png")
    if rows_2d and args.verbose:
        print("[progress] plotting 2D graph", flush=True)
    if rows_2d:
        plot(rows_2d, dimension="2d", output_path=output_dir / "sweep_complexity_2d.png")

    if rows_1d:
        print(f"Saved {output_dir / 'sweep_complexity_1d.png'}")
        print(f"Saved {output_dir / 'sweep_complexity_1d.csv'}")
    if rows_2d:
        print(f"Saved {output_dir / 'sweep_complexity_2d.png'}")
        print(f"Saved {output_dir / 'sweep_complexity_2d.csv'}")


if __name__ == "__main__":
    main()
