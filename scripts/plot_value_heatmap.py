from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot V-function heatmap(s) for saved taxi policies.")
    p.add_argument("policy_files", nargs="+", help="Path(s) to saved policy JSON files")
    p.add_argument("--output", type=str, default="value_heatmap.png", help="Output PNG path")
    p.add_argument("--figsize-scale", type=float, default=1.0, help="Global figure size multiplier")
    p.add_argument("--annotate", action="store_true", help="Draw rounded V values inside cells for small heatmaps")
    return p.parse_args()


def load_value_matrix(path: Path) -> tuple[list[list[float]], int, int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["values"]
    horizon_steps = int(payload["horizon_steps"])
    method = str(payload.get("method", "unknown"))

    max_position = max(int(item["position"]) for item in values)
    n_positions = max_position + 1
    matrix = [[0.0 for _ in range(horizon_steps)] for _ in range(n_positions)]

    for item in values:
        position = int(item["position"])
        time_step = int(item["time_step"])
        value = float(item["value"])
        matrix[position][time_step] = value

    return matrix, n_positions, horizon_steps, method


def plot_values(ax, matrix: list[list[float]], title: str, annotate: bool):
    n_positions = len(matrix)
    horizon_steps = len(matrix[0]) if matrix else 0
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Driver Position")
    ax.set_xticks(range(0, horizon_steps, max(1, horizon_steps // 10)))
    ax.set_yticks(range(0, n_positions, max(1, n_positions // 10)))

    if annotate and n_positions * horizon_steps <= 250:
        for position in range(n_positions):
            for time_step in range(horizon_steps):
                ax.text(
                    time_step,
                    position,
                    f"{matrix[position][time_step]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=5,
                )

    return image


def main() -> None:
    args = parse_args()
    policy_paths = [Path(path) for path in args.policy_files]
    loaded = [(path, *load_value_matrix(path)) for path in policy_paths]

    fig_width = max(8.0, 5.0 * len(loaded)) * args.figsize_scale
    fig_height = max(4.0, 0.35 * max(n_positions for _, _, n_positions, _, _ in loaded)) * args.figsize_scale
    fig, axes = plt.subplots(1, len(loaded), figsize=(fig_width, fig_height), squeeze=False)

    images = []
    for ax, (path, matrix, n_positions, horizon_steps, method) in zip(axes[0], loaded):
        title = f"{path.stem}\nmethod={method}, positions={n_positions}, horizon={horizon_steps}"
        images.append(plot_values(ax, matrix, title, args.annotate))

    divider = make_axes_locatable(axes[0, -1])
    cax = divider.append_axes("right", size="3.5%", pad=0.18)
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.ax.set_ylabel("V(s)", rotation=270, labelpad=14)

    fig.suptitle("Saved Policy Value Function", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 0.95))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved value heatmap to {output_path}")


if __name__ == "__main__":
    main()
