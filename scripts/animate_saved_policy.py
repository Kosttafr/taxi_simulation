from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate saved taxi policy/value maps over time.")
    p.add_argument("policy_file", help="Path to saved policy JSON")
    p.add_argument(
        "--kind",
        choices=["policy", "value", "both", "combined"],
        default="both",
        help="What to animate",
    )
    p.add_argument("--rows", type=int, default=None, help="Grid height")
    p.add_argument("--cols", type=int, default=None, help="Grid width")
    p.add_argument("--analytic-dt", type=float, default=0.05, help="Time step used in analytic surge formula")
    p.add_argument("--fps", type=int, default=4, help="Frames per second for GIF")
    p.add_argument("--dpi", type=int, default=160, help="Output DPI")
    p.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix; defaults to policy filename stem in same folder",
    )
    p.add_argument("--annotate", action="store_true", help="Draw values/actions inside cells for small grids")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_file)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    horizon_steps = int(payload["horizon_steps"])
    actions = payload["actions"]
    values = payload["values"]

    n_positions = max(int(item["position"]) for item in actions) + 1
    cols, rows = resolve_grid_shape(n_positions, args.cols, args.rows)
    output_prefix = Path(args.output_prefix) if args.output_prefix else policy_path.with_suffix("")

    policy_frames = build_frames(actions, rows, cols, horizon_steps, key="action", default_value=0.0)
    value_frames = build_frames(values, rows, cols, horizon_steps, key="value", default_value=0.0)
    surge_frames = build_analytic_surge_frames(rows=rows, cols=cols, horizon_steps=horizon_steps, analytic_dt=args.analytic_dt)

    if args.kind in {"policy", "both"}:
        save_policy_animation(
            frames=policy_frames,
            title_prefix=f"{policy_path.stem} | policy",
            output_path=output_prefix.parent / f"{output_prefix.name}_policy.gif",
            fps=args.fps,
            dpi=args.dpi,
            annotate=args.annotate,
            rows=rows,
            cols=cols,
        )
    if args.kind in {"value", "both"}:
        save_animation(
            frames=value_frames,
            title_prefix=f"{policy_path.stem} | value",
            colorbar_label="V(s)",
            output_path=output_prefix.parent / f"{output_prefix.name}_value.gif",
            fps=args.fps,
            dpi=args.dpi,
            annotate=args.annotate,
            discrete=False,
        )
    if args.kind == "combined":
        save_combined_animation(
            policy_frames=policy_frames,
            value_frames=value_frames,
            surge_frames=surge_frames,
            title_prefix=policy_path.stem,
            output_path=output_prefix.parent / f"{output_prefix.name}_combined.gif",
            fps=args.fps,
            dpi=args.dpi,
            annotate=args.annotate,
            rows=rows,
            cols=cols,
        )


def resolve_grid_shape(n_positions: int, cols: int | None, rows: int | None) -> tuple[int, int]:
    if cols is not None and rows is not None:
        if cols * rows != n_positions:
            raise ValueError(f"cols*rows must equal number of positions ({n_positions})")
        return cols, rows
    if cols is not None:
        if n_positions % cols != 0:
            raise ValueError("n_positions must be divisible by cols")
        return cols, n_positions // cols
    if rows is not None:
        if n_positions % rows != 0:
            raise ValueError("n_positions must be divisible by rows")
        return n_positions // rows, rows

    side = int(math.isqrt(n_positions))
    if side * side == n_positions:
        return side, side
    raise ValueError("Could not infer grid shape; provide --rows and --cols")


def build_frames(
    entries: list[dict[str, object]],
    rows: int,
    cols: int,
    horizon_steps: int,
    *,
    key: str,
    default_value: float,
) -> list[list[list[float]]]:
    frames = [[[default_value for _ in range(cols)] for _ in range(rows)] for _ in range(horizon_steps)]
    for item in entries:
        position = int(item["position"])
        time_step = int(item["time_step"])
        x = position % cols
        y = position // cols
        frames[time_step][y][x] = float(item[key])
    return frames


def build_analytic_surge_frames(*, rows: int, cols: int, horizon_steps: int, analytic_dt: float) -> list[list[list[float]]]:
    frames = []
    safe_dt = max(1e-6, analytic_dt)
    for time_step in range(horizon_steps):
        time_value = time_step * safe_dt
        frame = []
        for y in range(rows):
            row = []
            for x in range(cols):
                x_phase = 0.0 if cols == 1 else x / (cols - 1)
                y_phase = 1.0 if rows == 1 else y / (rows - 1)
                surge = math.cos(math.pi * x_phase * y_phase + math.pi * time_value) + 2.0
                row.append(surge)
            frame.append(row)
        frames.append(frame)
    return frames


def save_policy_animation(
    *,
    frames: list[list[list[float]]],
    title_prefix: str,
    output_path: Path,
    fps: int,
    dpi: int,
    annotate: bool,
    rows: int,
    cols: int,
) -> None:
    fig_w = max(5.0, cols * 0.55)
    fig_h = max(4.5, rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_coords = list(range(cols))
    y_coords = list(range(rows))
    X, Y = np.meshgrid(x_coords, y_coords)

    u_frames: list[list[list[float]]] = []
    v_frames: list[list[list[float]]] = []
    distance_frames: list[list[list[float]]] = []
    display_u_frames: list[list[list[float]]] = []
    display_v_frames: list[list[list[float]]] = []
    for frame in frames:
        u_frame = []
        v_frame = []
        d_frame = []
        display_u_frame = []
        display_v_frame = []
        for y in range(rows):
            u_row = []
            v_row = []
            d_row = []
            display_u_row = []
            display_v_row = []
            for x in range(cols):
                action_idx = int(round(frame[y][x]))
                action_x = action_idx % cols
                action_y = action_idx // cols
                dx = action_x - x
                dy = action_y - y
                distance = abs(dx) + abs(dy)
                euclidean = math.hypot(dx, dy)
                if euclidean > 1e-9:
                    scale = min(0.85, 0.85 / euclidean)
                    display_dx = dx * scale
                    display_dy = dy * scale
                else:
                    display_dx = 0.0
                    display_dy = 0.0
                u_row.append(dx)
                v_row.append(dy)
                d_row.append(distance)
                display_u_row.append(display_dx)
                display_v_row.append(display_dy)
            u_frame.append(u_row)
            v_frame.append(v_row)
            d_frame.append(d_row)
            display_u_frame.append(display_u_row)
            display_v_frame.append(display_v_row)
        u_frames.append(u_frame)
        v_frames.append(v_frame)
        distance_frames.append(d_frame)
        display_u_frames.append(display_u_frame)
        display_v_frames.append(display_v_frame)

    max_distance = max(max(max(row) for row in frame) for frame in distance_frames) or 1.0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=max_distance)
    quiver = ax.quiver(
        X,
        Y,
        display_u_frames[0],
        display_v_frames[0],
        distance_frames[0],
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap=cmap,
        norm=norm,
        pivot="mid",
        width=0.0045,
        headwidth=4.0,
        headlength=5.5,
        headaxislength=4.5,
    )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(range(0, cols, max(1, cols // 8)))
    ax.set_yticks(range(0, rows, max(1, rows // 8)))
    ax.grid(color="#d1d5db", linewidth=0.5, alpha=0.4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.18)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.set_ylabel("Move Distance", rotation=270, labelpad=14)

    text_overlays = []
    if annotate and rows * cols <= 225:
        for y in range(rows):
            row_items = []
            for x in range(cols):
                text = ax.text(x, y, "", ha="center", va="center", color="black", fontsize=5)
                row_items.append(text)
            text_overlays.append(row_items)

    def update(frame_idx: int):
        quiver.set_UVC(display_u_frames[frame_idx], display_v_frames[frame_idx], distance_frames[frame_idx])
        quiver.set_clim(0.0, max_distance)
        ax.set_title(f"{title_prefix} | time_step={frame_idx}")
        if text_overlays:
            for y in range(rows):
                for x in range(cols):
                    action_idx = int(round(frames[frame_idx][y][x]))
                    action_x = action_idx % cols
                    action_y = action_idx // cols
                    text_overlays[y][x].set_text(f"{action_x},{action_y}")
        artists = [quiver]
        for row_items in text_overlays:
            artists.extend(row_items)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def save_animation(
    *,
    frames: list[list[list[float]]],
    title_prefix: str,
    colorbar_label: str,
    output_path: Path,
    fps: int,
    dpi: int,
    annotate: bool,
    discrete: bool,
) -> None:
    rows = len(frames[0])
    cols = len(frames[0][0])
    fig_w = max(5.0, cols * 0.55)
    fig_h = max(4.5, rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin = min(min(min(row) for row in frame) for frame in frames)
    vmax = max(max(max(row) for row in frame) for frame in frames)
    if discrete:
        max_action = int(vmax)
        cmap = plt.get_cmap("viridis", max_action + 1)
        norm = BoundaryNorm(range(max_action + 2), cmap.N)
        image = ax.imshow(frames[0], origin="lower", cmap=cmap, norm=norm, aspect="auto")
    else:
        image = ax.imshow(frames[0], origin="lower", cmap="magma", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(range(0, cols, max(1, cols // 8)))
    ax.set_yticks(range(0, rows, max(1, rows // 8)))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.18)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=14)

    text_overlays = []
    if annotate and rows * cols <= 225:
        for y in range(rows):
            row_items = []
            for x in range(cols):
                text = ax.text(x, y, "", ha="center", va="center", color="white", fontsize=6)
                row_items.append(text)
            text_overlays.append(row_items)

    def update(frame_idx: int):
        frame = frames[frame_idx]
        image.set_data(frame)
        image.set_clim(vmin, vmax)
        ax.set_title(f"{title_prefix} | time_step={frame_idx}")
        if text_overlays:
            for y in range(rows):
                for x in range(cols):
                    value = frame[y][x]
                    text_overlays[y][x].set_text(str(int(round(value))) if discrete else f"{value:.1f}")
        artists = [image]
        for row_items in text_overlays:
            artists.extend(row_items)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def save_combined_animation(
    *,
    policy_frames: list[list[list[float]]],
    value_frames: list[list[list[float]]],
    surge_frames: list[list[list[float]]],
    title_prefix: str,
    output_path: Path,
    fps: int,
    dpi: int,
    annotate: bool,
    rows: int,
    cols: int,
) -> None:
    fig_w = max(14.0, cols * 1.45)
    fig_h = max(4.5, rows * 0.5)
    fig, (ax_policy, ax_value, ax_surge) = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    x_coords = list(range(cols))
    y_coords = list(range(rows))
    X, Y = np.meshgrid(x_coords, y_coords)

    u_frames: list[list[list[float]]] = []
    v_frames: list[list[list[float]]] = []
    distance_frames: list[list[list[float]]] = []
    display_u_frames: list[list[list[float]]] = []
    display_v_frames: list[list[list[float]]] = []
    for frame in policy_frames:
        u_frame = []
        v_frame = []
        d_frame = []
        display_u_frame = []
        display_v_frame = []
        for y in range(rows):
            u_row = []
            v_row = []
            d_row = []
            display_u_row = []
            display_v_row = []
            for x in range(cols):
                action_idx = int(round(frame[y][x]))
                action_x = action_idx % cols
                action_y = action_idx // cols
                dx = action_x - x
                dy = action_y - y
                distance = abs(dx) + abs(dy)
                euclidean = math.hypot(dx, dy)
                if euclidean > 1e-9:
                    scale = min(0.85, 0.85 / euclidean)
                    display_dx = dx * scale
                    display_dy = dy * scale
                else:
                    display_dx = 0.0
                    display_dy = 0.0
                u_row.append(dx)
                v_row.append(dy)
                d_row.append(distance)
                display_u_row.append(display_dx)
                display_v_row.append(display_dy)
            u_frame.append(u_row)
            v_frame.append(v_row)
            d_frame.append(d_row)
            display_u_frame.append(display_u_row)
            display_v_frame.append(display_v_row)
        u_frames.append(u_frame)
        v_frames.append(v_frame)
        distance_frames.append(d_frame)
        display_u_frames.append(display_u_frame)
        display_v_frames.append(display_v_frame)

    max_distance = max(max(max(row) for row in frame) for frame in distance_frames) or 1.0
    policy_cmap = plt.get_cmap("viridis")
    policy_norm = plt.Normalize(vmin=0.0, vmax=max_distance)
    quiver = ax_policy.quiver(
        X,
        Y,
        display_u_frames[0],
        display_v_frames[0],
        distance_frames[0],
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap=policy_cmap,
        norm=policy_norm,
        pivot="mid",
        width=0.0045,
        headwidth=4.0,
        headlength=5.5,
        headaxislength=4.5,
    )
    ax_policy.set_xlim(-0.5, cols - 0.5)
    ax_policy.set_ylim(-0.5, rows - 0.5)
    ax_policy.set_aspect("equal")
    ax_policy.set_xlabel("X")
    ax_policy.set_ylabel("Y")
    ax_policy.grid(color="#d1d5db", linewidth=0.5, alpha=0.4)

    value_vmin = min(min(min(row) for row in frame) for frame in value_frames)
    value_vmax = max(max(max(row) for row in frame) for frame in value_frames)
    value_image = ax_value.imshow(value_frames[0], origin="lower", cmap="magma", vmin=value_vmin, vmax=value_vmax, aspect="auto")
    ax_value.set_xlabel("X")
    ax_value.set_ylabel("Y")

    surge_vmin = min(min(min(row) for row in frame) for frame in surge_frames)
    surge_vmax = max(max(max(row) for row in frame) for frame in surge_frames)
    surge_image = ax_surge.imshow(surge_frames[0], origin="lower", cmap="RdYlGn_r", vmin=surge_vmin, vmax=surge_vmax, aspect="auto")
    ax_surge.set_xlabel("X")
    ax_surge.set_ylabel("Y")

    divider_policy = make_axes_locatable(ax_policy)
    cax_policy = divider_policy.append_axes("right", size="4%", pad=0.15)
    cbar_policy = fig.colorbar(ScalarMappable(norm=policy_norm, cmap=policy_cmap), cax=cax_policy)
    cbar_policy.ax.set_ylabel("Move Distance", rotation=270, labelpad=14)

    divider_value = make_axes_locatable(ax_value)
    cax_value = divider_value.append_axes("right", size="4%", pad=0.15)
    cbar_value = fig.colorbar(value_image, cax=cax_value)
    cbar_value.ax.set_ylabel("V(s)", rotation=270, labelpad=14)

    divider_surge = make_axes_locatable(ax_surge)
    cax_surge = divider_surge.append_axes("right", size="4%", pad=0.15)
    cbar_surge = fig.colorbar(surge_image, cax=cax_surge)
    cbar_surge.ax.set_ylabel("Surge", rotation=270, labelpad=14)

    policy_text_overlays = []
    value_text_overlays = []
    surge_text_overlays = []
    if annotate and rows * cols <= 225:
        for y in range(rows):
            policy_row_items = []
            value_row_items = []
            surge_row_items = []
            for x in range(cols):
                policy_text = ax_policy.text(x, y, "", ha="center", va="center", color="black", fontsize=5)
                value_text = ax_value.text(x, y, "", ha="center", va="center", color="white", fontsize=5)
                surge_text = ax_surge.text(x, y, "", ha="center", va="center", color="black", fontsize=5)
                policy_row_items.append(policy_text)
                value_row_items.append(value_text)
                surge_row_items.append(surge_text)
            policy_text_overlays.append(policy_row_items)
            value_text_overlays.append(value_row_items)
            surge_text_overlays.append(surge_row_items)

    def update(frame_idx: int):
        quiver.set_UVC(display_u_frames[frame_idx], display_v_frames[frame_idx], distance_frames[frame_idx])
        quiver.set_clim(0.0, max_distance)
        value_image.set_data(value_frames[frame_idx])
        value_image.set_clim(value_vmin, value_vmax)
        surge_image.set_data(surge_frames[frame_idx])
        surge_image.set_clim(surge_vmin, surge_vmax)
        ax_policy.set_title(f"Policy | time_step={frame_idx}")
        ax_value.set_title(f"Value | time_step={frame_idx}")
        ax_surge.set_title(f"Surge | time_step={frame_idx}")
        fig.suptitle(f"{title_prefix} | time_step={frame_idx}", fontsize=14)
        if policy_text_overlays:
            for y in range(rows):
                for x in range(cols):
                    action_idx = int(round(policy_frames[frame_idx][y][x]))
                    action_x = action_idx % cols
                    action_y = action_idx // cols
                    policy_text_overlays[y][x].set_text(f"{action_x},{action_y}")
                    value_text_overlays[y][x].set_text(f"{value_frames[frame_idx][y][x]:.1f}")
                    surge_text_overlays[y][x].set_text(f"{surge_frames[frame_idx][y][x]:.2f}")
        artists = [quiver, value_image, surge_image]
        for row_items in policy_text_overlays:
            artists.extend(row_items)
        for row_items in value_text_overlays:
            artists.extend(row_items)
        for row_items in surge_text_overlays:
            artists.extend(row_items)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(policy_frames), interval=max(1, int(1000 / max(1, fps))), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
