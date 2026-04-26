from __future__ import annotations

import time
from typing import Optional


class ConsoleRenderer:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    def render(self, sim, event) -> None:
        if sim.config.field_dimension == 2:
            return self._render_2d(sim, event)
        surges = sim.get_surges()
        max_surge = max(surges)
        min_surge = min(surges)
        span = max(max_surge - min_surge, 1e-6)
        model_label = sim.get_model_label()
        cycle_label = sim.get_model_cycle_label()

        chars = []
        for idx, surge in enumerate(surges):
            level = int((surge - min_surge) / span * 8)
            symbol = " .:-=+*#%@"[min(level + 1, 9)]
            if idx == sim.driver_position:
                chars.append(f"[{symbol}]")
            else:
                chars.append(f" {symbol} ")

        print("Field:", "".join(chars))
        print(f"Simulation: {model_label} | Cycle: {cycle_label}")
        print(
            "Info:"
            f" pos={sim.driver_position}"
            f" balance=${sim.stats.balance:.2f}"
            f" trips={sim.stats.trips_completed}"
            f" time={sim.stats.time_steps}"
            f" empty={sim.stats.empty_distance}"
            f" paid={sim.stats.paid_distance}"
            f" updates={sim.stats.surge_updates}"
            f" offers={sim.stats.offers_seen}"
            f" skipped={sim.stats.offers_skipped}"
            f" waiting={sim.stats.waiting_ticks}"
        )
        if event and event.kind != "init":
            print(event.message)
        print("-" * 80)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def _render_2d(self, sim, event) -> None:
        surges = sim.get_surges()
        max_surge = max(surges)
        min_surge = min(surges)
        span = max(max_surge - min_surge, 1e-6)
        model_label = sim.get_model_label()
        cycle_label = sim.get_model_cycle_label()

        print("Field:")
        for y in range(sim.config.grid_height):
            row = []
            for x in range(sim.config.grid_width):
                idx = sim.coord_to_index(x, y)
                surge = surges[idx]
                level = int((surge - min_surge) / span * 8)
                symbol = " .:-=+*#%@"[min(level + 1, 9)]
                if idx == sim.driver_position:
                    row.append(f"[{symbol}]")
                else:
                    row.append(f" {symbol} ")
            print("".join(row))

        print(f"Simulation: {model_label} | Cycle: {cycle_label}")
        print(
            "Info:"
            f" pos={sim.format_position(sim.driver_position)}"
            f" balance=${sim.stats.balance:.2f}"
            f" trips={sim.stats.trips_completed}"
            f" time={sim.stats.time_steps}"
            f" empty={sim.stats.empty_distance}"
            f" paid={sim.stats.paid_distance}"
            f" updates={sim.stats.surge_updates}"
        )
        if event and event.kind != "init":
            print(event.message)
        print("-" * 80)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def close(self) -> None:
        return


class TkRenderer:
    """Simple graphical wrapper using Tkinter. Imported lazily to keep headless tests clean."""

    def __init__(self, cell_width: int = 70, height: int = 280, delay_ms: int = 400) -> None:
        import tkinter as tk

        self._tk = tk
        self.cell_width = cell_width
        self.height = height
        self.delay_ms = delay_ms
        self.window_width = 900
        self._closed = False
        self.root: Optional[tk.Tk] = tk.Tk()
        self.root.title("Taxi Simulation (1D Field)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.geometry(f"{self.window_width}x{self.height + 60}")
        self.root.minsize(self.window_width, self.height + 60)
        self.canvas = tk.Canvas(self.root, width=self.window_width, height=height, bg="#101418", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Starting...")
        self.status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 11),
            fg="#E8EEF2",
            bg="#101418",
            anchor="w",
            justify="left",
        )
        self.status.pack(fill="x")
        self._scene_cache: dict = {}

    def render(self, sim, event) -> None:
        if self.root is None or self._closed:
            return
        try:
            self._draw(sim, event)
            self.root.update_idletasks()
            self.root.update()
            if self.delay_ms > 0:
                self.root.after(self.delay_ms)
                self.root.update()
        except self._tk.TclError:
            self._closed = True
            self.root = None

    def _draw(self, sim, event) -> None:
        if sim.config.field_dimension == 2:
            self._draw_2d(sim, event)
            return
        surges = sim.get_surges()
        n = len(surges)
        self._ensure_scene(n)
        w = self._scene_cache["cell_w"]
        grid_x0 = 20
        grid_y0 = 70
        cell_h = max(70, self.height - 140)
        grid_y1 = grid_y0 + cell_h
        max_surge = max(surges) if surges else 1.0
        min_surge = min(surges) if surges else 0.0
        span = max(max_surge - min_surge, 1e-6)
        taxi_ids = self._scene_cache["taxi"]
        heat_rects = self._scene_cache["heat_rects"]
        idx_texts = self._scene_cache["idx_texts"]
        surge_texts = self._scene_cache["surge_texts"]
        legend_steps = self._scene_cache["legend_steps"]
        legend_rects = self._scene_cache["legend_rects"]
        legend_bounds = self._scene_cache["legend_bounds"]
        legend_low_text = self._scene_cache["legend_low_text"]
        legend_high_text = self._scene_cache["legend_high_text"]
        markers = self._scene_cache["markers"]
        title_text = self._scene_cache["title_text"]
        subtitle_text = self._scene_cache["subtitle_text"]

        self.canvas.itemconfigure(title_text, text=f"Taxi Simulation: {sim.get_model_label()}")
        self.canvas.itemconfigure(subtitle_text, text=f"Cycle: {sim.get_model_cycle_label()}")

        for idx, surge in enumerate(surges):
            x0 = grid_x0 + idx * w
            x1 = grid_x0 + (idx + 1) * w
            ratio = (surge - min_surge) / span
            y0 = grid_y0
            y1 = grid_y1
            color = self._surge_color(ratio)
            self.canvas.coords(heat_rects[idx], x0, y0, x1, y1)
            self.canvas.itemconfigure(heat_rects[idx], fill=color)
            self.canvas.coords(idx_texts[idx], (x0 + x1) / 2, grid_y1 + 15)
            self.canvas.itemconfigure(idx_texts[idx], text=str(idx))
            self.canvas.coords(surge_texts[idx], (x0 + x1) / 2, grid_y0 + 14)
            self.canvas.itemconfigure(surge_texts[idx], text=f"{surge:.2f}")

        taxi_idx = sim.driver_position
        x0 = grid_x0 + taxi_idx * w
        x1 = grid_x0 + (taxi_idx + 1) * w
        cx = (x0 + x1) / 2
        cy = (grid_y0 + grid_y1) / 2
        self.canvas.coords(taxi_ids["oval"], cx - 14, cy - 14, cx + 14, cy + 14)
        self.canvas.coords(taxi_ids["text"], cx, cy)
        self.canvas.tag_raise(taxi_ids["oval"])
        self.canvas.tag_raise(taxi_ids["text"])

        trip_ctx = getattr(sim, "trip_context", {}) or {}
        chosen_idx = trip_ctx.get("chosen_cell")
        passenger_idx = trip_ctx.get("passenger_cell")
        destination_idx = trip_ctx.get("destination_cell")
        offered_idx = trip_ctx.get("offered_pickup_cell")

        self._place_cell_outline(markers["chosen_outline"], chosen_idx, grid_x0, grid_y0, grid_y1, w)
        self._place_badge(markers["chosen_badge_oval"], markers["chosen_badge_text"], chosen_idx, "C", "#E2E8F0", grid_x0, grid_y0, w)
        self._place_badge(markers["offer_badge_oval"], markers["offer_badge_text"], offered_idx, "O", "#38BDF8", grid_x0, grid_y0, w, side="center")
        self._place_badge(markers["passenger_badge_oval"], markers["passenger_badge_text"], passenger_idx, "P", "#22C55E", grid_x0, grid_y0, w, side="left")
        self._place_badge(markers["dest_badge_oval"], markers["dest_badge_text"], destination_idx, "D", "#EF4444", grid_x0, grid_y0, w, side="right")
        for item_id in markers.values():
            self.canvas.tag_raise(item_id)
        self.canvas.tag_raise(taxi_ids["oval"])
        self.canvas.tag_raise(taxi_ids["text"])

        for i in range(legend_steps):
            self.canvas.itemconfigure(legend_rects[i], fill=self._surge_color(i / max(legend_steps - 1, 1)))
        legend_x0, legend_y0, legend_w, legend_h = legend_bounds
        self.canvas.coords(legend_low_text, legend_x0, legend_y0 + legend_h + 12)
        self.canvas.coords(legend_high_text, legend_x0 + legend_w, legend_y0 + legend_h + 12)
        self.canvas.itemconfigure(legend_low_text, text=f"low {min_surge:.2f}")
        self.canvas.itemconfigure(legend_high_text, text=f"high {max_surge:.2f}")

        self.status_var.set(
            f"Balance ${sim.stats.balance:.2f} | Trips {sim.stats.trips_completed} | "
            f"Time {sim.stats.time_steps} | Empty {sim.stats.empty_distance} | Paid {sim.stats.paid_distance} | "
            f"Surge updates {sim.stats.surge_updates} | Offers {sim.stats.offers_seen} | "
            f"Skipped {sim.stats.offers_skipped} | Waiting {sim.stats.waiting_ticks}"
            + (f" | {event.message}" if event and event.kind != "init" else "")
        )

    def _draw_2d(self, sim, event) -> None:
        surges = sim.get_surges()
        width = sim.config.grid_width
        height_cells = sim.config.grid_height
        self._ensure_scene_2d(width, height_cells)
        cell_w = self._scene_cache["cell_w"]
        cell_h = self._scene_cache["cell_h"]
        grid_x0 = 20
        grid_y0 = 70
        max_surge = max(surges) if surges else 1.0
        min_surge = min(surges) if surges else 0.0
        span = max(max_surge - min_surge, 1e-6)

        self.canvas.itemconfigure(self._scene_cache["title_text"], text=f"Taxi Simulation: {sim.get_model_label()}")
        self.canvas.itemconfigure(self._scene_cache["subtitle_text"], text=f"Cycle: {sim.get_model_cycle_label()}")

        for idx, rect_id in self._scene_cache["heat_rects"].items():
            x, y = sim.index_to_coord(idx)
            x0 = grid_x0 + x * cell_w
            y0 = grid_y0 + y * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            surge = surges[idx]
            ratio = (surge - min_surge) / span
            self.canvas.coords(rect_id, x0, y0, x1, y1)
            self.canvas.itemconfigure(rect_id, fill=self._surge_color(ratio))
            self.canvas.coords(self._scene_cache["surge_texts"][idx], (x0 + x1) / 2, y0 + 12)
            self.canvas.itemconfigure(self._scene_cache["surge_texts"][idx], text=f"{surge:.2f}")

        taxi_x, taxi_y = sim.index_to_coord(sim.driver_position)
        tx0 = grid_x0 + taxi_x * cell_w
        ty0 = grid_y0 + taxi_y * cell_h
        tx1 = tx0 + cell_w
        ty1 = ty0 + cell_h
        cx = (tx0 + tx1) / 2
        cy = (ty0 + ty1) / 2
        taxi_ids = self._scene_cache["taxi"]
        self.canvas.coords(taxi_ids["oval"], cx - 14, cy - 14, cx + 14, cy + 14)
        self.canvas.coords(taxi_ids["text"], cx, cy)

        trip_ctx = getattr(sim, "trip_context", {}) or {}
        for key, label, fill in (
            ("chosen_cell", "C", "#E2E8F0"),
            ("offered_pickup_cell", "O", "#38BDF8"),
            ("passenger_cell", "P", "#22C55E"),
            ("destination_cell", "D", "#EF4444"),
        ):
            self._place_badge_2d(self._scene_cache["markers"][key], trip_ctx.get(key), label, fill, sim, cell_w, cell_h)

        legend_x0, legend_y0, legend_w, legend_h = self._scene_cache["legend_bounds"]
        for i, rect_id in enumerate(self._scene_cache["legend_rects"]):
            self.canvas.itemconfigure(rect_id, fill=self._surge_color(i / max(self._scene_cache["legend_steps"] - 1, 1)))
        self.canvas.coords(self._scene_cache["legend_low_text"], legend_x0, legend_y0 + legend_h + 12)
        self.canvas.coords(self._scene_cache["legend_high_text"], legend_x0 + legend_w, legend_y0 + legend_h + 12)
        self.canvas.itemconfigure(self._scene_cache["legend_low_text"], text=f"low {min_surge:.2f}")
        self.canvas.itemconfigure(self._scene_cache["legend_high_text"], text=f"high {max_surge:.2f}")

        self.status_var.set(
            f"Balance ${sim.stats.balance:.2f} | Trips {sim.stats.trips_completed} | "
            f"Time {sim.stats.time_steps} | Empty {sim.stats.empty_distance} | Paid {sim.stats.paid_distance}"
            + (f" | {event.message}" if event and event.kind != "init" else "")
        )

    def _ensure_scene(self, n: int) -> None:
        cached_n = self._scene_cache.get("n")
        if cached_n == n:
            return

        self.canvas.delete("all")
        grid_x0 = 20
        grid_y0 = 70
        cell_h = max(70, self.height - 140)
        grid_y1 = grid_y0 + cell_h
        cell_w = max(32, min(self.cell_width, int((self.window_width - 40) / max(n, 1))))

        title_text = self.canvas.create_text(
            20, 20, anchor="w",
            fill="#F8FAFC", font=("Helvetica", 16, "bold"),
            text="Taxi Simulation"
        )
        subtitle_text = self.canvas.create_text(
            20,
            48,
            anchor="w",
            fill="#94A3B8",
            font=("Helvetica", 10),
            text="Cycle",
        )
        self.canvas.create_rectangle(
            grid_x0 - 2, grid_y0 - 2, grid_x0 + n * cell_w + 2, grid_y1 + 2, outline="#334155", width=2
        )

        heat_rects = []
        idx_texts = []
        surge_texts = []
        for idx in range(n):
            x0 = grid_x0 + idx * cell_w
            x1 = grid_x0 + (idx + 1) * cell_w
            heat_rects.append(
                self.canvas.create_rectangle(x0, grid_y0, x1, grid_y1, fill="#0F172A", outline="#0F172A")
            )
            idx_texts.append(
                self.canvas.create_text((x0 + x1) / 2, grid_y1 + 15, text=str(idx), fill="#CBD5E1", font=("Helvetica", 9))
            )
            surge_texts.append(
                self.canvas.create_text((x0 + x1) / 2, grid_y0 + 14, text="", fill="#F8FAFC", font=("Helvetica", 9, "bold"))
            )

        taxi_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#111827", outline="#F8FAFC", width=2)
        taxi_text = self.canvas.create_text(0, 0, text="CAR", fill="#F8FAFC", font=("Helvetica", 8, "bold"))
        chosen_outline = self.canvas.create_rectangle(0, 0, 0, 0, outline="#E2E8F0", width=3, state="hidden")
        chosen_badge_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#E2E8F0", outline="", state="hidden")
        chosen_badge_text = self.canvas.create_text(0, 0, text="C", fill="#0F172A", font=("Helvetica", 7, "bold"), state="hidden")
        offer_badge_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#38BDF8", outline="", state="hidden")
        offer_badge_text = self.canvas.create_text(0, 0, text="O", fill="#082F49", font=("Helvetica", 7, "bold"), state="hidden")
        passenger_badge_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#22C55E", outline="", state="hidden")
        passenger_badge_text = self.canvas.create_text(0, 0, text="P", fill="#06210F", font=("Helvetica", 7, "bold"), state="hidden")
        dest_badge_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#EF4444", outline="", state="hidden")
        dest_badge_text = self.canvas.create_text(0, 0, text="D", fill="#2A0B0B", font=("Helvetica", 7, "bold"), state="hidden")

        legend_x0 = grid_x0
        legend_y0 = grid_y1 + 32
        legend_w = min(260, n * cell_w)
        legend_h = 14
        legend_steps = 20
        legend_rects = []
        for i in range(legend_steps):
            lx0 = legend_x0 + i * legend_w / legend_steps
            lx1 = legend_x0 + (i + 1) * legend_w / legend_steps
            legend_rects.append(
                self.canvas.create_rectangle(lx0, legend_y0, lx1, legend_y0 + legend_h, fill="#000000", outline="")
            )
        self.canvas.create_rectangle(legend_x0, legend_y0, legend_x0 + legend_w, legend_y0 + legend_h, outline="#334155")
        legend_low_text = self.canvas.create_text(
            legend_x0,
            legend_y0 + legend_h + 12,
            anchor="w",
            text="",
            fill="#94A3B8",
            font=("Helvetica", 9),
        )
        legend_high_text = self.canvas.create_text(
            legend_x0 + legend_w,
            legend_y0 + legend_h + 12,
            anchor="e",
            text="",
            fill="#94A3B8",
            font=("Helvetica", 9),
        )

        self._scene_cache = {
            "n": n,
            "cell_w": cell_w,
            "heat_rects": heat_rects,
            "idx_texts": idx_texts,
            "surge_texts": surge_texts,
            "taxi": {"oval": taxi_oval, "text": taxi_text},
            "title_text": title_text,
            "subtitle_text": subtitle_text,
            "markers": {
                "chosen_outline": chosen_outline,
                "chosen_badge_oval": chosen_badge_oval,
                "chosen_badge_text": chosen_badge_text,
                "offer_badge_oval": offer_badge_oval,
                "offer_badge_text": offer_badge_text,
                "passenger_badge_oval": passenger_badge_oval,
                "passenger_badge_text": passenger_badge_text,
                "dest_badge_oval": dest_badge_oval,
                "dest_badge_text": dest_badge_text,
            },
            "legend_steps": legend_steps,
            "legend_rects": legend_rects,
            "legend_bounds": (legend_x0, legend_y0, legend_w, legend_h),
            "legend_low_text": legend_low_text,
            "legend_high_text": legend_high_text,
        }

    def _ensure_scene_2d(self, width: int, height_cells: int) -> None:
        cached_shape = self._scene_cache.get("shape")
        if cached_shape == (width, height_cells):
            return

        self.canvas.delete("all")
        grid_x0 = 20
        grid_y0 = 70
        available_w = self.window_width - 60
        available_h = self.height - 140
        cell_w = max(28, min(self.cell_width, int(available_w / max(width, 1))))
        cell_h = max(28, min(self.cell_width, int(available_h / max(height_cells, 1))))
        grid_x1 = grid_x0 + width * cell_w
        grid_y1 = grid_y0 + height_cells * cell_h

        title_text = self.canvas.create_text(20, 20, anchor="w", fill="#F8FAFC", font=("Helvetica", 16, "bold"), text="Taxi Simulation")
        subtitle_text = self.canvas.create_text(20, 48, anchor="w", fill="#94A3B8", font=("Helvetica", 10), text="Cycle")
        self.canvas.create_rectangle(grid_x0 - 2, grid_y0 - 2, grid_x1 + 2, grid_y1 + 2, outline="#334155", width=2)

        heat_rects = {}
        surge_texts = {}
        for y in range(height_cells):
            for x in range(width):
                idx = y * width + x
                x0 = grid_x0 + x * cell_w
                y0 = grid_y0 + y * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                heat_rects[idx] = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#0F172A", outline="#0F172A")
                surge_texts[idx] = self.canvas.create_text((x0 + x1) / 2, y0 + 12, text="", fill="#F8FAFC", font=("Helvetica", 8, "bold"))
                self.canvas.create_text((x0 + x1) / 2, y1 - 10, text=f"{x},{y}", fill="#CBD5E1", font=("Helvetica", 7))

        taxi_oval = self.canvas.create_oval(0, 0, 0, 0, fill="#111827", outline="#F8FAFC", width=2)
        taxi_text = self.canvas.create_text(0, 0, text="CAR", fill="#F8FAFC", font=("Helvetica", 8, "bold"))

        markers = {}
        for key in ("chosen_cell", "offered_pickup_cell", "passenger_cell", "destination_cell"):
            markers[key] = {
                "oval": self.canvas.create_oval(0, 0, 0, 0, state="hidden", outline=""),
                "text": self.canvas.create_text(0, 0, state="hidden", font=("Helvetica", 7, "bold")),
            }

        legend_x0 = grid_x0
        legend_y0 = grid_y1 + 24
        legend_w = min(260, width * cell_w)
        legend_h = 14
        legend_steps = 20
        legend_rects = []
        for i in range(legend_steps):
            lx0 = legend_x0 + i * legend_w / legend_steps
            lx1 = legend_x0 + (i + 1) * legend_w / legend_steps
            legend_rects.append(self.canvas.create_rectangle(lx0, legend_y0, lx1, legend_y0 + legend_h, fill="#000000", outline=""))
        self.canvas.create_rectangle(legend_x0, legend_y0, legend_x0 + legend_w, legend_y0 + legend_h, outline="#334155")
        legend_low_text = self.canvas.create_text(legend_x0, legend_y0 + legend_h + 12, anchor="w", text="", fill="#94A3B8", font=("Helvetica", 9))
        legend_high_text = self.canvas.create_text(legend_x0 + legend_w, legend_y0 + legend_h + 12, anchor="e", text="", fill="#94A3B8", font=("Helvetica", 9))

        self._scene_cache = {
            "shape": (width, height_cells),
            "cell_w": cell_w,
            "cell_h": cell_h,
            "heat_rects": heat_rects,
            "surge_texts": surge_texts,
            "taxi": {"oval": taxi_oval, "text": taxi_text},
            "markers": markers,
            "title_text": title_text,
            "subtitle_text": subtitle_text,
            "legend_steps": legend_steps,
            "legend_rects": legend_rects,
            "legend_bounds": (legend_x0, legend_y0, legend_w, legend_h),
            "legend_low_text": legend_low_text,
            "legend_high_text": legend_high_text,
        }

    def _place_badge_2d(self, marker: dict, idx, label: str, fill: str, sim, cell_w: int, cell_h: int) -> None:
        if idx is None:
            self.canvas.itemconfigure(marker["oval"], state="hidden")
            self.canvas.itemconfigure(marker["text"], state="hidden")
            return
        grid_x0 = 20
        grid_y0 = 70
        x, y = sim.index_to_coord(idx)
        cx = grid_x0 + x * cell_w + cell_w - 12
        cy = grid_y0 + y * cell_h + 14
        self.canvas.coords(marker["oval"], cx - 9, cy - 9, cx + 9, cy + 9)
        self.canvas.coords(marker["text"], cx, cy)
        self.canvas.itemconfigure(marker["oval"], fill=fill, state="normal")
        self.canvas.itemconfigure(marker["text"], text=label, fill="#0F172A", state="normal")

    def _place_cell_outline(self, item_id: int, idx, grid_x0: int, grid_y0: int, grid_y1: int, w: int) -> None:
        if idx is None:
            self.canvas.itemconfigure(item_id, state="hidden")
            return
        x0 = grid_x0 + idx * w + 2
        x1 = grid_x0 + (idx + 1) * w - 2
        self.canvas.coords(item_id, x0, grid_y0 + 2, x1, grid_y1 - 2)
        self.canvas.itemconfigure(item_id, state="normal")

    def _place_badge(
        self,
        oval_id: int,
        text_id: int,
        idx,
        label: str,
        fill: str,
        grid_x0: int,
        grid_y0: int,
        w: int,
        side: str = "center",
    ) -> None:
        if idx is None:
            self.canvas.itemconfigure(oval_id, state="hidden")
            self.canvas.itemconfigure(text_id, state="hidden")
            return

        x0 = grid_x0 + idx * w
        x1 = grid_x0 + (idx + 1) * w
        if side == "left":
            cx = x0 + 12
        elif side == "right":
            cx = x1 - 12
        else:
            cx = (x0 + x1) / 2
        cy = grid_y0 + 28
        self.canvas.coords(oval_id, cx - 9, cy - 9, cx + 9, cy + 9)
        self.canvas.coords(text_id, cx, cy)
        self.canvas.itemconfigure(oval_id, fill=fill, state="normal")
        self.canvas.itemconfigure(text_id, text=label, state="normal")

    def _surge_color(self, ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        # Low surge = green, mid = yellow, high = red.
        if ratio < 0.5:
            t = ratio / 0.5
            r = int(34 + (250 - 34) * t)
            g = int(197 + (204 - 197) * t)
            b = int(94 + (21 - 94) * t)
        else:
            t = (ratio - 0.5) / 0.5
            r = int(250 + (239 - 250) * t)
            g = int(204 + (68 - 204) * t)
            b = int(21 + (68 - 21) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def close(self) -> None:
        # Keep window open after simulation, but allow normal close on Ubuntu WM.
        while not self._closed and self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
                time.sleep(0.03)
            except self._tk.TclError:
                self._closed = True
                self.root = None

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _on_close(self) -> None:
        self._closed = True
        if self.root is not None:
            try:
                self.root.destroy()
            except self._tk.TclError:
                pass
            self.root = None
