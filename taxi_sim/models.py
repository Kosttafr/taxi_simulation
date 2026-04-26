from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class CellState:
    supply: float
    demand: float

    @property
    def surge(self) -> float:
        # User-requested definition: surge = supply / demand.
        safe_demand = self.demand if self.demand > 1e-6 else 1e-6
        return self.supply / safe_demand


class RandomizationModel:
    name = "base"

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        raise NotImplementedError

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        raise NotImplementedError

    def describe(self, tick: int) -> str:
        return self.name


class AnalyticWaveModel(RandomizationModel):
    name = "analytic_wave"

    def __init__(self, dt: float = 0.05, grid_width: int = 10, grid_height: int = 1) -> None:
        self.dt = max(1e-6, dt)
        self.grid_width = max(1, grid_width)
        self.grid_height = max(1, grid_height)

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        cells = [CellState(supply=1.0, demand=1.0) for _ in range(n_cells)]
        self.update(rng, cells, tick=0)
        return cells

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        if not cells:
            return
        time_value = tick * self.dt
        for idx, cell in enumerate(cells):
            x, y = self._coords(idx)
            x_phase = 0.0 if self.grid_width == 1 else x / (self.grid_width - 1)
            y_phase = 1.0 if self.grid_height == 1 else y / (self.grid_height - 1)
            surge = math.cos(math.pi * x_phase * y_phase + math.pi * time_value) + 2.0
            cell.supply = max(0.0, surge)
            cell.demand = 1.0

    def describe(self, tick: int) -> str:
        time_value = tick * self.dt
        return f"{self.name}: S(x,y,t)=cos(pi*x*y+pi*t)+2 | t={time_value:.2f}"

    def _coords(self, idx: int) -> tuple[int, int]:
        return idx % self.grid_width, idx // self.grid_width


class UniformResetModel(RandomizationModel):
    name = "uniform_reset"

    def __init__(
        self,
        supply_range: tuple[float, float] = (0.5, 3.0),
        demand_range: tuple[float, float] = (0.5, 3.0),
    ) -> None:
        self.supply_range = supply_range
        self.demand_range = demand_range

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        return [
            CellState(
                supply=rng.uniform(*self.supply_range),
                demand=rng.uniform(*self.demand_range),
            )
            for _ in range(n_cells)
        ]

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        for cell in cells:
            cell.supply = rng.uniform(*self.supply_range)
            cell.demand = rng.uniform(*self.demand_range)


class RandomWalkModel(RandomizationModel):
    name = "random_walk"

    def __init__(
        self,
        step_size: float = 0.6,
        min_value: float = 0.2,
        max_value: float = 5.0,
    ) -> None:
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        return [
            CellState(
                supply=rng.uniform(1.0, 2.5),
                demand=rng.uniform(1.0, 2.5),
            )
            for _ in range(n_cells)
        ]

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        for cell in cells:
            cell.supply = min(
                self.max_value,
                max(self.min_value, cell.supply + rng.uniform(-self.step_size, self.step_size)),
            )
            cell.demand = min(
                self.max_value,
                max(self.min_value, cell.demand + rng.uniform(-self.step_size, self.step_size)),
            )


class HotspotShiftModel(RandomizationModel):
    name = "hotspot_shift"

    def __init__(self, base: float = 1.0, hotspot_bonus: float = 3.0) -> None:
        self.base = base
        self.hotspot_bonus = hotspot_bonus
        self._hotspot_idx = 0

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        self._hotspot_idx = rng.randrange(n_cells)
        cells = [CellState(supply=self.base, demand=self.base) for _ in range(n_cells)]
        self._apply_hotspot(rng, cells)
        return cells

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        self._hotspot_idx = rng.randrange(len(cells))
        for cell in cells:
            cell.supply = self.base + rng.uniform(-0.2, 0.2)
            cell.demand = self.base + rng.uniform(-0.2, 0.2)
        self._apply_hotspot(rng, cells)

    def _apply_hotspot(self, rng: random.Random, cells: list[CellState]) -> None:
        hotspot = cells[self._hotspot_idx]
        # Create a strong surge in the selected cell (high supply / low demand as requested formula).
        hotspot.supply = self.base + self.hotspot_bonus + rng.uniform(0.0, 1.0)
        hotspot.demand = max(0.2, self.base * 0.5 + rng.uniform(-0.1, 0.1))


class CommuteCycleModel(RandomizationModel):
    """
    Simulate a daily city rhythm:
    morning -> high surge on outer cells
    evening -> high surge near the center
    """

    name = "commute_cycle"

    def __init__(
        self,
        base: float = 1.6,
        amplitude: float = 2.4,
        period_ticks: int = 60,
        noise: float = 0.18,
        mode: str = "cycle",
    ) -> None:
        self.base = base
        self.amplitude = amplitude
        self.period_ticks = max(8, period_ticks)
        self.noise = noise
        self.mode = mode
        self.period_order = ["morning", "day", "evening", "night"]
        if self.mode not in {"cycle", *self.period_order}:
            raise ValueError(f"Unknown commute mode: {self.mode}")

    def initialize(self, rng: random.Random, n_cells: int) -> list[CellState]:
        cells = [CellState(supply=self.base, demand=self.base) for _ in range(n_cells)]
        self.update(rng, cells, tick=0)
        return cells

    def update(self, rng: random.Random, cells: list[CellState], tick: int) -> None:
        if len(cells) == 1:
            cells[0].supply = self.base + self.amplitude
            cells[0].demand = self.base
            return

        center = (len(cells) - 1) / 2

        if self.mode == "cycle":
            total_ticks = self.period_ticks * len(self.period_order)
            day_phase = tick % total_ticks
            period_idx = int(day_phase // self.period_ticks)
            period_progress = (day_phase % self.period_ticks) / self.period_ticks
            # Hold the current phase for 80% of its duration, then blend into the next one.
            if period_progress < 0.8:
                blend = 0.0
            else:
                blend = (period_progress - 0.8) / 0.2
            current_mode = self.period_order[period_idx]
            next_mode = self.period_order[(period_idx + 1) % len(self.period_order)]
        else:
            current_mode = self.mode
            next_mode = self.mode
            blend = 0.0

        for idx, cell in enumerate(cells):
            outerness = abs(idx - center) / center if center > 0 else 0.0
            centerness = 1.0 - outerness
            day_mix = 0.58 - 0.08 * outerness
            nightlife = 0.12 + 0.04 * centerness

            hotspotness = self._blend_profiles(
                current_mode=current_mode,
                next_mode=next_mode,
                blend=blend,
                outerness=outerness,
                centerness=centerness,
                daytime_band=day_mix,
                nightlife=nightlife,
            )
            cell.supply = max(
                0.2,
                self.base + self.amplitude * hotspotness + rng.uniform(-self.noise, self.noise),
            )
            cell.demand = max(
                0.2,
                self.base + self.amplitude * (1.0 - hotspotness) + rng.uniform(-self.noise, self.noise),
            )

    def _blend_profiles(
        self,
        *,
        current_mode: str,
        next_mode: str,
        blend: float,
        outerness: float,
        centerness: float,
        daytime_band: float,
        nightlife: float,
    ) -> float:
        current_value = self._profile_value(current_mode, outerness, centerness, daytime_band, nightlife)
        next_value = self._profile_value(next_mode, outerness, centerness, daytime_band, nightlife)
        return current_value * (1.0 - blend) + next_value * blend

    def _profile_value(
        self,
        mode: str,
        outerness: float,
        centerness: float,
        daytime_band: float,
        nightlife: float,
    ) -> float:
        if mode == "morning":
            return outerness
        if mode == "day":
            return daytime_band
        if mode == "evening":
            return centerness
        if mode == "night":
            return nightlife
        raise ValueError(f"Unknown commute mode: {mode}")

    def describe(self, tick: int) -> str:
        if self.mode != "cycle":
            return f"{self.name}:{self.mode}"

        total_ticks = self.period_ticks * len(self.period_order)
        day_phase = tick % total_ticks
        period_idx = int(day_phase // self.period_ticks)
        period_progress = (day_phase % self.period_ticks) / self.period_ticks
        current_mode = self.period_order[period_idx]
        next_mode = self.period_order[(period_idx + 1) % len(self.period_order)]
        if period_progress < 0.8:
            stable_progress = period_progress / 0.8
            return f"{self.name}:{current_mode} | phase {stable_progress:.0%}"

        blend = (period_progress - 0.8) / 0.2
        return f"{self.name}:{current_mode}->{next_mode} | transition {blend:.0%}"
