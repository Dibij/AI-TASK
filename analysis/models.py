"""
analysis/models.py — Typed dataclasses for experiment data.

These are the canonical in-memory representations used throughout the
analysis pipeline.  Both the new-format JSON loader and the legacy CSV
adapter produce these structures, so all downstream code (metrics, plots,
reports) only ever works with these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Robot constants — single source of truth for the analysis layer
# ---------------------------------------------------------------------------

#: Actual robot names used in the coursework (order matches robot IDs 0-3).
ROBOT_NAMES: list[str] = ["Blue", "Red", "Green", "Yellow"]

#: Hex colour for each robot — used consistently in every plot.
ROBOT_COLOURS: dict[str, str] = {
    "Blue":   "#2980B9",
    "Red":    "#E74C3C",
    "Green":  "#27AE60",
    "Yellow": "#F1C40F",
}

#: Human-readable names for each experiment configuration.
CONFIG_LABELS: dict[str, str] = {
    "A": "Shortest-Path Baseline",
    "B": "Risk-Aware A*",
    "C": "Conservative Risk-Aware",
    "D": "Risk-Aware (No Collision Avoidance)",
    "E": "Risk-Aware (Full, Collision Enabled)",
}

# Composite score weights (overall scorecard).
# Higher is better for success_rate and efficiency;
# lower is better for all death / overhead metrics.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "success_rate":    0.35,
    "efficiency":      0.30,
    "mine_death_rate": 0.10,   # inverted
    "collision_rate":  0.10,   # inverted
    "timeout_rate":    0.05,   # inverted
    "avg_replans":     0.05,   # inverted, normalised by max
    "avg_waits":       0.05,   # inverted, normalised by max
}


# ---------------------------------------------------------------------------
# Per-robot, per-run result
# ---------------------------------------------------------------------------

@dataclass
class RobotResult:
    """Outcome data for a single robot in a single simulation run."""

    name: str
    reached: bool
    hit_mine: bool
    hit_robot: bool
    timed_out: bool
    steps: int
    oracle_steps: int           # shortest-path oracle length
    replans: int
    waits: int
    unknown_cells_entered: int

    @property
    def efficiency(self) -> Optional[float]:
        """Path efficiency = oracle_steps / actual_steps.

        Returns None if the robot never moved (steps == 0) to avoid
        division-by-zero masking real failures.
        """
        if self.steps <= 0:
            return None
        return self.oracle_steps / self.steps

    @property
    def excess_steps(self) -> int:
        """Extra steps taken beyond the oracle shortest path."""
        return max(0, self.steps - self.oracle_steps)

    def to_dict(self) -> dict:
        return {
            "name":                  self.name,
            "reached":               self.reached,
            "hit_mine":              self.hit_mine,
            "hit_robot":             self.hit_robot,
            "timed_out":             self.timed_out,
            "steps":                 self.steps,
            "oracle_steps":          self.oracle_steps,
            "efficiency":            self.efficiency,
            "excess_steps":          self.excess_steps,
            "replans":               self.replans,
            "waits":                 self.waits,
            "unknown_cells_entered": self.unknown_cells_entered,
        }


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Complete outcome data for one simulation run (one round)."""

    run_id: int
    config: str                     # planner config key, e.g. "A", "B"
    seed: int
    ticks: int
    world_mines: int
    world_walls: int
    robots: list[RobotResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.robots if r.reached)

    @property
    def success_rate(self) -> float:
        if not self.robots:
            return 0.0
        return self.success_count / len(self.robots)

    @property
    def mean_efficiency(self) -> Optional[float]:
        effs = [r.efficiency for r in self.robots if r.efficiency is not None]
        return sum(effs) / len(effs) if effs else None

    def to_dict(self) -> dict:
        return {
            "run_id":      self.run_id,
            "config":      self.config,
            "seed":        self.seed,
            "ticks":       self.ticks,
            "world":       {"mines": self.world_mines, "walls": self.world_walls},
            "robots":      [r.to_dict() for r in self.robots],
        }


# ---------------------------------------------------------------------------
# Experiment configuration snapshot
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Configuration snapshot stored in config.json."""

    schema_version: str
    timestamp: str
    experiment_name: str
    experiment_type: str          # "comparison" | "sweep"
    base_seed: int
    num_seeds: int
    planner_configs: dict         # key → {name, description, planner, ...}
    robot_names: list[str]
    grid: dict                    # {rows, cols, mine_density, wall_density}
    sim: dict                     # {max_ticks, conflict_policy}
    migrated_from_legacy: bool = False

    def to_dict(self) -> dict:
        return {
            "schema_version":       self.schema_version,
            "timestamp":            self.timestamp,
            "experiment_name":      self.experiment_name,
            "experiment_type":      self.experiment_type,
            "base_seed":            self.base_seed,
            "num_seeds":            self.num_seeds,
            "planner_configs":      self.planner_configs,
            "robot_names":          self.robot_names,
            "grid":                 self.grid,
            "sim":                  self.sim,
            "migrated_from_legacy": self.migrated_from_legacy,
        }
