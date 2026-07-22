"""
analysis/metrics.py — Statistical computation for experiment data.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd

from analysis.models import RunResult, COMPOSITE_WEIGHTS


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0,
            "p25": 0.0, "p75": 0.0, "p95": 0.0,
            "ci95": 0.0, "n": 0.0
        }
    
    n = len(values)
    mean = sum(values) / n
    sorted_values = sorted(values)
    
    mid = n // 2
    median = (sorted_values[mid] + sorted_values[~mid]) / 2.0 if n % 2 == 0 else sorted_values[mid]
    
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
        
    ci95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    
    if n > 1:
        try:
            p25 = statistics.quantiles(sorted_values, n=4)[0]
            p75 = statistics.quantiles(sorted_values, n=4)[2]
            p95 = statistics.quantiles(sorted_values, n=20)[18]
        except Exception:
            p25 = sorted_values[max(0, int(n * 0.25))]
            p75 = sorted_values[min(n - 1, int(n * 0.75))]
            p95 = sorted_values[min(n - 1, int(n * 0.95))]
    else:
        p25 = median
        p75 = median
        p95 = median
        
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "ci95": ci95,
        "n": float(n),
    }


def runs_to_dataframe(runs: list[RunResult]) -> pd.DataFrame:
    """Convert a list of RunResult objects into a long-format DataFrame."""
    records = [
        {
            "run_id": run.run_id,
            "config": run.config,
            "seed": run.seed,
            "ticks": run.ticks,
            "world_mines": run.world_mines,
            "world_walls": run.world_walls,
            "robot_name": robot.name,
            "reached": robot.reached,
            "hit_mine": robot.hit_mine,
            "hit_robot": robot.hit_robot,
            "timed_out": robot.timed_out,
            "steps": robot.steps,
            "oracle_steps": robot.oracle_steps,
            "efficiency": robot.efficiency if robot.efficiency is not None else float("nan"),
            "excess_steps": robot.excess_steps,
            "replans": robot.replans,
            "waits": robot.waits,
            "unknown_cells_entered": robot.unknown_cells_entered,
        }
        for run in runs
        for robot in run.robots
    ]
    return pd.DataFrame(records)


def _extract_metrics(runs: list[RunResult], predicate) -> dict[str, list[float]]:
    """Helper to extract lists of metrics for a subset of robots."""
    successes = []
    efficiencies = []
    mine_deaths = []
    collision_deaths = []
    timeouts = []
    replans = []
    waits = []
    excess_steps = []
    unknown_cells = []
    ticks_list = []
    
    for run in runs:
        run_matches = False
        for robot in run.robots:
            if predicate(run, robot):
                run_matches = True
                successes.append(1.0 if robot.reached else 0.0)
                if robot.efficiency is not None:
                    efficiencies.append(robot.efficiency)
                mine_deaths.append(1.0 if robot.hit_mine else 0.0)
                collision_deaths.append(1.0 if robot.hit_robot else 0.0)
                timeouts.append(1.0 if robot.timed_out else 0.0)
                replans.append(float(robot.replans))
                waits.append(float(robot.waits))
                excess_steps.append(float(robot.excess_steps))
                unknown_cells.append(float(robot.unknown_cells_entered))
        if run_matches:
            ticks_list.append(float(run.ticks))
            
    return {
        "success_rate": successes,
        "efficiency": efficiencies,
        "mine_death_rate": mine_deaths,
        "collision_rate": collision_deaths,
        "timeout_rate": timeouts,
        "replans": replans,
        "waits": waits,
        "excess_steps": excess_steps,
        "unknown_cells_entered": unknown_cells,
        "ticks": ticks_list,
    }


def _aggregate_from_metrics(metrics: dict[str, list[float]]) -> dict[str, dict]:
    """Helper to compute stats for all metric lists."""
    return {k: compute_stats(v) for k, v in metrics.items()}


def aggregate_overall(runs: list[RunResult]) -> dict[str, dict]:
    """Compute overall statistics across all runs and robots."""
    metrics = _extract_metrics(runs, lambda run, robot: True)
    return _aggregate_from_metrics(metrics)


def aggregate_by_config(runs: list[RunResult]) -> dict[str, dict]:
    """Compute statistics grouped by config key."""
    configs = sorted({r.config for r in runs})
    result = {}
    for cfg in configs:
        # Use default argument (cfg=cfg) to avoid late-binding closure bug
        metrics = _extract_metrics(runs, lambda run, robot, cfg=cfg: run.config == cfg)
        result[cfg] = _aggregate_from_metrics(metrics)
    return result


def aggregate_by_robot(runs: list[RunResult]) -> dict[str, dict]:
    """Compute statistics grouped by robot name."""
    from analysis.models import ROBOT_NAMES
    robot_names = sorted(
        {robot.name for run in runs for robot in run.robots},
        key=lambda n: ROBOT_NAMES.index(n) if n in ROBOT_NAMES else 999,
    )
    result = {}
    for name in robot_names:
        # Use default argument (name=name) to avoid late-binding closure bug
        metrics = _extract_metrics(runs, lambda run, robot, name=name: robot.name == name)
        result[name] = _aggregate_from_metrics(metrics)
    return result


def build_summary_dict(runs: list[RunResult], meta: Any) -> dict:
    """Compose the full summary.json structure."""
    return {
        "schema_version": "2.0",
        "num_runs": len(runs),
        "overall": aggregate_overall(runs),
        "by_config": aggregate_by_config(runs),
        "by_robot": aggregate_by_robot(runs),
    }


def config_scorecard(runs: list[RunResult]) -> pd.DataFrame:
    """Compute the overall composite score for each configuration."""
    df = runs_to_dataframe(runs)
    if df.empty:
        return pd.DataFrame()
        
    scorecard = df.groupby("config").agg(
        success_rate=("reached", "mean"),
        efficiency=("efficiency", "mean"),
        mine_death_rate=("hit_mine", "mean"),
        collision_rate=("hit_robot", "mean"),
        timeout_rate=("timed_out", "mean"),
        avg_replans=("replans", "mean"),
        avg_waits=("waits", "mean"),
        avg_unknown_cells=("unknown_cells_entered", "mean"),
        avg_excess_steps=("excess_steps", "mean"),
    ).reset_index()

    epsilon = 1e-9
    
    def min_max_norm(series: pd.Series) -> pd.Series:
        mn = series.min()
        mx = series.max()
        return (series - mn) / (mx - mn + epsilon)

    scorecard["overall_score"] = (
        COMPOSITE_WEIGHTS.get("success_rate", 0.0) * scorecard["success_rate"]
        + COMPOSITE_WEIGHTS.get("efficiency", 0.0) * scorecard["efficiency"].fillna(0.0)
        + COMPOSITE_WEIGHTS.get("mine_death_rate", 0.0) * (1.0 - scorecard["mine_death_rate"])
        + COMPOSITE_WEIGHTS.get("collision_rate", 0.0) * (1.0 - scorecard["collision_rate"])
        + COMPOSITE_WEIGHTS.get("timeout_rate", 0.0) * (1.0 - scorecard["timeout_rate"])
        + COMPOSITE_WEIGHTS.get("avg_replans", 0.0) * (1.0 - min_max_norm(scorecard["avg_replans"]))
        + COMPOSITE_WEIGHTS.get("avg_waits", 0.0) * (1.0 - min_max_norm(scorecard["avg_waits"]))
    )
    
    return scorecard.sort_values("overall_score", ascending=False)
