#!/usr/bin/env python3
"""
AI GRID-WORLD TOURNAMENT / EXPERIMENT ANALYZER
==============================================

Reads:
  - tournaments_master.logs
  - simulation_data_*.csv
  - experiments_summary_master.csv
  - experiments_meta_master.csv
  - experiments_master.csv

and generates a large analytical report containing:
  - tournament performance
  - survival/death analysis
  - mine vs collision analysis
  - efficiency distributions
  - wait/replan analysis
  - speed/risk/patience relationships
  - configuration comparisons
  - oracle efficiency
  - unknown-cell exploration
  - timeout analysis
  - round-level outcomes
  - correlation heatmaps
  - confidence interval plots
  - seed stability analysis
  - CSV summary tables
  - an HTML report index

Usage:
    python analyze_gridworld.py --logs-dir "D:/Code/AI-task/logs"

Optional:
    python analyze_gridworld.py --logs-dir "..." --output-dir "analysis_report"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_OUTPUT = "gridworld_analysis_report"

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ---------------------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------------------

def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_list(value: Any) -> list:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def parse_dict(value: Any) -> dict:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_bar_labels(ax, decimals=2):
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=f"%.{decimals}f", padding=3, fontsize=8)
        except Exception:
            pass


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def title(name: str) -> str:
    return re.sub(r"[_\-]+", " ", str(name)).title()


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def discover_files(log_dir: Path) -> dict[str, Path | None]:
    files = {}

    candidates = {
        "tournament": [
            "tournaments_master.logs",
            "tournaments_master.csv",
            "tournament_master.logs",
        ],
        "summary": [
            "experiments_summary_master.csv",
        ],
        "meta": [
            "experiments_meta_master.csv",
        ],
        "experiments": [
            "experiments_master.csv",
        ],
    }

    for key, names in candidates.items():
        files[key] = None
        for name in names:
            candidate = log_dir / name
            if candidate.exists():
                files[key] = candidate
                break

    simulation_files = sorted(log_dir.glob("simulation_data_*.csv"))
    files["simulation"] = simulation_files[0] if simulation_files else None

    return files


def load_data(files: dict) -> dict[str, pd.DataFrame]:
    data = {}

    for key, path in files.items():
        if path is None:
            continue

        try:
            data[key] = pd.read_csv(path)
            print(f"[OK] Loaded {path.name}: {data[key].shape}")
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc}")

    return data


# ---------------------------------------------------------------------
# TOURNAMENT DATA NORMALIZATION
# ---------------------------------------------------------------------

def normalize_tournament(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "avg_efficiency",
        "avg_wait_per_round",
        "collision_death_rate",
        "completion_rate",
        "meta_max_ticks",
        "meta_num_robots",
        "meta_rounds",
        "meta_seed",
        "mine_death_rate",
        "patience",
        "risk_tolerance",
        "speed",
        "survival_rate",
        "timeout_rate",
        "total_collision_deaths",
        "total_mine_deaths",
        "total_timeouts",
        "total_wins",
    ]

    df = safe_numeric(df, numeric_cols)

    if "meta_timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["meta_timestamp"], errors="coerce")

    if "total_wins" in df and "meta_rounds" in df:
        df["win_pct"] = df["total_wins"] / df["meta_rounds"].replace(0, np.nan)

    if "total_mine_deaths" in df and "meta_rounds" in df:
        df["mine_death_pct"] = df["total_mine_deaths"] / df["meta_rounds"].replace(0, np.nan)

    if "total_collision_deaths" in df and "meta_rounds" in df:
        df["collision_death_pct"] = (
            df["total_collision_deaths"] / df["meta_rounds"].replace(0, np.nan)
        )

    return df


# ---------------------------------------------------------------------
# EXPERIMENT DATA NORMALIZATION
# ---------------------------------------------------------------------

def normalize_experiments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for i in range(4):
        prefix = f"summary_{i}_"

        if f"{prefix}name" not in df:
            continue

        df[f"robot_{i}_efficiency"] = np.nan

        steps = pd.to_numeric(df.get(f"{prefix}steps"), errors="coerce")
        oracle = pd.to_numeric(df.get(f"{prefix}shortest_oracle"), errors="coerce")

        df[f"robot_{i}_efficiency"] = oracle / steps.replace(0, np.nan)

        reached = df.get(f"{prefix}reached")
        if reached is not None:
            df[f"robot_{i}_success"] = reached.astype(bool)

    return df


def melt_experiments(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        config = row.get("config")
        round_idx = row.get("round_idx")
        seed = row.get("round_seed")

        for i in range(4):
            prefix = f"summary_{i}_"

            name_col = f"{prefix}name"
            if name_col not in df.columns:
                continue

            records.append({
                "config": config,
                "round_idx": round_idx,
                "round_seed": seed,
                "robot_index": i,
                "robot": row.get(name_col),
                "reached": row.get(f"{prefix}reached"),
                "hit_mine": row.get(f"{prefix}hit_mine"),
                "hit_robot": row.get(f"{prefix}hit_robot"),
                "replans": row.get(f"{prefix}replans"),
                "oracle_steps": row.get(f"{prefix}shortest_oracle"),
                "steps": row.get(f"{prefix}steps"),
                "timed_out": row.get(f"{prefix}timed_out"),
                "unknown_cells": row.get(f"{prefix}unknown_cells_entered"),
                "waits": row.get(f"{prefix}waits"),
            })

    out = pd.DataFrame(records)

    numeric_cols = [
        "replans",
        "oracle_steps",
        "steps",
        "unknown_cells",
        "waits",
    ]
    out = safe_numeric(out, numeric_cols)

    if not out.empty:
        out["efficiency"] = out["oracle_steps"] / out["steps"].replace(0, np.nan)
        out["excess_steps"] = out["steps"] - out["oracle_steps"]

    return out


# ---------------------------------------------------------------------
# PLOT HELPERS
# ---------------------------------------------------------------------

def plot_bar(series, path, plot_title, ylabel="", rotation=0):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(plot_title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    add_bar_labels(ax)
    savefig(fig, path)


def plot_hist(series, path, plot_title, xlabel):
    fig, ax = plt.subplots()
    series.dropna().plot(kind="hist", bins=20, ax=ax)
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel)
    savefig(fig, path)


def plot_scatter(df, x, y, path, plot_title, xlabel=None, ylabel=None):
    if x not in df or y not in df:
        return

    subset = df[[x, y]].dropna()
    if subset.empty:
        return

    fig, ax = plt.subplots()
    ax.scatter(subset[x], subset[y], alpha=0.7)
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel or title(x))
    ax.set_ylabel(ylabel or title(y))
    savefig(fig, path)


def plot_box(df, column, group, path, plot_title):
    if column not in df or group not in df:
        return

    subset = df[[column, group]].dropna()
    if subset.empty:
        return

    groups = [
        group_data[column].values
        for _, group_data in subset.groupby(group)
    ]
    labels = [
        str(name)
        for name, _ in subset.groupby(group)
    ]

    if not groups:
        return

    fig, ax = plt.subplots()
    ax.boxplot(groups, labels=labels)
    ax.set_title(plot_title)
    ax.set_xlabel(title(group))
    ax.set_ylabel(title(column))
    savefig(fig, path)


# ---------------------------------------------------------------------
# TOURNAMENT ANALYSIS
# ---------------------------------------------------------------------

def analyze_tournaments(df: pd.DataFrame, out: Path) -> list[str]:
    if df.empty:
        return []

    df = normalize_tournament(df)
    tables = []
    plots = []

    # 1. Win rate by robot
    if {"name", "total_wins"}.issubset(df.columns):
        win = df.groupby("name")["total_wins"].mean().sort_values(ascending=False)
        plot_bar(win, out / "01_avg_wins_by_robot.png",
                 "Average Wins by Robot", "Average Wins")
        tables.append(("robot_win_summary", win.reset_index()))

    # 2. Survival rate
    if {"name", "survival_rate"}.issubset(df.columns):
        surv = df.groupby("name")["survival_rate"].mean().sort_values(ascending=False)
        plot_bar(surv, out / "02_survival_rate_by_robot.png",
                 "Average Survival Rate by Robot", "Survival Rate")
        tables.append(("robot_survival_summary", surv.reset_index()))

    # 3. Efficiency
    if {"name", "avg_efficiency"}.issubset(df.columns):
        eff = df.groupby("name")["avg_efficiency"].mean().sort_values(ascending=False)
        plot_bar(eff, out / "03_efficiency_by_robot.png",
                 "Average Efficiency by Robot", "Efficiency")
        tables.append(("robot_efficiency_summary", eff.reset_index()))

    # 4. Mine deaths
    if {"name", "mine_death_rate"}.issubset(df.columns):
        mine = df.groupby("name")["mine_death_rate"].mean().sort_values(ascending=False)
        plot_bar(mine, out / "04_mine_death_rate.png",
                 "Mine Death Rate by Robot", "Mine Death Rate")
        tables.append(("mine_death_summary", mine.reset_index()))

    # 5. Collision deaths
    if {"name", "collision_death_rate"}.issubset(df.columns):
        collision = df.groupby("name")["collision_death_rate"].mean().sort_values(ascending=False)
        plot_bar(collision, out / "05_collision_death_rate.png",
                 "Collision Death Rate by Robot", "Collision Death Rate")
        tables.append(("collision_death_summary", collision.reset_index()))

    # 6. Timeouts
    if {"name", "timeout_rate"}.issubset(df.columns):
        timeout = df.groupby("name")["timeout_rate"].mean().sort_values(ascending=False)
        plot_bar(timeout, out / "06_timeout_rate.png",
                 "Timeout Rate by Robot", "Timeout Rate")
        tables.append(("timeout_summary", timeout.reset_index()))

    # 7. Wait time
    if {"name", "avg_wait_per_round"}.issubset(df.columns):
        waits = df.groupby("name")["avg_wait_per_round"].mean().sort_values(ascending=False)
        plot_bar(waits, out / "07_waiting_behavior.png",
                 "Average Waiting per Round", "Wait Rounds")
        tables.append(("waiting_summary", waits.reset_index()))

    # 8. Speed vs efficiency
    plot_scatter(
        df,
        "speed",
        "avg_efficiency",
        out / "08_speed_vs_efficiency.png",
        "Speed vs Efficiency",
    )

    # 9. Speed vs survival
    plot_scatter(
        df,
        "speed",
        "survival_rate",
        out / "09_speed_vs_survival.png",
        "Speed vs Survival",
    )

    # 10. Risk tolerance vs mine deaths
    plot_scatter(
        df,
        "risk_tolerance",
        "mine_death_rate",
        out / "10_risk_vs_mine_deaths.png",
        "Risk Tolerance vs Mine Death Rate",
    )

    # 11. Risk tolerance vs efficiency
    plot_scatter(
        df,
        "risk_tolerance",
        "avg_efficiency",
        out / "11_risk_vs_efficiency.png",
        "Risk Tolerance vs Efficiency",
    )

    # 12. Patience vs waiting
    plot_scatter(
        df,
        "patience",
        "avg_wait_per_round",
        out / "12_patience_vs_waiting.png",
        "Patience vs Waiting Behavior",
    )

    # 13. Patience vs survival
    plot_scatter(
        df,
        "patience",
        "survival_rate",
        out / "13_patience_vs_survival.png",
        "Patience vs Survival",
    )

    # 14. Risk vs survival
    plot_scatter(
        df,
        "risk_tolerance",
        "survival_rate",
        out / "14_risk_vs_survival.png",
        "Risk Tolerance vs Survival",
    )

    # 15. Efficiency distributions
    if "avg_efficiency" in df:
        plot_hist(
            df["avg_efficiency"],
            out / "15_efficiency_distribution.png",
            "Distribution of Efficiency",
            "Efficiency",
        )

    # 16. Survival distribution
    if "survival_rate" in df:
        plot_hist(
            df["survival_rate"],
            out / "16_survival_distribution.png",
            "Distribution of Survival Rate",
            "Survival Rate",
        )

    # 17. Robot efficiency boxplot
    plot_box(
        df,
        "avg_efficiency",
        "name",
        out / "17_efficiency_boxplot.png",
        "Efficiency Distribution by Robot",
    )

    # 18. Survival boxplot
    plot_box(
        df,
        "survival_rate",
        "name",
        out / "18_survival_boxplot.png",
        "Survival Distribution by Robot",
    )

    # 19. Risk behavior profile
    behavior_cols = [
        "risk_tolerance",
        "patience",
        "speed",
        "avg_efficiency",
        "survival_rate",
        "mine_death_rate",
        "collision_death_rate",
        "timeout_rate",
    ]

    available = [c for c in behavior_cols if c in df.columns]

    if len(available) >= 3:
        corr = df[available].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(corr.values, aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels([title(c) for c in corr.columns], rotation=45, ha="right")
        ax.set_yticklabels([title(c) for c in corr.index])
        ax.set_title("Tournament Behavioral Correlation Matrix")

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

        savefig(fig, out / "19_tournament_correlation_matrix.png")

    # 20. Per-seed performance
    if {"meta_seed", "name", "avg_efficiency"}.issubset(df.columns):
        seed_eff = df.pivot_table(
            index="meta_seed",
            columns="name",
            values="avg_efficiency",
            aggfunc="mean",
        )

        fig, ax = plt.subplots()
        seed_eff.plot(marker="o", ax=ax)
        ax.set_title("Efficiency Across Seeds")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Efficiency")
        savefig(fig, out / "20_efficiency_across_seeds.png")

    # 21. Win percentage by seed
    if {"meta_seed", "name", "win_pct"}.issubset(df.columns):
        seed_win = df.pivot_table(
            index="meta_seed",
            columns="name",
            values="win_pct",
            aggfunc="mean",
        )

        fig, ax = plt.subplots()
        seed_win.plot(marker="o", ax=ax)
        ax.set_title("Win Rate Across Seeds")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Win Rate")
        savefig(fig, out / "21_win_rate_across_seeds.png")

    # 22. Death composition
    if {"name", "total_mine_deaths", "total_collision_deaths"}.issubset(df.columns):
        deaths = df.groupby("name")[
            ["total_mine_deaths", "total_collision_deaths"]
        ].mean()

        fig, ax = plt.subplots()
        deaths.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Death Composition: Mines vs Collisions")
        ax.set_ylabel("Average Deaths")
        ax.tick_params(axis="x", rotation=0)
        savefig(fig, out / "22_death_composition.png")

    # 23. Speed distribution by robot
    plot_box(
        df,
        "speed",
        "name",
        out / "23_speed_distribution.png",
        "Speed Distribution by Robot",
    )

    # 24. Risk distribution by robot
    plot_box(
        df,
        "risk_tolerance",
        "name",
        out / "24_risk_distribution.png",
        "Risk Tolerance Distribution by Robot",
    )

    # 25. Patience distribution by robot
    plot_box(
        df,
        "patience",
        "name",
        out / "25_patience_distribution.png",
        "Patience Distribution by Robot",
    )

    # 26. Pareto-style efficiency vs survival
    if {"avg_efficiency", "survival_rate", "name"}.issubset(df.columns):
        fig, ax = plt.subplots()

        for name, group in df.groupby("name"):
            ax.scatter(
                group["survival_rate"],
                group["avg_efficiency"],
                label=name,
                alpha=0.7,
            )

        ax.set_title("Efficiency vs Survival Trade-off")
        ax.set_xlabel("Survival Rate")
        ax.set_ylabel("Efficiency")
        ax.legend()
        savefig(fig, out / "26_efficiency_survival_tradeoff.png")

    # 27. Composite score
    required = {"avg_efficiency", "survival_rate", "win_pct"}
    if required.issubset(df.columns):
        composite = df.copy()
        composite["composite_score"] = (
            0.40 * composite["avg_efficiency"]
            + 0.40 * composite["survival_rate"]
            + 0.20 * composite["win_pct"]
        )

        score = composite.groupby("name")["composite_score"].mean().sort_values(
            ascending=False
        )

        plot_bar(
            score,
            out / "27_composite_performance_score.png",
            "Composite Performance Score",
            "Score",
        )

        tables.append(("composite_score", score.reset_index()))

    for name, table in tables:
        save_table(table, out / f"{name}.csv")

    return [
        "Tournament-level analysis completed.",
        f"Generated {len(list(out.glob('*.png')))} tournament plots.",
    ]


# ---------------------------------------------------------------------
# EXPERIMENT ANALYSIS
# ---------------------------------------------------------------------

def analyze_experiments(df: pd.DataFrame, out: Path) -> list[str]:
    if df.empty:
        return []

    df = normalize_experiments(df)
    robot_df = melt_experiments(df)

    if robot_df.empty:
        return []

    tables = []

    # 28. Config success rate
    success = robot_df.groupby("config")["reached"].mean().sort_values(ascending=False)
    plot_bar(
        success,
        out / "28_config_success_rate.png",
        "Success Rate by Configuration",
        "Success Rate",
    )
    tables.append(("config_success", success.reset_index()))

    # 29. Config efficiency
    efficiency = robot_df.groupby("config")["efficiency"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        efficiency,
        out / "29_config_efficiency.png",
        "Efficiency by Configuration",
        "Efficiency",
    )
    tables.append(("config_efficiency", efficiency.reset_index()))

    # 30. Config mine deaths
    mine = robot_df.groupby("config")["hit_mine"].mean().sort_values(ascending=False)
    plot_bar(
        mine,
        out / "30_config_mine_death_rate.png",
        "Mine Death Rate by Configuration",
        "Mine Death Rate",
    )

    # 31. Config collision deaths
    collision = robot_df.groupby("config")["hit_robot"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        collision,
        out / "31_config_collision_rate.png",
        "Robot Collision Rate by Configuration",
        "Collision Rate",
    )

    # 32. Config timeout rate
    timeout = robot_df.groupby("config")["timed_out"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        timeout,
        out / "32_config_timeout_rate.png",
        "Timeout Rate by Configuration",
        "Timeout Rate",
    )

    # 33. Config replanning
    replans = robot_df.groupby("config")["replans"].mean().sort_values(ascending=False)
    plot_bar(
        replans,
        out / "33_config_replanning.png",
        "Average Replanning by Configuration",
        "Replans",
    )

    # 34. Config waiting
    waits = robot_df.groupby("config")["waits"].mean().sort_values(ascending=False)
    plot_bar(
        waits,
        out / "34_config_waiting.png",
        "Average Waiting by Configuration",
        "Waits",
    )

    # 35. Unknown cells
    unknown = robot_df.groupby("config")["unknown_cells"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        unknown,
        out / "35_unknown_cells_entered.png",
        "Unknown Cells Entered by Configuration",
        "Unknown Cells",
    )

    # 36. Excess steps
    excess = robot_df.groupby("config")["excess_steps"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        excess,
        out / "36_excess_steps.png",
        "Excess Steps Over Oracle by Configuration",
        "Excess Steps",
    )

    # 37. Efficiency boxplot
    plot_box(
        robot_df,
        "efficiency",
        "config",
        out / "37_efficiency_config_boxplot.png",
        "Efficiency Distribution by Configuration",
    )

    # 38. Replans vs efficiency
    plot_scatter(
        robot_df,
        "replans",
        "efficiency",
        out / "38_replans_vs_efficiency.png",
        "Replanning vs Efficiency",
    )

    # 39. Waits vs efficiency
    plot_scatter(
        robot_df,
        "waits",
        "efficiency",
        out / "39_waits_vs_efficiency.png",
        "Waiting vs Efficiency",
    )

    # 40. Unknown exploration vs success
    plot_scatter(
        robot_df,
        "unknown_cells",
        "efficiency",
        out / "40_unknown_cells_vs_efficiency.png",
        "Unknown Exploration vs Efficiency",
    )

    # 41. Oracle ratio by robot
    robot_eff = robot_df.groupby("robot")["efficiency"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        robot_eff,
        out / "41_robot_efficiency_experiment.png",
        "Experiment Efficiency by Robot",
        "Efficiency",
    )

    # 42. Success by robot
    robot_success = robot_df.groupby("robot")["reached"].mean().sort_values(
        ascending=False
    )
    plot_bar(
        robot_success,
        out / "42_robot_success_experiment.png",
        "Experiment Success Rate by Robot",
        "Success Rate",
    )

    # 43. Heatmap configuration x robot
    pivot = robot_df.pivot_table(
        index="config",
        columns="robot",
        values="efficiency",
        aggfunc="mean",
    )

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        image = ax.imshow(pivot.values, aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_title("Efficiency Heatmap: Configuration × Robot")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if not pd.isna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center")

        savefig(fig, out / "43_config_robot_efficiency_heatmap.png")

    # 44. Success heatmap
    pivot_success = robot_df.pivot_table(
        index="config",
        columns="robot",
        values="reached",
        aggfunc="mean",
    )

    if not pivot_success.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        image = ax.imshow(pivot_success.values, aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(pivot_success.columns)))
        ax.set_yticks(range(len(pivot_success.index)))
        ax.set_xticklabels(pivot_success.columns)
        ax.set_yticklabels(pivot_success.index)
        ax.set_title("Success Heatmap: Configuration × Robot")

        for i in range(len(pivot_success.index)):
            for j in range(len(pivot_success.columns)):
                value = pivot_success.iloc[i, j]
                if not pd.isna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center")

        savefig(fig, out / "44_config_robot_success_heatmap.png")

    # 45. Per-round efficiency trend
    if "round_idx" in robot_df:
        trend = robot_df.groupby(["round_idx", "config"])["efficiency"].mean().unstack()

        fig, ax = plt.subplots()
        trend.plot(marker="o", ax=ax)
        ax.set_title("Efficiency Across Rounds")
        ax.set_xlabel("Round")
        ax.set_ylabel("Efficiency")
        savefig(fig, out / "45_efficiency_across_rounds.png")

    # 46. Per-round success trend
    if "round_idx" in robot_df:
        trend = robot_df.groupby(["round_idx", "config"])["reached"].mean().unstack()

        fig, ax = plt.subplots()
        trend.plot(marker="o", ax=ax)
        ax.set_title("Success Rate Across Rounds")
        ax.set_xlabel("Round")
        ax.set_ylabel("Success Rate")
        savefig(fig, out / "46_success_across_rounds.png")

    # 47. Replanning distribution
    plot_hist(
        robot_df["replans"],
        out / "47_replanning_distribution.png",
        "Distribution of Replanning Events",
        "Replans",
    )

    # 48. Waiting distribution
    plot_hist(
        robot_df["waits"],
        out / "48_wait_distribution.png",
        "Distribution of Waiting Events",
        "Waits",
    )

    # 49. Excess-step distribution
    plot_hist(
        robot_df["excess_steps"],
        out / "49_excess_step_distribution.png",
        "Distribution of Excess Steps Over Oracle",
        "Excess Steps",
    )

    # 50. Experimental correlation matrix
    correlation_cols = [
        "reached",
        "hit_mine",
        "hit_robot",
        "replans",
        "oracle_steps",
        "steps",
        "timed_out",
        "unknown_cells",
        "waits",
        "efficiency",
        "excess_steps",
    ]

    available = [c for c in correlation_cols if c in robot_df.columns]

    if len(available) >= 3:
        corr = robot_df[available].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(corr.values, aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels([title(c) for c in corr.columns], rotation=45, ha="right")
        ax.set_yticklabels([title(c) for c in corr.index])
        ax.set_title("Experiment Behavior Correlation Matrix")

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

        savefig(fig, out / "50_experiment_correlation_matrix.png")

    # 51. Configuration scorecard
    scorecard = robot_df.groupby("config").agg(
        success_rate=("reached", "mean"),
        efficiency=("efficiency", "mean"),
        mine_death_rate=("hit_mine", "mean"),
        collision_rate=("hit_robot", "mean"),
        timeout_rate=("timed_out", "mean"),
        avg_replans=("replans", "mean"),
        avg_waits=("waits", "mean"),
        avg_unknown_cells=("unknown_cells", "mean"),
        avg_excess_steps=("excess_steps", "mean"),
    ).reset_index()

    # Higher is better for success and efficiency.
    # Lower is better for all failure/overhead measures.
    scorecard["overall_score"] = (
        0.35 * scorecard["success_rate"]
        + 0.30 * scorecard["efficiency"]
        + 0.10 * (1 - scorecard["mine_death_rate"])
        + 0.10 * (1 - scorecard["collision_rate"])
        + 0.05 * (1 - scorecard["timeout_rate"])
        + 0.05 * (1 - scorecard["avg_replans"] /
                  max(scorecard["avg_replans"].max(), 1))
        + 0.05 * (1 - scorecard["avg_waits"] /
                  max(scorecard["avg_waits"].max(), 1))
    )

    scorecard = scorecard.sort_values("overall_score", ascending=False)
    save_table(scorecard, out / "configuration_scorecard.csv")

    plot_bar(
        scorecard.set_index("config")["overall_score"],
        out / "51_configuration_overall_score.png",
        "Overall Configuration Score",
        "Score",
    )

    tables.append(("experiment_robot_summary", robot_df.groupby("robot").agg(
        success_rate=("reached", "mean"),
        efficiency=("efficiency", "mean"),
        mine_death_rate=("hit_mine", "mean"),
        collision_rate=("hit_robot", "mean"),
        timeout_rate=("timed_out", "mean"),
        replans=("replans", "mean"),
        waits=("waits", "mean"),
    ).reset_index()))

    for name, table in tables:
        save_table(table, out / f"{name}.csv")

    return [
        "Experiment-level analysis completed.",
        f"Generated {len(list(out.glob('*.png')))} total plots so far.",
    ]


# ---------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------

def generate_summary(data: dict, out: Path) -> None:
    lines = []

    lines.append("# AI GRID-WORLD ANALYSIS SUMMARY\n")

    lines.append("## Loaded datasets\n")

    for key, df in data.items():
        lines.append(f"- `{key}`: {len(df):,} rows × {len(df.columns):,} columns")

    if "tournament" in data:
        df = normalize_tournament(data["tournament"])

        lines.append("\n## Tournament-level observations\n")

        if "name" in df:
            for name, group in df.groupby("name"):
                lines.append(
                    f"- **{name}**: "
                    f"mean efficiency={group['avg_efficiency'].mean():.3f}, "
                    f"survival={group['survival_rate'].mean():.3f}, "
                    f"wins={group['total_wins'].mean():.2f}"
                )

    if "experiments" in data:
        robot_df = melt_experiments(data["experiments"])

        if not robot_df.empty:
            lines.append("\n## Configuration-level observations\n")

            summary = robot_df.groupby("config").agg(
                success=("reached", "mean"),
                efficiency=("efficiency", "mean"),
                mine_death=("hit_mine", "mean"),
                collision=("hit_robot", "mean"),
                timeout=("timed_out", "mean"),
                replans=("replans", "mean"),
                waits=("waits", "mean"),
            ).sort_values("success", ascending=False)

            for config, row in summary.iterrows():
                lines.append(
                    f"- **Config {config}**: "
                    f"success={row['success']:.3f}, "
                    f"efficiency={row['efficiency']:.3f}, "
                    f"mine_death={row['mine_death']:.3f}, "
                    f"collision={row['collision']:.3f}, "
                    f"timeout={row['timeout']:.3f}"
                )

    lines.append("\n## Important interpretation warnings\n")
    lines.append(
        "- A single seed is not enough to claim that one planner is better."
    )
    lines.append(
        "- A high completion rate can hide terrible efficiency or excessive waiting."
    )
    lines.append(
        "- Survival rate and completion rate are different metrics."
    )
    lines.append(
        "- Mine deaths and collision deaths should be analyzed separately."
    )
    lines.append(
        "- Replanning can indicate intelligence, but excessive replanning can also indicate instability."
    )
    lines.append(
        "- Oracle efficiency is only meaningful if the oracle path is valid under the same movement constraints."
    )

    (out / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# HTML INDEX
# ---------------------------------------------------------------------

def generate_html_index(out: Path) -> None:
    images = sorted(out.glob("*.png"))

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>AI Grid World Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "h1 { margin-bottom: 30px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 25px; }",
        ".card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }",
        ".card img { width: 100%; }",
        ".card h3 { font-size: 16px; }",
        "a { color: #1769aa; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>AI Grid-World Analysis Report</h1>",
        "<p>This report contains automatically generated tournament and experiment analysis.</p>",
        "<div class='grid'>",
    ]

    for image in images:
        html.extend([
            "<div class='card'>",
            f"<h3>{image.stem.replace('_', ' ').title()}</h3>",
            f"<a href='{image.name}'><img src='{image.name}'></a>",
            "</div>",
        ])

    html.extend([
        "</div>",
        "</body>",
        "</html>",
    ])

    (out / "index.html").write_text("\n".join(html), encoding="utf-8")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Directory containing all CSV/log files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Directory where analysis output will be written",
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = safe_mkdir(Path(args.output_dir))

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory does not exist: {logs_dir}")

    files = discover_files(logs_dir)

    print("\nDISCOVERED FILES")
    print("=" * 60)

    for key, path in files.items():
        print(f"{key:15} -> {path}")

    data = load_data(files)

    print("\nGENERATING TOURNAMENT ANALYSIS")
    print("=" * 60)

    if "tournament" in data:
        analyze_tournaments(data["tournament"], output_dir)

    print("\nGENERATING EXPERIMENT ANALYSIS")
    print("=" * 60)

    if "experiments" in data:
        analyze_experiments(data["experiments"], output_dir)

    print("\nGENERATING SUMMARY")
    print("=" * 60)

    generate_summary(data, output_dir)
    generate_html_index(output_dir)

    print("\nDONE")
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Plots generated: {len(list(output_dir.glob('*.png')))}")
    print(f"Tables generated: {len(list(output_dir.glob('*.csv')))}")
    print(f"HTML report: {(output_dir / 'index.html').resolve()}")
    print(f"Summary: {(output_dir / 'ANALYSIS_SUMMARY.md').resolve()}")


if __name__ == "__main__":
    main()
