"""
analysis/plots.py — All visualisations for the GridWorld analysis pipeline.

Sections:
    1. STYLE & HELPERS        — rcParams, savefig, add_bar_labels, title()
    2. BASE PLOT HELPERS      — plot_bar, plot_hist, plot_scatter, plot_box
    3. TOURNAMENT PLOTS       — plots 01-27, wrapped in plot_tournament_figures()
    4. EXPERIMENT PLOTS       — plots 28-51, wrapped in plot_experiment_figures()
    5. NEW PLOTS              — radar chart (52), failure stacked bar (53),
                                colour-coded box plot (54), regression scatter (55)
    6. MASTER FUNCTION        — plot_all()

Each function that groups by robot name uses ROBOT_COLOURS from models.py so
colours are always consistent across every plot in the report.

Input convention (experiment plots):
    robot_df — long-format DataFrame produced by analysis.metrics.runs_to_dataframe().
               One row per (run x robot). Columns: run_id, config, seed, ticks,
               world_mines, world_walls, robot_name, reached, hit_mine, hit_robot,
               timed_out, steps, oracle_steps, efficiency, excess_steps,
               replans, waits, unknown_cells_entered.

Input convention (tournament plots):
    tournament_df — raw DataFrame from analysis.loader.load_legacy_tournaments().
                    Columns include name, avg_efficiency, survival_rate, etc.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from analysis.models import ROBOT_NAMES, ROBOT_COLOURS, CONFIG_LABELS

matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
warnings.filterwarnings("ignore")


# =============================================================================
# 1. STYLE & HELPERS
# =============================================================================

plt.rcParams.update({
    "figure.figsize":  (12, 7),
    "figure.dpi":      130,
    "savefig.dpi":     150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
})


def savefig(fig: plt.Figure, path: Path) -> None:
    """Save a figure with tight layout and close it to free memory."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, decimals: int = 2) -> None:
    """Annotate each bar in a bar chart with its numeric value."""
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=f"%.{decimals}f", padding=3, fontsize=8)
        except Exception:
            pass


def title(name: str) -> str:
    """Convert snake_case or kebab-case to Title Case."""
    return re.sub(r"[_\-]+", " ", str(name)).title()


def _robot_colours_for(names: list[str]) -> list[str]:
    """Return hex colours for a list of robot names (unknown names get grey)."""
    return [ROBOT_COLOURS.get(n, "#888888") for n in names]


# =============================================================================
# 2. BASE PLOT HELPERS
# =============================================================================

def plot_bar(
    series: pd.Series,
    path: Path,
    plot_title: str,
    ylabel: str = "",
    rotation: int = 0,
    colour_by_robot: bool = False,
) -> None:
    """Bar chart from a Series; optionally colour bars by robot name."""
    fig, ax = plt.subplots()
    colors = _robot_colours_for(list(series.index)) if colour_by_robot else None
    series.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(plot_title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    add_bar_labels(ax)
    savefig(fig, path)


def plot_hist(series: pd.Series, path: Path, plot_title: str, xlabel: str) -> None:
    """Histogram of a numeric series."""
    fig, ax = plt.subplots()
    series.dropna().plot(kind="hist", bins=20, ax=ax)
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel)
    savefig(fig, path)


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    plot_title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Scatter plot of two numeric columns."""
    if x not in df.columns or y not in df.columns:
        return
    subset = df[[x, y]].dropna()
    if subset.empty:
        return
    fig, ax = plt.subplots()
    ax.scatter(subset[x], subset[y], alpha=0.6)
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel or title(x))
    ax.set_ylabel(ylabel or title(y))
    savefig(fig, path)


def plot_box(
    df: pd.DataFrame,
    column: str,
    group: str,
    path: Path,
    plot_title: str,
    colour_by_robot: bool = False,
) -> None:
    """Box plot of a numeric column grouped by a categorical column."""
    if column not in df.columns or group not in df.columns:
        return
    subset = df[[column, group]].dropna()
    if subset.empty:
        return

    groups_names = [str(n) for n, _ in subset.groupby(group)]
    groups_data  = [g[column].values for _, g in subset.groupby(group)]
    if not groups_data:
        return

    fig, ax = plt.subplots()
    bp = ax.boxplot(groups_data, patch_artist=colour_by_robot)
    ax.set_xticklabels(groups_names)
    ax.set_title(plot_title)
    ax.set_xlabel(title(group))
    ax.set_ylabel(title(column))

    if colour_by_robot:
        for patch, name in zip(bp["boxes"], groups_names):
            patch.set_facecolor(ROBOT_COLOURS.get(name, "#888888"))
            patch.set_alpha(0.7)

    savefig(fig, path)


def _annotated_heatmap(
    pivot: pd.DataFrame,
    path: Path,
    plot_title: str,
    cbar_label: str = "",
) -> None:
    """Viridis heatmap with cell value annotations and a labelled colorbar."""
    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_title(plot_title)
    mean_val = float(np.nanmean(pivot.values))
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                txt_color = "white" if float(val) < mean_val else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color)
    savefig(fig, path)


# =============================================================================
# 3. TOURNAMENT PLOTS  (plots 01-27)
# =============================================================================

def plot_tournament_figures(df: pd.DataFrame, out: Path) -> None:
    """Generate plots 01-27 from tournaments_master data."""
    if df.empty:
        return
    out.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "avg_efficiency", "avg_wait_per_round", "collision_death_rate",
        "completion_rate", "meta_max_ticks", "meta_num_robots", "meta_rounds",
        "meta_seed", "mine_death_rate", "patience", "risk_tolerance", "speed",
        "survival_rate", "timeout_rate", "total_collision_deaths",
        "total_mine_deaths", "total_timeouts", "total_wins",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "total_wins" in df.columns and "meta_rounds" in df.columns:
        df["win_pct"] = df["total_wins"] / df["meta_rounds"].replace(0, np.nan)

    # 01-07: grouped bar charts by robot
    for metric, fname, ptitle, ylabel in [
        ("total_wins",         "01_avg_wins_by_robot.png",   "Average Wins by Robot",        "Average Wins"),
        ("survival_rate",      "02_survival_rate_by_robot.png", "Survival Rate by Robot",    "Survival Rate"),
        ("avg_efficiency",     "03_efficiency_by_robot.png", "Efficiency by Robot",           "Efficiency"),
        ("mine_death_rate",    "04_mine_death_rate.png",     "Mine Death Rate by Robot",      "Mine Death Rate"),
        ("collision_death_rate","05_collision_death_rate.png","Collision Death Rate by Robot","Collision Death Rate"),
        ("timeout_rate",       "06_timeout_rate.png",        "Timeout Rate by Robot",         "Timeout Rate"),
        ("avg_wait_per_round", "07_waiting_behavior.png",    "Average Waiting per Round",     "Wait Rounds"),
    ]:
        if {"name", metric}.issubset(df.columns):
            s = df.groupby("name")[metric].mean().reindex(
                [n for n in ROBOT_NAMES if n in df["name"].unique()]
            ).dropna()
            plot_bar(s, out / fname, ptitle, ylabel, colour_by_robot=True)

    # 08-14: scatter (personality vs outcomes)
    for x, y, fname, ptitle in [
        ("speed",          "avg_efficiency",     "08_speed_vs_efficiency.png",  "Speed vs Efficiency"),
        ("speed",          "survival_rate",      "09_speed_vs_survival.png",    "Speed vs Survival"),
        ("risk_tolerance", "mine_death_rate",    "10_risk_vs_mine_deaths.png",  "Risk Tolerance vs Mine Death Rate"),
        ("risk_tolerance", "avg_efficiency",     "11_risk_vs_efficiency.png",   "Risk Tolerance vs Efficiency"),
        ("patience",       "avg_wait_per_round", "12_patience_vs_waiting.png",  "Patience vs Waiting Behavior"),
        ("patience",       "survival_rate",      "13_patience_vs_survival.png", "Patience vs Survival"),
        ("risk_tolerance", "survival_rate",      "14_risk_vs_survival.png",     "Risk Tolerance vs Survival"),
    ]:
        plot_scatter(df, x, y, out / fname, ptitle)

    # 15-16: distributions
    if "avg_efficiency" in df.columns:
        plot_hist(df["avg_efficiency"], out / "15_efficiency_distribution.png",
                  "Distribution of Efficiency", "Efficiency")
    if "survival_rate" in df.columns:
        plot_hist(df["survival_rate"], out / "16_survival_distribution.png",
                  "Distribution of Survival Rate", "Survival Rate")

    # 17-18: box plots with robot colour coding
    plot_box(df, "avg_efficiency", "name", out / "17_efficiency_boxplot.png",
             "Efficiency Distribution by Robot", colour_by_robot=True)
    plot_box(df, "survival_rate",  "name", out / "18_survival_boxplot.png",
             "Survival Distribution by Robot", colour_by_robot=True)

    # 19: correlation matrix
    behavior_cols = [
        "risk_tolerance", "patience", "speed", "avg_efficiency", "survival_rate",
        "mine_death_rate", "collision_death_rate", "timeout_rate",
    ]
    available = [c for c in behavior_cols if c in df.columns]
    if len(available) >= 3:
        corr = df[available].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        img = ax.imshow(corr.values, aspect="auto", cmap="viridis")
        fig.colorbar(img, ax=ax, label="Pearson r")
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

    # 20-21: line charts across seeds
    for metric, fname, ptitle, ylabel_lbl in [
        ("avg_efficiency", "20_efficiency_across_seeds.png", "Efficiency Across Seeds",  "Efficiency"),
        ("win_pct",        "21_win_rate_across_seeds.png",   "Win Rate Across Seeds",    "Win Rate"),
    ]:
        if {"meta_seed", "name", metric}.issubset(df.columns):
            pivot = df.pivot_table(index="meta_seed", columns="name",
                                   values=metric, aggfunc="mean")
            fig, ax = plt.subplots()
            for col in pivot.columns:
                ax.plot(pivot.index, pivot[col], marker="o",
                        color=ROBOT_COLOURS.get(str(col), "#888888"), label=col)
            ax.set_title(ptitle)
            ax.set_xlabel("Seed")
            ax.set_ylabel(ylabel_lbl)
            ax.legend()
            savefig(fig, out / fname)

    # 22: death composition stacked bar
    if {"name", "total_mine_deaths", "total_collision_deaths"}.issubset(df.columns):
        deaths = df.groupby("name")[["total_mine_deaths", "total_collision_deaths"]].mean()
        deaths = deaths.reindex([n for n in ROBOT_NAMES if n in deaths.index])
        fig, ax = plt.subplots()
        deaths.plot(kind="bar", stacked=True, ax=ax, color=["#E74C3C", "#2980B9"])
        ax.set_title("Death Composition: Mines vs Collisions")
        ax.set_ylabel("Average Deaths")
        ax.tick_params(axis="x", rotation=0)
        savefig(fig, out / "22_death_composition.png")

    # 23-25: personality box plots
    for metric, fname, ptitle in [
        ("speed",          "23_speed_distribution.png",    "Speed Distribution by Robot"),
        ("risk_tolerance", "24_risk_distribution.png",     "Risk Tolerance Distribution by Robot"),
        ("patience",       "25_patience_distribution.png", "Patience Distribution by Robot"),
    ]:
        plot_box(df, metric, "name", out / fname, ptitle, colour_by_robot=True)

    # 26: efficiency vs survival scatter coloured by robot
    if {"avg_efficiency", "survival_rate", "name"}.issubset(df.columns):
        fig, ax = plt.subplots()
        for name, group in df.groupby("name"):
            ax.scatter(group["survival_rate"], group["avg_efficiency"],
                       label=name, color=ROBOT_COLOURS.get(str(name), "#888888"), alpha=0.7)
        ax.set_title("Efficiency vs Survival Trade-off")
        ax.set_xlabel("Survival Rate")
        ax.set_ylabel("Efficiency")
        ax.legend()
        savefig(fig, out / "26_efficiency_survival_tradeoff.png")

    # 27: composite tournament score
    if {"avg_efficiency", "survival_rate", "win_pct"}.issubset(df.columns):
        tmp = df.copy()
        tmp["composite_score"] = (
            0.40 * tmp["avg_efficiency"]
            + 0.40 * tmp["survival_rate"]
            + 0.20 * tmp["win_pct"]
        )
        score = tmp.groupby("name")["composite_score"].mean()
        score = score.reindex([n for n in ROBOT_NAMES if n in score.index])
        plot_bar(score, out / "27_composite_performance_score.png",
                 "Composite Performance Score", "Score", colour_by_robot=True)


# =============================================================================
# 4. EXPERIMENT PLOTS  (plots 28-51)
# =============================================================================

def plot_experiment_figures(robot_df: pd.DataFrame, out: Path) -> None:
    """Generate plots 28-51 from the long-format experiment DataFrame."""
    if robot_df.empty:
        return
    out.mkdir(parents=True, exist_ok=True)

    # Normalise robot name column — metrics.py uses 'robot_name'; legacy used 'robot'
    rn_col = "robot_name" if "robot_name" in robot_df.columns else "robot"

    # 28-36: config-level bar charts
    for metric, fname, ptitle, ylabel in [
        ("reached",               "28_config_success_rate.png",   "Success Rate by Configuration",             "Success Rate"),
        ("efficiency",            "29_config_efficiency.png",     "Efficiency by Configuration",               "Efficiency"),
        ("hit_mine",              "30_config_mine_death_rate.png","Mine Death Rate by Configuration",          "Mine Death Rate"),
        ("hit_robot",             "31_config_collision_rate.png", "Collision Rate by Configuration",           "Collision Rate"),
        ("timed_out",             "32_config_timeout_rate.png",   "Timeout Rate by Configuration",             "Timeout Rate"),
        ("replans",               "33_config_replanning.png",     "Average Replanning by Configuration",       "Replans"),
        ("waits",                 "34_config_waiting.png",        "Average Waiting by Configuration",          "Waits"),
        ("unknown_cells_entered", "35_unknown_cells_entered.png", "Unknown Cells Entered by Configuration",    "Unknown Cells"),
        ("excess_steps",          "36_excess_steps.png",          "Excess Steps Over Oracle by Configuration", "Excess Steps"),
    ]:
        if metric not in robot_df.columns:
            continue
        s = robot_df.groupby("config")[metric].mean().sort_values(ascending=False)
        plot_bar(s, out / fname, ptitle, ylabel)

    # 37: efficiency box plot by config
    plot_box(robot_df, "efficiency", "config",
             out / "37_efficiency_config_boxplot.png",
             "Efficiency Distribution by Configuration")

    # 38-40: scatter plots
    for x, y, fname, ptitle in [
        ("replans",               "efficiency", "38_replans_vs_efficiency.png",        "Replanning vs Efficiency"),
        ("waits",                 "efficiency", "39_waits_vs_efficiency.png",          "Waiting vs Efficiency"),
        ("unknown_cells_entered", "efficiency", "40_unknown_cells_vs_efficiency.png",  "Unknown Exploration vs Efficiency"),
    ]:
        plot_scatter(robot_df, x, y, out / fname, ptitle)

    # 41-42: by robot
    if rn_col in robot_df.columns:
        present = [n for n in ROBOT_NAMES if n in robot_df[rn_col].unique()]
        for metric, fname, ptitle, ylabel in [
            ("efficiency", "41_robot_efficiency_experiment.png", "Efficiency by Robot",    "Efficiency"),
            ("reached",    "42_robot_success_experiment.png",    "Success Rate by Robot",  "Success Rate"),
        ]:
            if metric in robot_df.columns:
                s = robot_df.groupby(rn_col)[metric].mean().reindex(present)
                plot_bar(s, out / fname, ptitle, ylabel, colour_by_robot=True)

    # 43: efficiency heatmap config x robot
    if rn_col in robot_df.columns and "efficiency" in robot_df.columns:
        pivot = robot_df.pivot_table(
            index="config", columns=rn_col, values="efficiency", aggfunc="mean"
        )
        if not pivot.empty:
            _annotated_heatmap(pivot, out / "43_config_robot_efficiency_heatmap.png",
                               "Efficiency Heatmap: Configuration x Robot",
                               cbar_label="Mean Path Efficiency")

    # 44: success heatmap config x robot
    if rn_col in robot_df.columns and "reached" in robot_df.columns:
        pivot_s = robot_df.pivot_table(
            index="config", columns=rn_col, values="reached", aggfunc="mean"
        )
        if not pivot_s.empty:
            _annotated_heatmap(pivot_s, out / "44_config_robot_success_heatmap.png",
                               "Success Heatmap: Configuration x Robot",
                               cbar_label="Success Rate")

    # 45-46: per-round trends
    round_col = next((c for c in ("run_id", "round_idx") if c in robot_df.columns), None)
    if round_col:
        for metric, fname, ptitle, ylabel_lbl in [
            ("efficiency", "45_efficiency_across_rounds.png", "Efficiency Across Rounds", "Efficiency"),
            ("reached",    "46_success_across_rounds.png",    "Success Rate Across Rounds","Success Rate"),
        ]:
            if metric in robot_df.columns:
                trend = robot_df.groupby([round_col, "config"])[metric].mean().unstack()
                fig, ax = plt.subplots()
                trend.plot(marker="o", ax=ax)
                ax.set_title(ptitle)
                ax.set_xlabel("Round")
                ax.set_ylabel(ylabel_lbl)
                savefig(fig, out / fname)

    # 47-49: distributions
    for metric, fname, ptitle, xlabel in [
        ("replans",      "47_replanning_distribution.png",  "Distribution of Replanning Events", "Replans"),
        ("waits",        "48_wait_distribution.png",        "Distribution of Waiting Events",    "Waits"),
        ("excess_steps", "49_excess_step_distribution.png", "Distribution of Excess Steps",      "Excess Steps"),
    ]:
        if metric in robot_df.columns:
            plot_hist(robot_df[metric], out / fname, ptitle, xlabel)

    # 50: experiment correlation matrix
    corr_cols = [
        "reached", "hit_mine", "hit_robot", "replans", "oracle_steps",
        "steps", "timed_out", "unknown_cells_entered", "waits", "efficiency", "excess_steps",
    ]
    available = [c for c in corr_cols if c in robot_df.columns]
    if len(available) >= 3:
        corr = robot_df[available].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        img = ax.imshow(corr.values, aspect="auto", cmap="viridis")
        fig.colorbar(img, ax=ax, label="Pearson r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels([title(c) for c in corr.columns], rotation=45, ha="right")
        ax.set_yticklabels([title(c) for c in corr.index])
        ax.set_title("Experiment Behaviour Correlation Matrix")
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)
        savefig(fig, out / "50_experiment_correlation_matrix.png")

    # 51: configuration overall score (min-max normalized)
    sc_df = robot_df.groupby("config").agg(
        success_rate=("reached",    "mean"),
        efficiency=  ("efficiency", "mean"),
        mine_death_rate=("hit_mine",  "mean"),
        collision_rate= ("hit_robot", "mean"),
        timeout_rate=   ("timed_out", "mean"),
        avg_replans=    ("replans",   "mean"),
        avg_waits=      ("waits",     "mean"),
    ).reset_index()

    if not sc_df.empty:
        eps = 1e-9

        def _mm(s: pd.Series) -> pd.Series:
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + eps)

        sc_df["overall_score"] = (
            0.35 * sc_df["success_rate"]
            + 0.30 * sc_df["efficiency"].fillna(0.0)
            + 0.10 * (1.0 - sc_df["mine_death_rate"])
            + 0.10 * (1.0 - sc_df["collision_rate"])
            + 0.05 * (1.0 - sc_df["timeout_rate"])
            + 0.05 * (1.0 - _mm(sc_df["avg_replans"]))
            + 0.05 * (1.0 - _mm(sc_df["avg_waits"]))
        )
        sc_df = sc_df.sort_values("overall_score", ascending=False)
        plot_bar(sc_df.set_index("config")["overall_score"],
                 out / "51_configuration_overall_score.png",
                 "Overall Configuration Score", "Score")


# =============================================================================
# 5. NEW PLOTS  (52-55)
# =============================================================================

def plot_radar_chart(robot_df: pd.DataFrame, out: Path) -> None:
    """Plot 52 — Radar chart comparing robots across 6 performance dimensions.

    Dimensions: success rate, efficiency, mine safety, collision safety,
    timeout safety, low replans.  All axes normalised to [0, 1]; higher = better.
    """
    if robot_df.empty:
        return
    rn_col = "robot_name" if "robot_name" in robot_df.columns else "robot"
    if rn_col not in robot_df.columns:
        return

    dims = [
        ("Success Rate",      "reached",   "mean", False),
        ("Efficiency",        "efficiency","mean", False),
        ("Mine Safety",       "hit_mine",  "mean", True),
        ("Collision Safety",  "hit_robot", "mean", True),
        ("Timeout Safety",    "timed_out", "mean", True),
        ("Low Replans",       "replans",   "mean", True),
    ]
    dim_labels = [d[0] for d in dims]
    n = len(dim_labels)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    max_replans = float(robot_df["replans"].max()) if "replans" in robot_df.columns else 1.0

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

    for robot_name in ROBOT_NAMES:
        sub = robot_df[robot_df[rn_col] == robot_name]
        if sub.empty:
            continue

        values: list[float] = []
        for _, col, agg, invert in dims:
            if col not in sub.columns:
                values.append(0.0)
                continue
            v = float(sub[col].agg(agg) if not sub[col].isna().all() else 0.0)
            if invert:
                if col == "replans" and max_replans > 0:
                    v = v / max_replans  # normalise before inverting
                v = 1.0 - min(v, 1.0)
            values.append(v)

        values += values[:1]
        color = ROBOT_COLOURS.get(robot_name, "#888888")
        ax.plot(angles, values, linewidth=2, linestyle="solid",
                label=robot_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], dim_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Robot Performance Radar (Higher = Better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    savefig(fig, out / "52_robot_radar_chart.png")


def plot_failure_reasons(robot_df: pd.DataFrame, out: Path) -> None:
    """Plot 53 — Stacked bar showing failure composition per configuration."""
    needed = {"config", "reached", "hit_mine", "hit_robot", "timed_out"}
    if robot_df.empty or not needed.issubset(robot_df.columns):
        return

    grp = robot_df.groupby("config").agg(
        success=("reached",  "mean"),
        mine=   ("hit_mine", "mean"),
        coll=   ("hit_robot","mean"),
        timeout=("timed_out","mean"),
    ).reset_index()

    configs = grp["config"].tolist()
    x = list(range(len(configs)))

    fig, ax = plt.subplots()
    ax.bar(x, grp["success"], label="Success",    color="#27AE60", alpha=0.85)
    ax.bar(x, grp["mine"],    label="Mine Death", color="#E74C3C", alpha=0.85,
           bottom=grp["success"])
    ax.bar(x, grp["coll"],    label="Collision",  color="#2980B9", alpha=0.85,
           bottom=grp["success"] + grp["mine"])
    ax.bar(x, grp["timeout"], label="Timeout",    color="#F1C40F", alpha=0.85,
           bottom=grp["success"] + grp["mine"] + grp["coll"])
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Proportion")
    ax.set_title("Failure Reasons by Configuration")
    ax.legend()
    savefig(fig, out / "53_failure_reasons.png")


def plot_efficiency_boxplot_coloured(robot_df: pd.DataFrame, out: Path) -> None:
    """Plot 54 — Efficiency box plots with each box coloured by robot name."""
    rn_col = "robot_name" if "robot_name" in robot_df.columns else "robot"
    if rn_col not in robot_df.columns or "efficiency" not in robot_df.columns:
        return
    plot_box(robot_df, "efficiency", rn_col,
             out / "54_efficiency_by_robot_boxplot.png",
             "Efficiency Distribution by Robot",
             colour_by_robot=True)


def plot_replans_vs_efficiency_regression(robot_df: pd.DataFrame, out: Path) -> None:
    """Plot 55 — Replans vs efficiency scatter with robot colours and OLS regression."""
    rn_col = "robot_name" if "robot_name" in robot_df.columns else "robot"
    needed = {"replans", "efficiency"}
    if not needed.issubset(robot_df.columns) or rn_col not in robot_df.columns:
        return

    sub = robot_df[["replans", "efficiency", rn_col]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots()
    for robot_name in ROBOT_NAMES:
        g = sub[sub[rn_col] == robot_name]
        if g.empty:
            continue
        ax.scatter(g["replans"], g["efficiency"],
                   alpha=0.5, color=ROBOT_COLOURS.get(robot_name, "#888888"),
                   label=robot_name, s=18)

    x_all = sub["replans"].values
    y_all = sub["efficiency"].values
    if len(x_all) >= 2:
        coeffs = np.polyfit(x_all, y_all, deg=1)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                color="white", linewidth=2, linestyle="--",
                label=f"Regression (slope={coeffs[0]:.3f})")

    ax.set_xlabel("Replans")
    ax.set_ylabel("Efficiency")
    ax.set_title("Replanning vs Efficiency (with Regression)")
    ax.legend(fontsize=8)
    savefig(fig, out / "55_replans_vs_efficiency_regression.png")


# =============================================================================
# 6. MASTER FUNCTION
# =============================================================================

def plot_all(
    robot_df: pd.DataFrame,
    tournament_df: pd.DataFrame,
    out: Path,
) -> int:
    """Generate all plots (01-55) and return the count of PNG files created.

    Args:
        robot_df:       Long-format experiment DataFrame (metrics.runs_to_dataframe).
        tournament_df:  Tournament DataFrame (loader.load_legacy_tournaments).
        out:            Root output directory.

    Returns:
        Number of PNG files generated.
    """
    out.mkdir(parents=True, exist_ok=True)

    if not tournament_df.empty:
        plot_tournament_figures(tournament_df, out)

    if not robot_df.empty:
        plot_experiment_figures(robot_df, out)
        plot_radar_chart(robot_df, out)
        plot_failure_reasons(robot_df, out)
        plot_efficiency_boxplot_coloured(robot_df, out)
        plot_replans_vs_efficiency_regression(robot_df, out)

    count = len(list(out.glob("*.png")))
    print(f"[plots] Generated {count} figures in {out}")
    return count
