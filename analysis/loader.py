"""
analysis/loader.py — Load experiment data from new-format packages or legacy CSVs.

Two formats are supported:

  New format (schema_version >= "2.0"):
      experiment_dir/
          config.json
          runs.jsonl          ← NDJSON: one JSON object per line (primary)
          runs.json           ← JSON array fallback (backward compat)
          summary.json

  Legacy format (flat CSV files):
      logs/
          experiments_master.csv
          experiments_meta_master.csv

High-level API:
    exp = Experiment.load("logs/experiment_001/")  # new format
    exp = Experiment.load_legacy("logs/")          # legacy CSVs
    exp.runs        → list[RunResult]
    exp.meta        → ExperimentMeta
    exp.robots      → list[str]  (unique robot names found in runs)
    exp.configs     → list[str]  (unique config keys found in runs)

Low-level API:
    detect_format(path)          → "new" | "legacy"
    load_experiment(dir)         → (runs, meta)
    load_legacy(logs_dir)        → (runs, meta)
    load_legacy_tournaments(dir) → pd.DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.models import ExperimentMeta, RunResult, RobotResult, ROBOT_NAMES


# ---------------------------------------------------------------------------
# High-level Experiment API
# ---------------------------------------------------------------------------

class Experiment:
    """High-level wrapper around a loaded experiment.

    Preferred entry point for analysis code:

        exp = Experiment.load("logs/experiment_001/")
        exp = Experiment.load_legacy("logs/")

    Attributes:
        runs   : All RunResult objects for this experiment.
        meta   : ExperimentMeta configuration snapshot.
        robots : Unique robot names found across all runs.
        configs: Unique config keys found across all runs.
    """

    def __init__(self, runs: list[RunResult], meta: Optional[ExperimentMeta]):
        self.runs   = runs
        self.meta   = meta

    @property
    def robots(self) -> list[str]:
        """Unique robot names appearing in any run, in canonical order."""
        seen = {r.name for run in self.runs for r in run.robots}
        return [n for n in ROBOT_NAMES if n in seen]

    @property
    def configs(self) -> list[str]:
        """Unique config keys across all runs, in sorted order."""
        return sorted({run.config for run in self.runs})

    @classmethod
    def load(cls, experiment_dir: str | Path) -> "Experiment":
        """Load a new-format experiment package."""
        runs, meta = load_experiment(experiment_dir)
        return cls(runs, meta)

    @classmethod
    def load_legacy(cls, logs_dir: str | Path) -> "Experiment":
        """Load from legacy flat-CSV format."""
        runs, meta = load_legacy(logs_dir)
        return cls(runs, meta)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str | Path) -> str:
    """Return "new" if the path contains config.json, otherwise "legacy".

    Args:
        path: Path to either an experiment directory (new format) or a
              logs directory (legacy format).

    Returns:
        "new" or "legacy".
    """
    p = Path(path)
    if (p / "config.json").exists() and (p / "runs.json").exists():
        return "new"
    return "legacy"


# ---------------------------------------------------------------------------
# New-format loader
# ---------------------------------------------------------------------------

def load_experiment(experiment_dir: str | Path) -> tuple[list[RunResult], ExperimentMeta]:
    """Load a new-format experiment package.

    Args:
        experiment_dir: Directory containing config.json and runs.json.

    Returns:
        (runs, meta) — list of RunResult objects and the ExperimentMeta.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    d = Path(experiment_dir)

    config_path  = d / "config.json"
    ndjson_path  = d / "runs.jsonl"   # primary (NDJSON — streams well at scale)
    json_path    = d / "runs.json"    # fallback (older JSON-array format)

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {d}")
    if not ndjson_path.exists() and not json_path.exists():
        raise FileNotFoundError(f"Neither runs.jsonl nor runs.json found in {d}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    meta = ExperimentMeta(
        schema_version       = cfg.get("schema_version", "2.0"),
        timestamp            = cfg.get("timestamp", ""),
        experiment_name      = cfg.get("experiment_name", ""),
        experiment_type      = cfg.get("experiment_type", "comparison"),
        base_seed            = cfg.get("base_seed", 0),
        num_seeds            = cfg.get("num_seeds", 0),
        planner_configs      = cfg.get("planner_configs", {}),
        robot_names          = cfg.get("robot_names", ROBOT_NAMES),
        grid                 = cfg.get("grid", {}),
        sim                  = cfg.get("sim", {}),
        migrated_from_legacy = cfg.get("migrated_from_legacy", False),
    )

    # Load runs — NDJSON preferred for streaming performance
    raw_entries: list[dict] = []
    if ndjson_path.exists():
        with open(ndjson_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_entries.append(json.loads(line))
    else:
        with open(json_path, encoding="utf-8") as f:
            raw_entries = json.load(f).get("runs", [])

    runs: list[RunResult] = []
    for entry in raw_entries:
        robots = [
            RobotResult(
                name                  = r["name"],
                reached               = bool(r.get("reached", False)),
                hit_mine              = bool(r.get("hit_mine", False)),
                hit_robot             = bool(r.get("hit_robot", False)),
                timed_out             = bool(r.get("timed_out", False)),
                steps                 = int(r.get("steps", 0)),
                oracle_steps          = int(r.get("oracle_steps", 0)),
                replans               = int(r.get("replans", 0)),
                waits                 = int(r.get("waits", 0)),
                unknown_cells_entered = int(r.get("unknown_cells_entered", 0)),
            )
            for r in entry.get("robots", [])
        ]
        world = entry.get("world", {})
        runs.append(RunResult(
            run_id      = int(entry.get("run_id", 0)),
            config      = str(entry.get("config", "")),
            seed        = int(entry.get("seed", 0)),
            ticks       = int(entry.get("ticks", 0)),
            world_mines = int(world.get("mines", 0)),
            world_walls = int(world.get("walls", 0)),
            robots      = robots,
        ))

    return runs, meta


# ---------------------------------------------------------------------------
# Legacy CSV adapter
# ---------------------------------------------------------------------------

def load_legacy(
    logs_dir: str | Path,
) -> tuple[list[RunResult], Optional[ExperimentMeta]]:
    """Load experiment data from the legacy flat-CSV format.

    Reads experiments_master.csv (wide-format per-round data) and converts
    it into a list of RunResult objects compatible with the rest of the
    analysis pipeline.

    Also reads experiments_meta_master.csv to construct a partial
    ExperimentMeta (some fields are unavailable in the legacy format).

    Args:
        logs_dir: Directory containing the legacy CSV files.

    Returns:
        (runs, meta) — list of RunResult objects and ExperimentMeta (may be
        None if meta CSV is absent).
    """
    d = Path(logs_dir)
    experiments_path = d / "experiments_master.csv"

    if not experiments_path.exists():
        raise FileNotFoundError(f"experiments_master.csv not found in {d}")

    df = pd.read_csv(experiments_path, on_bad_lines="skip")
    runs = _adapt_experiments_df(df)

    meta = _load_legacy_meta(d)
    return runs, meta


def _adapt_experiments_df(df: pd.DataFrame) -> list[RunResult]:
    """Convert a legacy experiments_master DataFrame into RunResult objects."""
    runs: list[RunResult] = []
    run_id = 0

    for _, row in df.iterrows():
        robots: list[RobotResult] = []

        # Discover available robot indices from column names
        # (handles any N robots without hardcoding 4)
        robot_indices = _discover_robot_indices(df.columns)

        for i in robot_indices:
            prefix = f"summary_{i}_"
            name_col = f"{prefix}name"
            if name_col not in df.columns:
                continue

            name = row.get(name_col)
            # Skip robots not in our known set (e.g. stale data from old runs)
            if name not in {"Red", "Green", "Blue", "Yellow"}:
                continue

            steps        = _safe_int(row.get(f"{prefix}steps", 0))
            oracle_steps = _safe_int(row.get(f"{prefix}shortest_oracle", 0))

            robots.append(RobotResult(
                name                  = str(name),
                reached               = bool(row.get(f"{prefix}reached", False)),
                hit_mine              = bool(row.get(f"{prefix}hit_mine", False)),
                hit_robot             = bool(row.get(f"{prefix}hit_robot", False)),
                timed_out             = bool(row.get(f"{prefix}timed_out", False)),
                steps                 = steps,
                oracle_steps          = oracle_steps,
                replans               = _safe_int(row.get(f"{prefix}replans", 0)),
                waits                 = _safe_int(row.get(f"{prefix}waits", 0)),
                unknown_cells_entered = _safe_int(
                    row.get(f"{prefix}unknown_cells_entered", 0)
                ),
            ))

        if not robots:
            continue

        run_id += 1
        runs.append(RunResult(
            run_id      = run_id,
            config      = str(row.get("config", "")),
            seed        = _safe_int(row.get("round_seed", 0)),
            ticks       = _safe_int(row.get("ticks", 0)),
            world_mines = _safe_int(row.get("world_mines", 0)),
            world_walls = _safe_int(row.get("world_walls", 0)),
            robots      = robots,
        ))

    return runs


def _load_legacy_meta(logs_dir: Path) -> Optional[ExperimentMeta]:
    """Build a partial ExperimentMeta from legacy meta CSV, if it exists."""
    meta_path = logs_dir / "experiments_meta_master.csv"
    if not meta_path.exists():
        return None

    try:
        df = pd.read_csv(meta_path)
        if df.empty:
            return None
        row = df.iloc[-1]  # use the most recent experiment row
        import ast
        config_keys = ast.literal_eval(str(row.get("config_keys", "[]")))
        return ExperimentMeta(
            schema_version       = "1.0-legacy",
            timestamp            = str(row.get("timestamp", "")),
            experiment_name      = "Legacy Experiment",
            experiment_type      = str(row.get("type", "comparison")),
            base_seed            = int(row.get("seed", 0)),
            num_seeds            = int(row.get("num_seeds", 0)),
            planner_configs      = {k: {} for k in config_keys},
            robot_names          = ROBOT_NAMES,
            grid                 = {},
            sim                  = {},
            migrated_from_legacy = True,
        )
    except Exception:
        return None


def load_legacy_tournaments(logs_dir: str | Path) -> pd.DataFrame:
    """Load tournaments_master.csv as a raw DataFrame for tournament analysis.

    Returns an empty DataFrame if the file does not exist.
    """
    path = Path(logs_dir) / "tournaments_master.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, on_bad_lines="skip")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_robot_indices(columns: pd.Index) -> list[int]:
    """Extract unique robot index integers from summary_N_name column names."""
    import re
    indices: set[int] = set()
    for col in columns:
        m = re.match(r"^summary_(\d+)_name$", col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _safe_int(value) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return 0
