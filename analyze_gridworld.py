#!/usr/bin/env python3
"""
analyze_gridworld.py — Backward-compatibility shim.

This file is intentionally thin. All analysis logic has been moved into the
analysis/ package:

    analysis/models.py   — data types (RobotResult, RunResult, ExperimentMeta)
    analysis/loader.py   — data loading (new-format NDJSON + legacy CSV)
    analysis/metrics.py  — statistical computation and aggregation
    analysis/plots.py    — all 55 visualisations
    analysis/report.py   — Markdown / HTML / PDF report generation
    analysis/main.py     — CLI entry point

Usage (unchanged from before):

    python analyze_gridworld.py --logs-dir logs/
    python analyze_gridworld.py --experiment-dir logs/experiment_001/

Or run the package directly:

    python -m analysis.main --logs-dir logs/
"""

from analysis.main import main
import sys

if __name__ == "__main__":
    # Re-map legacy --logs-dir flag; --experiment-dir already matches.
    sys.exit(main())
