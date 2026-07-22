"""
analysis/main.py — CLI entry point for the GridWorld analysis pipeline.

Replaces the monolithic analyze_gridworld.py with a clean, composable pipeline:

    Load data  →  Compute metrics  →  Generate plots  →  Write reports

Usage:
    # New format (experiments.py output):
    python -m analysis.main --experiment-dir logs/experiment_20260722_001/

    # Legacy CSV format (old experiments_master.csv):
    python -m analysis.main --logs-dir logs/

    # Both with custom output directory:
    python -m analysis.main --experiment-dir logs/exp_001/ --output-dir my_report/

    # Debug: skip plots (fast text-only report):
    python -m analysis.main --experiment-dir logs/exp_001/ --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.main",
        description="GridWorld experiment analyzer — generates plots, metrics, and reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--experiment-dir",
        metavar="PATH",
        help="Path to a new-format experiment package (contains config.json + runs.jsonl).",
    )
    source.add_argument(
        "--logs-dir",
        metavar="PATH",
        help="Path to legacy logs directory (contains experiments_master.csv).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory for figures and reports. "
             "Defaults to <experiment-dir>/figures/ or <logs-dir>/report/.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation (fast text-only report).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        default=False,
        help="Skip PDF generation even if weasyprint/pdfkit is available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print full tracebacks on errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full analysis pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    try:
        from analysis.loader import Experiment, detect_format, load_legacy_tournaments
        from analysis.metrics import runs_to_dataframe, build_summary_dict

        if args.experiment_dir:
            exp_path = Path(args.experiment_dir)
            print(f"[load] Loading new-format experiment: {exp_path}")
            exp = Experiment.load(exp_path)
            experiment_dir = exp_path
            tournament_df  = pd.DataFrame()   # tournament data not in new format
        else:
            logs_path = Path(args.logs_dir)
            print(f"[load] Loading legacy logs: {logs_path}")
            exp = Experiment.load_legacy(logs_path)
            experiment_dir = logs_path
            tournament_df  = load_legacy_tournaments(logs_path)

        print(f"[load] {len(exp.runs):,} runs loaded | "
              f"configs: {exp.configs} | robots: {exp.robots}")

    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Failed to load data: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Compute metrics
    # ------------------------------------------------------------------
    try:
        robot_df = runs_to_dataframe(exp.runs)
        summary  = build_summary_dict(exp.runs, exp.meta)
        print(f"[metrics] {len(robot_df):,} robot records processed")
    except Exception as exc:
        print(f"[ERROR] Metrics computation failed: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 3. Resolve output directories
    # ------------------------------------------------------------------
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = experiment_dir / "report"

    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.json next to config.json (or into report dir for legacy)
    summary_path = experiment_dir / "summary.json"
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[metrics] summary.json -> {summary_path}")
    except Exception as exc:
        print(f"[WARN] Could not write summary.json: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        try:
            from analysis.plots import plot_all
            n_plots = plot_all(robot_df, tournament_df, figures_dir)
            print(f"[plots] {n_plots} figures -> {figures_dir}")
        except Exception as exc:
            print(f"[WARN] Plot generation failed: {exc}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()
    else:
        print("[plots] Skipped (--no-plots)")

    # ------------------------------------------------------------------
    # 5. Generate reports
    # ------------------------------------------------------------------
    try:
        from analysis.report import write_all_reports
        paths = write_all_reports(
            experiment_dir=out_root,
            meta=exp.meta,
            summary=summary,
            robot_df=robot_df,
        )
        print(f"[report] report.md  -> {paths['md']}")
        print(f"[report] report.html -> {paths['html']}")
        if not args.no_pdf:
            if paths.get("pdf"):
                print(f"[report] report.pdf  -> {paths['pdf']}")
            else:
                print("[report] PDF skipped (install weasyprint or pdfkit to enable)")
    except Exception as exc:
        print(f"[WARN] Report generation failed: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()

    print("[done] Analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
