"""
migrate_logs.py — Migration utility to convert legacy flat-CSV logs to the new format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.loader import load_legacy
from experiments import build_summary_dict, generate_reports


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy CSV logs to the new experiment package format.")
    parser.add_argument("--input", type=str, default="logs/", help="Input directory containing legacy CSVs")
    parser.add_argument("--output", type=str, default="logs/migrated/", help="Output directory for the new format")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not (input_dir / "experiments_master.csv").exists():
        print(f"Error: Could not find experiments_master.csv in {input_dir}")
        return

    print(f"Loading legacy data from {input_dir}...")
    try:
        runs, meta = load_legacy(input_dir)
    except Exception as e:
        print(f"Failed to load legacy data: {e}")
        return

    if not runs:
        print("No runs found to migrate.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Warning: Some fields like planning_time may be unavailable in the legacy format.")
    
    # Generate and write config.json
    if meta:
        config_data = meta.to_dict()
    else:
        # Fallback if no meta
        config_data = {
            "schema_version": "1.0-legacy",
            "timestamp": "unknown",
            "experiment_name": "Legacy Migration",
            "experiment_type": "comparison",
            "base_seed": 0,
            "num_seeds": 0,
            "planner_configs": {},
            "robot_names": ["Blue", "Red", "Green", "Yellow"],
            "grid": {},
            "sim": {},
            "migrated_from_legacy": True
        }
    
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
        
    print(f"Wrote config to {config_file}")
    
    # Stream runs to runs.jsonl
    runs_file = output_dir / "runs.jsonl"
    runs_data = []
    
    with open(runs_file, "w") as f:
        for run in runs:
            r_dict = run.to_dict()
            runs_data.append(r_dict)
            f.write(json.dumps(r_dict) + "\n")
            
    print(f"Wrote {len(runs)} runs to {runs_file}")
    
    # Reconstruct summary.json
    config_keys = sorted(list(set(r.config for r in runs)))
    num_robots = len(runs[0].robots) if runs else 4
    
    summary_doc = build_summary_dict(runs_data, config_keys, num_robots)
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_doc, f, indent=2)
        
    print(f"Wrote summary to {summary_file}")
    
    # Generate reports
    print("Generating reports...")
    generate_reports(str(output_dir))
    
    print(f"\nMigration complete: {len(runs)} runs migrated. Output at {output_dir}")

if __name__ == "__main__":
    main()
