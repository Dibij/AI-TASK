"""
analysis/report.py — Generate markdown, html, and pdf reports.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.models import ExperimentMeta, CONFIG_LABELS

def _format_pct(val: float) -> str:
    """Format a float as a percentage string."""
    return f"{val * 100:.1f}%" if pd.notnull(val) else "N/A"

def _format_float(val: float) -> str:
    """Format a float to 3 decimal places."""
    return f"{val:.3f}" if pd.notnull(val) else "N/A"

def generate_markdown_report(
    meta: ExperimentMeta | None,
    summary: dict,
    robot_df: pd.DataFrame,
    figures_dir: Path,
) -> str:
    """Generate a comprehensive Markdown report from experiment data."""
    lines = []
    
    # 1. Experiment Overview
    lines.append("# Experiment Report\n")
    lines.append("## 1. Experiment Overview\n")
    if meta:
        lines.append(f"- **Name:** {meta.experiment_name}")
        lines.append(f"- **Timestamp:** {meta.timestamp}")
        lines.append(f"- **Type:** {meta.experiment_type}")
        lines.append(f"- **Base Seed:** {meta.base_seed} ({meta.num_seeds} seeds)")
    else:
        lines.append("- *Metadata not available.*")
    lines.append("")

    # 2. Configuration
    lines.append("## 2. Configuration\n")
    lines.append("| Config Key | Description |")
    lines.append("|---|---|")
    
    if not robot_df.empty:
        configs_present = sorted(robot_df["config"].unique())
        for cfg in configs_present:
            desc = CONFIG_LABELS.get(cfg, "Unknown Configuration")
            if meta and hasattr(meta, "planner_configs") and meta.planner_configs and cfg in meta.planner_configs:
                desc = meta.planner_configs[cfg].get("description", desc)
            lines.append(f"| **{cfg}** | {desc} |")
    else:
        lines.append("| N/A | No configs available |")
    lines.append("")

    # 3. Overall Performance
    lines.append("## 3. Overall Performance\n")
    overall = summary.get("overall", {})
    def _mean(m: str) -> float:
        val = overall.get(m, {})
        if isinstance(val, dict):
            return val.get("mean", 0.0)
        return val if pd.notnull(val) else 0.0

    lines.append(f"- **Overall Success Rate:** {_format_pct(_mean('success_rate'))}")
    lines.append(f"- **Overall Efficiency:** {_format_float(_mean('efficiency'))}")
    lines.append(f"- **Mine Death Rate:** {_format_pct(_mean('mine_death_rate'))}")
    lines.append(f"- **Collision Rate:** {_format_pct(_mean('collision_rate'))}")
    lines.append(f"- **Timeout Rate:** {_format_pct(_mean('timeout_rate'))}")
    lines.append("")

    # 4. Robot Performance
    lines.append("## 4. Robot Performance\n")
    r_group: pd.DataFrame = pd.DataFrame()
    if not robot_df.empty and "robot_name" in robot_df.columns:
        r_group = robot_df.groupby("robot_name").agg(
            success_rate=("reached", "mean"),
            efficiency=("efficiency", "mean"),
            replans=("replans", "mean"),
            waits=("waits", "mean"),
            mine_death_rate=("hit_mine", "mean")
        ).reset_index().rename(columns={"robot_name": "name"})

        lines.append("| Robot | Success Rate | Efficiency | Replans | Waits | Mine Death Rate |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in r_group.iterrows():
            lines.append(
                f"| {row['name']} | {_format_pct(row['success_rate'])} "
                f"| {_format_float(row['efficiency'])} | {_format_float(row['replans'])} "
                f"| {_format_float(row['waits'])} | {_format_pct(row['mine_death_rate'])} |"
            )
    else:
        lines.append("*No robot data available.*")
    lines.append("")

    # 5. Configuration Comparison
    lines.append("## 5. Configuration Comparison\n")
    c_group: pd.DataFrame = pd.DataFrame()
    if not robot_df.empty and "config" in robot_df.columns:
        c_group = robot_df.groupby("config").agg(
            success_rate=("reached", "mean"),
            efficiency=("efficiency", "mean"),
            replans=("replans", "mean"),
            waits=("waits", "mean"),
            mine_death_rate=("hit_mine", "mean")
        ).reset_index()

        lines.append("| Config | Success Rate | Efficiency | Replans | Waits | Mine Death Rate |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in c_group.iterrows():
            lines.append(
                f"| {row['config']} | {_format_pct(row['success_rate'])} "
                f"| {_format_float(row['efficiency'])} | {_format_float(row['replans'])} "
                f"| {_format_float(row['waits'])} | {_format_pct(row['mine_death_rate'])} |"
            )
    else:
        lines.append("*No config data available.*")
    lines.append("")

    # 6. Key Metrics
    lines.append("## 6. Key Metrics\n")
    lines.append("Below are the visualised key metrics from the experiment:\n")
    if figures_dir.exists():
        for fig in sorted(figures_dir.glob("*.png")):
            lines.append(f"![{fig.stem.replace('_', ' ').title()}](figures/{fig.name})")
    else:
        lines.append("*No figures available.*")
    lines.append("")

    # 7. Failure Analysis
    lines.append("## 7. Failure Analysis\n")
    if not robot_df.empty and "config" in robot_df.columns:
        f_group = robot_df.groupby("config").agg(
            mine_death=("hit_mine", "mean"),
            collision=("hit_robot", "mean"),
            timeout=("timed_out", "mean")
        ).reset_index()

        lines.append("| Config | Mine Death Rate | Collision Rate | Timeout Rate |")
        lines.append("|---|---|---|---|")
        for _, row in f_group.iterrows():
            lines.append(
                f"| {row['config']} | {_format_pct(row['mine_death'])} "
                f"| {_format_pct(row['collision'])} | {_format_pct(row['timeout'])} |"
            )
    else:
        lines.append("*No failure data available.*")
    lines.append("")

    # 8. Interesting Findings
    lines.append("## 8. Interesting Findings\n")
    if not c_group.empty and not r_group.empty:
        best_cfg  = c_group.loc[c_group["success_rate"].idxmax()]
        worst_cfg = c_group.loc[c_group["success_rate"].idxmin()]
        best_r    = r_group.loc[r_group["efficiency"].idxmax()]
        worst_r   = r_group.loc[r_group["efficiency"].idxmin()]

        lines.append(f"- **Best Performing Config:** {best_cfg['config']} with success rate {_format_pct(best_cfg['success_rate'])}.")
        lines.append(f"- **Worst Performing Config:** {worst_cfg['config']} with success rate {_format_pct(worst_cfg['success_rate'])}.")
        lines.append(f"- **Most Efficient Robot:** {best_r['name']} with efficiency {_format_float(best_r['efficiency'])}.")
        lines.append(f"- **Least Efficient Robot:** {worst_r['name']} with efficiency {_format_float(worst_r['efficiency'])}.")
    else:
        lines.append("*Not enough data for interesting findings.*")
    lines.append("")

    # 9. Performance Bottlenecks
    lines.append("## 9. Performance Bottlenecks\n")
    if not robot_df.empty:
        high_waits = c_group.loc[c_group["waits"].idxmax()]
        high_replans = c_group.loc[c_group["replans"].idxmax()]
        lines.append(f"- **Highest Wait Time:** Config {high_waits['config']} averages {_format_float(high_waits['waits'])} waits per run.")
        lines.append(f"- **Highest Replanning Overhead:** Config {high_replans['config']} averages {_format_float(high_replans['replans'])} replans per run.")
    else:
        lines.append("*Not enough data to determine bottlenecks.*")
    lines.append("")

    # 10. Planner Behaviour
    lines.append("## 10. Planner Behaviour\n")
    lines.append("Configurations exhibit distinct behavioural patterns. Higher replan counts often indicate dynamic pathfinding in response to unexpected obstacles, whereas high wait times suggest collision-avoidance mechanisms taking precedence over movement.\n")
    
    # 11. Suggested Improvements
    lines.append("## 11. Suggested Improvements\n")
    lines.append("Based on the data collected, consider the following improvements:")
    lines.append("1. **Tune Collision Avoidance:** Adjust wait priorities and fallback paths to reduce excessive idling when robots meet.")
    lines.append("2. **Optimise Replanning Thresholds:** Introduce a caching layer or replan-delay to prevent computational overhead in densely clustered obstacle regions.")
    lines.append("3. **Enhance Risk Modelling:** Adjust the risk-aware cost heuristic to balance safety against path efficiency, potentially reducing mine deaths without overly extending paths.")
    lines.append("")

    # WARNINGS
    lines.append("## Important Interpretation Warnings\n")
    lines.append("- A single seed is not enough to claim that one planner is better.")
    lines.append("- A high completion rate can hide terrible efficiency or excessive waiting.")
    lines.append("- Survival rate and completion rate are different metrics.")
    lines.append("- Mine deaths and collision deaths should be analyzed separately.")
    lines.append("- Replanning can indicate intelligence, but excessive replanning can also indicate instability.")
    lines.append("- Oracle efficiency is only meaningful if the oracle path is valid under the same movement constraints.")

    return "\n".join(lines)


def generate_html_report(
    markdown_text: str,
    figures_dir: Path,
    output_path: Path,
) -> Path:
    """Generate a styled HTML report from markdown text."""
    try:
        import markdown
        html_body = markdown.markdown(markdown_text, extensions=['tables'])
    except ImportError:
        html_body = f"<pre>{markdown_text}</pre>"
        
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>AI Grid World Analysis Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #1a1a2e; color: #e0e0e0; display: flex; }",
        "nav { width: 250px; background-color: #16213e; padding: 20px; height: 100vh; position: fixed; overflow-y: auto; }",
        "nav a { display: block; color: #e94560; text-decoration: none; margin-bottom: 10px; font-weight: bold; }",
        "nav a:hover { color: #fff; }",
        "main { margin-left: 270px; padding: 40px; max-width: 1000px; width: 100%; }",
        "h1, h2, h3 { color: #e94560; border-bottom: 1px solid #0f3460; padding-bottom: 5px; }",
        "table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }",
        "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #0f3460; }",
        "th { background-color: #0f3460; color: #fff; }",
        "tr:hover { background-color: #16213e; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }",
        ".card { background-color: #16213e; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }",
        ".card img { width: 100%; border-radius: 4px; }",
        ".card h3 { font-size: 16px; margin-top: 0; border: none; text-align: center; }",
        "</style>",
        "</head>",
        "<body>",
        "<nav>",
        "<h2>Navigation</h2>",
        "<a href='#1-experiment-overview'>1. Overview</a>",
        "<a href='#2-configuration'>2. Configuration</a>",
        "<a href='#3-overall-performance'>3. Overall Performance</a>",
        "<a href='#4-robot-performance'>4. Robot Performance</a>",
        "<a href='#5-configuration-comparison'>5. Config Comparison</a>",
        "<a href='#6-key-metrics'>6. Key Metrics</a>",
        "<a href='#7-failure-analysis'>7. Failure Analysis</a>",
        "<a href='#8-interesting-findings'>8. Interesting Findings</a>",
        "<a href='#9-performance-bottlenecks'>9. Bottlenecks</a>",
        "<a href='#10-planner-behaviour'>10. Planner Behaviour</a>",
        "<a href='#11-suggested-improvements'>11. Improvements</a>",
        "</nav>",
        "<main>",
        f"<div class='header'><p>Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>",
        html_body,
    ]
    
    images = []
    if figures_dir.exists():
        images = sorted(figures_dir.glob("*.png"))
        
    if images:
        html.extend(["<h2>Figure Gallery</h2>", "<div class='grid'>"])
        for image in images:
            html.extend([
                "<div class='card'>",
                f"<h3>{image.stem.replace('_', ' ').title()}</h3>",
                f"<a href='figures/{image.name}'><img src='figures/{image.name}' alt='{image.stem}'></a>",
                "</div>",
            ])
        html.append("</div>")

    html.extend([
        "</main>",
        "</body>",
        "</html>",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html), encoding="utf-8")
    return output_path


def generate_pdf_report(html_path: Path, output_path: Path) -> Path | None:
    """Generate a PDF report from the HTML report."""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(output_path))
        return output_path
    except Exception as e:
        pass
        
    try:
        import pdfkit
        pdfkit.from_file(str(html_path), str(output_path))
        return output_path
    except Exception as e:
        logging.warning(f"weasyprint and pdfkit failed or unavailable: {e}")
        return None


def write_all_reports(
    experiment_dir: Path,
    meta: ExperimentMeta | None,
    summary: dict,
    robot_df: pd.DataFrame,
) -> dict[str, Path | None]:
    """Generate markdown, html, and pdf reports."""
    figures_dir = experiment_dir / "figures"
    md_path = experiment_dir / "report.md"
    html_path = experiment_dir / "report.html"
    pdf_path = experiment_dir / "report.pdf"
    
    md_text = generate_markdown_report(meta, summary, robot_df, figures_dir)
    md_path.write_text(md_text, encoding="utf-8")
    
    generate_html_report(md_text, figures_dir, html_path)
    
    pdf_out = generate_pdf_report(html_path, pdf_path)
    
    return {
        "md": md_path,
        "html": html_path,
        "pdf": pdf_out
    }
