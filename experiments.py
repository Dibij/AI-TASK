"""
experiments.py — Headless experiment runner for the multi-agent grid world.

Supports reproducible, paired-comparison experiments across different
planner configurations and personality sweeps.

Features:
- Event-based JSONL logging
- Crash recovery (resume from existing directory)
- Statistical summaries (mean, stdev, 95% CI, paired t-tests)
- Configurable output directory
"""

import argparse
import json
import os
import datetime
import math
import collections
from typing import Dict, List, Optional, Tuple
import convert_logs

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from planner import CONFIGS, PlannerConfig
from game import Game, MAX_TICKS


def init_gui():
    import pygame
    import main
    import sys
    pygame.init()
    pygame.display.set_caption("Multi-Agent Pathfinding Simulation — Headless Sweeps (GUI mode)")
    screen = pygame.display.set_mode((main.WIN_W, main.WIN_H))
    renderer = main.Renderer(screen)
    return pygame, screen, renderer, main.BG

def pump_gui(pygame, screen, renderer, bg, game):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            import sys
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                import sys
                sys.exit()
    screen.fill(bg)
    renderer.draw_grid(game)
    renderer.draw_sidebar(game, tps=1000) # Uncapped
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Experiment configuration registry
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS: Dict[str, Dict] = {
    "A": {
        "name"        : "Shortest-Path Baseline",
        "description" : ("A* with uniform cost=1 per cell. "
                         "Ignores mine probability, unknown penalty, and collision risk. "
                         "This is the experimental control condition."),
        "planner"     : "shortest",
        "config"      : CONFIGS["shortest"],
    },
    "B": {
        "name"        : "Risk-Aware A*",
        "description" : ("A* with composite cost: movement + mine_risk_weight*P(mine) "
                         "+ unknown_weight*is_unknown + collision_weight*is_predicted_occupied. "
                         "Weights: mine=3.0, unknown=0.2, collision=0.5. "
                         "These are heuristic values, not mathematically optimal."),
        "planner"     : "risk_aware",
        "config"      : CONFIGS["risk_aware"],
    },
    "C": {
        "name"        : "Conservative Risk-Aware A*",
        "description" : ("Like B but with unknown_weight=2.0 (10x larger). "
                         "Strongly prefers known-safe paths even if longer. "
                         "Tests hypothesis: avoiding unknowns reduces mine deaths."),
        "planner"     : "conservative",
        "config"      : CONFIGS["conservative"],
    },
    "D": {
        "name"        : "Risk-Aware A* (No Collision Avoidance)",
        "description" : ("Like B but collision_weight=0.0. "
                         "Measures the contribution of collision avoidance alone."),
        "planner"     : "risk_aware",
        "config"      : PlannerConfig(
            movement_cost    = 1.0,
            mine_risk_weight = 3.0,
            unknown_weight   = 0.2,
            collision_weight = 0.0,   # disabled
            avoid_weight     = 100.0,
            planner_type     = "risk_aware",
            collision_horizon= 5,
        ),
    },
    "E": {
        "name"        : "Risk-Aware A* (Full, Collision Enabled)",
        "description" : ("Identical to B (collision_weight=0.5). "
                         "Paired with D to measure collision avoidance contribution."),
        "planner"     : "risk_aware",
        "config"      : CONFIGS["risk_aware"],
    },
}

PERSONALITY_SWEEPS: Dict[str, Dict] = {
    "risk_tolerance": {
        "param"     : "risk_tolerance",
        "label"     : "Risk Tolerance",
        "values"    : [0.1, 0.3, 0.5, 0.7, 0.9],
        "planner"   : "risk_aware",
    },
    "patience": {
        "param"     : "patience",
        "label"     : "Patience",
        "values"    : [0.1, 0.3, 0.5, 0.7, 0.9],
        "planner"   : "risk_aware",
    },
    "speed": {
        "param"     : "speed",
        "label"     : "Speed",
        "values"    : [1, 2],
        "planner"   : "risk_aware",
    },
}


# ---------------------------------------------------------------------------
# Statistics Helpers
# ---------------------------------------------------------------------------

def calculate_stats(data: List[float]) -> Dict[str, float]:
    if not data:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "ci95": 0.0}
    n = len(data)
    mean = sum(data) / n
    sorted_data = sorted(data)
    mid = n // 2
    median = (sorted_data[mid] + sorted_data[~mid]) / 2.0 if n % 2 == 0 else sorted_data[mid]
    if n > 1:
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    ci95 = 1.96 * stdev / math.sqrt(n) if n > 0 else 0.0
    return {"mean": mean, "median": median, "stdev": stdev, "ci95": ci95}


def paired_t_test(baseline: List[float], treatment: List[float]) -> Tuple[float, float]:
    """Returns (t_stat, p_value)"""
    if not HAS_SCIPY or len(baseline) < 2 or len(treatment) < 2 or len(baseline) != len(treatment):
        return (0.0, 1.0)
    res = scipy.stats.ttest_rel(baseline, treatment)
    return (float(res.statistic), float(res.pvalue))


# ---------------------------------------------------------------------------
# Core Runner
# ---------------------------------------------------------------------------

def run_paired_comparison(config_keys: List[str], seed: int, num_seeds: int,
                          num_robots: int = 4, output_dir: str = "logs",
                          resume_dir: Optional[str] = None, verbose: bool = True,
                          use_gui: bool = False):
    """
    Run multiple experiment configurations on the SAME set of seeds (paired).
    """
    # 1. Setup output directory
    if resume_dir:
        exp_dir = resume_dir
        if not os.path.exists(exp_dir):
            raise ValueError(f"Resume directory {exp_dir} does not exist.")
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        keys_str = "_".join(config_keys)
        exp_dir = os.path.join(output_dir, f"experiment_{keys_str}_seed{seed}_{ts}")
        os.makedirs(exp_dir, exist_ok=True)
        
        meta = {
            "type": "comparison",
            "config_keys": config_keys,
            "seed": seed,
            "num_seeds": num_seeds,
            "num_robots": num_robots,
            "timestamp": ts,
        }
        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

    events_file = os.path.join(exp_dir, "events.jsonl")
    
    # 2. Load completed events (for resume)
    completed_events = set()
    if os.path.exists(events_file):
        with open(events_file, "r") as f:
            for line in f:
                evt = json.loads(line)
                if evt.get("type") == "round_result":
                    completed_events.add((evt["config"], evt["round_seed"]))
    
    events_out = open(events_file, "a")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Paired Comparison Experiment")
        print(f"  Configs : {config_keys}")
        print(f"  Base Seed: {seed}")
        print(f"  Seeds    : {num_seeds}")
        print(f"  Output   : {exp_dir}")
        if resume_dir:
            print(f"  Resuming : {len(completed_events)} rounds already complete.")
        print(f"{'='*60}\n")

    # 3. Run missing rounds
    try:
        for round_idx in range(num_seeds):
            round_seed = seed + round_idx
            for key in config_keys:
                if (key, round_seed) in completed_events:
                    continue
                    
                cfg = EXPERIMENT_CONFIGS[key]
                
                game = Game(
                    seed=round_seed,
                    planner_name=cfg["planner"],
                    planner_config=cfg["config"],
                    num_robots=num_robots,
                    max_rounds=1
                )
                
                if use_gui:
                    pygame, screen, renderer, bg = init_gui()

                while not game.tournament_over:
                    game.tick_simulation()
                    if use_gui:
                        pump_gui(pygame, screen, renderer, bg, game)
                    
                summary = game.round_summary[0]
                
                evt = {
                    "type": "round_result",
                    "config": key,
                    "round_idx": round_idx,
                    "round_seed": round_seed,
                    "summary": summary,
                    "ticks": game.tick,
                    "world_mines": game.world.count_mines(),
                    "world_walls": game.world.count_walls()
                }
                events_out.write(json.dumps(evt) + "\n")
                events_out.flush()
                
                if verbose:
                    reached = sum(1 for v in summary.values() if v["reached"])
                    mine_d  = sum(1 for v in summary.values() if v["hit_mine"])
                    coll_d  = sum(1 for v in summary.values() if v["hit_robot"])
                    time_o  = sum(1 for v in summary.values() if v["timed_out"])
                    print(f"Round {round_idx+1:03d}/{num_seeds} | Config {key:1s} | Seed {round_seed:5d} -> "
                          f"Reach:{reached} Mine:{mine_d} Coll:{coll_d} Time:{time_o} Ticks:{game.tick}")
    finally:
        events_out.close()
    
    # 4. Generate Final Summary
    _generate_final_summary(exp_dir, config_keys, num_robots)


def _generate_final_summary(exp_dir: str, config_keys: List[str], num_robots: int):
    results_by_config = collections.defaultdict(list)
    events_file = os.path.join(exp_dir, "events.jsonl")
    if not os.path.exists(events_file):
        return
        
    with open(events_file, "r") as f:
        for line in f:
            evt = json.loads(line)
            if evt.get("type") == "round_result":
                results_by_config[evt["config"]].append(evt)
                
    stats = {}
    
    for key in config_keys:
        evts = results_by_config[key]
        if not evts:
            continue
            
        round_efficiencies = []
        round_mine_deaths = []
        round_coll_deaths = []
        round_successes = []
        
        for evt in evts:
            effs = []
            mines = 0
            colls = 0
            succs = 0
            for r_id, r_stat in evt["summary"].items():
                if r_stat["reached"]:
                    succs += 1
                    effs.append((r_stat["shortest_oracle"] or 1) / max(1, r_stat["steps"]))
                else:
                    effs.append(0.0)
                    if r_stat["hit_mine"]: mines += 1
                    if r_stat["hit_robot"]: colls += 1
            round_efficiencies.append(sum(effs) / num_robots)
            round_mine_deaths.append(mines / num_robots)
            round_coll_deaths.append(colls / num_robots)
            round_successes.append(succs / num_robots)
            
        stats[key] = {
            "efficiency": calculate_stats(round_efficiencies),
            "mine_deaths": calculate_stats(round_mine_deaths),
            "coll_deaths": calculate_stats(round_coll_deaths),
            "success": calculate_stats(round_successes),
            "raw": {
                "efficiency": round_efficiencies,
                "mine_deaths": round_mine_deaths,
                "coll_deaths": round_coll_deaths,
                "success": round_successes
            }
        }
        
    baseline_key = config_keys[0]
    comparisons = {}
    if HAS_SCIPY and baseline_key in stats:
        for key in config_keys[1:]:
            if key not in stats: continue
            comparisons[key] = {
                "efficiency": paired_t_test(stats[baseline_key]["raw"]["efficiency"], stats[key]["raw"]["efficiency"]),
                "success": paired_t_test(stats[baseline_key]["raw"]["success"], stats[key]["raw"]["success"])
            }
            
    summary_doc = {
        "stats": {k: {mk: mv for mk, mv in v.items() if mk != "raw"} for k, v in stats.items()},
        "comparisons": comparisons
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary_doc, f, indent=4)
        
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY ({exp_dir})")
    print(f"{'Config':<8} | {'Success Rate (95% CI)':<25} | {'Efficiency (95% CI)':<25} | {'Mine D.':<10}")
    print("-" * 80)
    for key in config_keys:
        if key not in stats: continue
        s = stats[key]
        succ = s['success']
        eff = s['efficiency']
        mines = s['mine_deaths']
        
        succ_str = f"{succ['mean']:.1%} ± {succ['ci95']:.1%}"
        eff_str = f"{eff['mean']:.3f} ± {eff['ci95']:.3f}"
        
        star = ""
        if key in comparisons:
            p_succ = comparisons[key]["success"][1]
            if p_succ < 0.05: star = "*"
            
        print(f"{key:<8} | {succ_str+star:<25} | {eff_str:<25} | {mines['mean']:.1%}")
    if HAS_SCIPY:
        print(f"\n* denotes p < 0.05 vs Baseline ({baseline_key}) in paired t-test.")
    else:
        print(f"\n(scipy not installed; t-tests skipped)")
    print("="*80)

    # Convert generated JSON files to CSV for easier analysis
    print("\nConverting and consolidating logs into master CSVs...")
    convert_logs.convert_jsonl_to_csv(os.path.join(exp_dir, "events.jsonl"))
    convert_logs.convert_json_to_csv(os.path.join(exp_dir, "meta.json"))
    convert_logs.convert_json_to_csv(os.path.join(exp_dir, "summary.json"))
    
    import consolidate_logs
    consolidate_logs.main()


# ---------------------------------------------------------------------------
# Personality sweep runner
# ---------------------------------------------------------------------------

def run_personality_sweep(sweep_key: str, seed: int, num_seeds: int,
                          num_robots: int = 4, output_dir: str = "logs",
                          resume_dir: Optional[str] = None, verbose: bool = True,
                          use_gui: bool = False):
    sweep_cfg = PERSONALITY_SWEEPS[sweep_key]
    param = sweep_cfg["param"]
    values = sweep_cfg["values"]
    planner_type = sweep_cfg["planner"]
    planner_config = CONFIGS[planner_type]

    if resume_dir:
        exp_dir = resume_dir
        if not os.path.exists(exp_dir):
            raise ValueError(f"Resume directory {exp_dir} does not exist.")
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(output_dir, f"sweep_{sweep_key}_seed{seed}_{ts}")
        os.makedirs(exp_dir, exist_ok=True)
        
        meta = {
            "type": "sweep",
            "sweep_key": sweep_key,
            "param": param,
            "values": values,
            "seed": seed,
            "num_seeds": num_seeds,
            "num_robots": num_robots,
            "timestamp": ts,
        }
        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

    events_file = os.path.join(exp_dir, "events.jsonl")
    
    completed_events = set()
    if os.path.exists(events_file):
        with open(events_file, "r") as f:
            for line in f:
                evt = json.loads(line)
                if evt.get("type") == "round_result":
                    completed_events.add((str(evt["value"]), evt["round_seed"]))
    
    events_out = open(events_file, "a")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Personality Sweep: {sweep_cfg['label']} ({param})")
        print(f"  Values  : {values}")
        print(f"  Output  : {exp_dir}")
        print(f"{'='*60}\n")

    try:
        for round_idx in range(num_seeds):
            round_seed = seed + round_idx
            for val in values:
                val_str = str(val)
                if (val_str, round_seed) in completed_events:
                    continue
                
                game = Game(
                    seed=round_seed,
                    planner_name=planner_type,
                    planner_config=planner_config,
                    num_robots=num_robots,
                    max_rounds=1
                )
                
                # Override parameter
                for r in game.robots:
                    setattr(r, param, val)
                
                if use_gui:
                    pygame, screen, renderer, bg = init_gui()

                while not game.tournament_over:
                    game.tick_simulation()
                    if use_gui:
                        pump_gui(pygame, screen, renderer, bg, game)
                    
                summary = game.round_summary[0]
                
                evt = {
                    "type": "round_result",
                    "sweep_key": sweep_key,
                    "value": val_str,
                    "round_idx": round_idx,
                    "round_seed": round_seed,
                    "summary": summary,
                    "ticks": game.tick
                }
                events_out.write(json.dumps(evt) + "\n")
                events_out.flush()
                
                if verbose:
                    reached = sum(1 for v in summary.values() if v["reached"])
                    print(f"Round {round_idx+1:03d}/{num_seeds} | Val {val_str:<4s} | Seed {round_seed:5d} -> Reach:{reached}")
    finally:
        events_out.close()
        
    _generate_final_summary(exp_dir, [str(v) for v in values], num_robots)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent grid-world experiment runner.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds/rounds to run per config")
    parser.add_argument("--robots", type=int, default=4, help="Number of robots per round")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"], choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default="logs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument("--personality-sweep", choices=list(PERSONALITY_SWEEPS.keys()))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Render the experiment using Pygame GUI")

    args = parser.parse_args()

    if args.personality_sweep:
        run_personality_sweep(
            sweep_key=args.personality_sweep,
            seed=args.seed,
            num_seeds=args.num_seeds,
            num_robots=args.robots,
            output_dir=args.output_dir,
            resume_dir=args.resume,
            verbose=not args.quiet,
            use_gui=args.gui,
        )
    else:
        run_paired_comparison(
            config_keys=args.configs,
            seed=args.seed,
            num_seeds=args.num_seeds,
            num_robots=args.robots,
            output_dir=args.output_dir,
            resume_dir=args.resume,
            verbose=not args.quiet,
            use_gui=args.gui,
        )


if __name__ == "__main__":
    main()
