"""
experiments.py — Headless experiment runner for the multi-agent grid world.

Supports reproducible, paired-comparison experiments across different
planner configurations.

Experiment Configurations
=========================
  A (shortest):     Shortest-path baseline — uniform cost A*, no risk weighting.
  B (risk_aware):   Risk-aware A* — mine probability + unknown penalty + collision avoidance.
  C (conservative): Conservative risk-aware A* — large unknown-cell penalty.
  D (no_collision): Risk-aware A* with collision weight = 0.
  E (risk_aware):   Risk-aware A* with collision weight enabled (same as B, default).

Usage
=====
  python experiments.py --seed 42 --rounds 10 --configs A B C
  python experiments.py --help

Output
======
  JSON results written to logs/experiment_<configs>_seed<seed>_<timestamp>.json
  Summary table printed to terminal.

Reproducibility
===============
  For each seed, all configurations are run on the EXACT SAME environment
  by using the same seed per round.  This creates paired comparisons.

  Example with seed=42:
    Round 1: GridWorld(seed=42), positions drawn deterministically.
    Round 2: GridWorld(seed=43), etc.
    All planners get the same sequence of worlds.
"""

import argparse
import json
import os
import datetime
import collections
import copy
import random
from typing import Dict, List, Optional

from grid import GridWorld
from robot import Robot
from planner import make_planner, CONFIGS, PlannerConfig
from game import MAX_TICKS


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


# ---------------------------------------------------------------------------
# Single-round headless simulation
# ---------------------------------------------------------------------------

def run_single_round(world: GridWorld, robots: List[Robot],
                     max_ticks: int = MAX_TICKS, log: bool = False) -> Dict:
    """
    Run one round of simulation on the given world with the given robots.
    Returns a per-round result dictionary.

    Args:
        world     : freshly initialised GridWorld
        robots    : list of Robot objects already reset for this round
        max_ticks : safety timeout
        log       : if True, collect a full action log

    Returns:
        dict with keys:
          ticks, per_robot results (reached, hit_mine, hit_robot, timed_out,
          steps, waits, replans, unknown_entered, oracle_steps)
    """
    messages = []
    log_fn = (lambda m: messages.append(m)) if log else None

    import collections as _coll

    for tick in range(1, max_ticks + 1):
        active = [r for r in robots if r.alive and not r.reached]
        if not active:
            break

        # Phase 1: decide
        actions = {}
        for robot in active:
            actions[robot.id] = robot.decide_action(robots, log_fn=log_fn)

        # Phase 2: apply
        for robot in active:
            robot.apply_action(actions.get(robot.id), log_fn=log_fn)

        # Same-cell collision detection
        pos_map = _coll.defaultdict(list)
        for r in robots:
            if r.alive and not r.reached:
                pos_map[r.pos].append(r)
        for pos, bots in pos_map.items():
            if len(bots) > 1:
                for b in bots:
                    b.alive     = False
                    b.hit_robot = True

        # Last-survivor win
        alive_active = [r for r in robots if r.alive and not r.reached]
        alive_all    = [r for r in robots if r.alive]
        if len(alive_all) == 1 and len(alive_active) == 1:
            alive_active[0].reached = True
            alive_active = []

        if not alive_active:
            break

    # Mark remaining alive-not-reached as timeouts
    for r in robots:
        if r.alive and not r.reached:
            r.timed_out = True
            r.alive     = False

    per_robot = []
    for r in robots:
        per_robot.append({
            "id"                   : r.id,
            "name"                 : r.name,
            "reached"              : r.reached,
            "hit_mine"             : r.hit_mine,
            "hit_robot"            : r.hit_robot,
            "timed_out"            : r.timed_out,
            "steps"                : r.steps_taken,
            "waits"                : r.waits_taken,
            "replans"              : r.replans_taken,
            "unknown_cells_entered": r.unknown_cells_entered,
            "oracle_steps"         : r._shortest_oracle,
        })

    return {"ticks": tick, "robots": per_robot, "log": messages if log else []}


# ---------------------------------------------------------------------------
# Multi-round experiment
# ---------------------------------------------------------------------------

def run_experiment(config_key: str, seed: int, num_rounds: int,
                   num_robots: int = 4, verbose: bool = True,
                   log_rounds: bool = False) -> Dict:
    """
    Run a full experiment: num_rounds rounds on deterministically generated worlds.

    Each round uses world seed = base_seed + round_index, ensuring paired comparison
    when multiple experiments share the same base_seed.

    Args:
        config_key  : key from EXPERIMENT_CONFIGS ('A'–'E')
        seed        : base random seed
        num_rounds  : number of rounds to run
        num_robots  : number of robots per round
        verbose     : print progress to terminal
        log_rounds  : collect full action log per round (slow)

    Returns:
        Full experiment results dict.
    """
    cfg = EXPERIMENT_CONFIGS[config_key]
    planner_name   = cfg["planner"]
    planner_config = cfg["config"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment {config_key}: {cfg['name']}")
        print(f"  Planner  : {planner_name}")
        print(f"  Seed     : {seed}")
        print(f"  Rounds   : {num_rounds}")
        print(f"  Robots   : {num_robots}")
        print(f"{'='*60}")

    # Create robots with fixed personalities for the experiment
    # Personalities are seeded from the base seed for reproducibility
    random.seed(seed)
    dummy_world = GridWorld(seed=seed)  # used to init robots; replaced each round
    robots = [
        Robot(i, dummy_world, planner_name=planner_name, planner_config=planner_config)
        for i in range(num_robots)
    ]

    round_results = []

    for rnd in range(num_rounds):
        round_seed = seed + rnd
        world = GridWorld(seed=round_seed)
        world.visited = collections.defaultdict(set)

        # Update world reference for all robots
        for r in robots:
            r.world = world

        # Pick start/target positions
        passable = [
            c for c in world._all_cells()
            if c not in world.walls and c not in world.mines
        ]
        if len(passable) < num_robots * 2:
            passable = [c for c in world._all_cells() if c not in world.walls]

        random.seed(round_seed)  # ensure reproducible positions
        chosen = random.sample(passable, min(num_robots * 2, len(passable)))
        starts  = chosen[:num_robots]
        targets = chosen[num_robots:num_robots*2]

        # Safe-zone: clear mines near starts
        for s in starts:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    world.mines.discard((s[0]+dr, s[1]+dc))
        world._compute_clues()

        # Reset robots
        for i, robot in enumerate(robots):
            robot.reset_round(
                starts[i], targets[i],
                oracle_wall_set=world.walls  # [ORACLE] for efficiency metric only
            )

        if verbose:
            print(f"  Round {rnd+1}/{num_rounds}  seed={round_seed}  "
                  f"mines={world.count_mines()}  walls={world.count_walls()}", end="")

        result = run_single_round(world, robots, max_ticks=MAX_TICKS, log=log_rounds)
        result["round"] = rnd + 1
        result["round_seed"] = round_seed
        result["world_mines"] = world.count_mines()
        result["world_walls"] = world.count_walls()
        round_results.append(result)

        # Accumulate into robot stats
        for robot in robots:
            robot.record_round_result()

        if verbose:
            reached = sum(1 for rr in result["robots"] if rr["reached"])
            mine_d  = sum(1 for rr in result["robots"] if rr["hit_mine"])
            coll_d  = sum(1 for rr in result["robots"] if rr["hit_robot"])
            print(f"  → ✅{reached} 💥mine={mine_d} 💥coll={coll_d} "
                  f"ticks={result['ticks']}")

    # Compute aggregate metrics
    total_rounds_each = num_rounds
    aggregate = {}
    for robot in robots:
        total = len(robot.total_efficiency)
        aggregate[robot.name] = {
            "planner"              : planner_name,
            "speed"                : robot.speed,
            "risk_tolerance"       : round(robot.risk_tolerance, 3),
            "patience"             : round(robot.patience, 3),
            # Safety
            "survival_rate"        : robot.total_wins / max(1, total),
            "mine_death_rate"      : robot.total_mine_deaths / max(1, total),
            "collision_death_rate" : robot.total_collision_deaths / max(1, total),
            "timeout_rate"         : robot.total_timeouts / max(1, total),
            # Task performance
            "completion_rate"      : robot.total_wins / max(1, total),
            "avg_efficiency"       : (sum(robot.total_efficiency) / max(1, total)),
            # Decision behaviour
            "avg_waits"            : (sum(robot.total_wait_times) / max(1, total)),
        }

    result_doc = {
        "meta": {
            "config_key"  : config_key,
            "config_name" : cfg["name"],
            "description" : cfg["description"],
            "planner"     : planner_name,
            "seed"        : seed,
            "num_rounds"  : num_rounds,
            "num_robots"  : num_robots,
            "max_ticks"   : MAX_TICKS,
            "timestamp"   : datetime.datetime.now().isoformat(),
        },
        "aggregate": aggregate,
        "rounds"   : round_results,
    }

    if verbose:
        _print_summary(result_doc)

    return result_doc


def _print_summary(result_doc: Dict) -> None:
    """Print a readable summary table to terminal."""
    meta = result_doc["meta"]
    print(f"\n--- Summary: {meta['config_key']} — {meta['config_name']} ---")
    print(f"{'Robot':<12} {'Survival':>9} {'Mine✗':>7} {'Coll✗':>7} "
          f"{'Timeout':>8} {'Efficiency':>11} {'Avg Waits':>10}")
    print("-" * 68)
    for name, stats in result_doc["aggregate"].items():
        print(
            f"{name:<12} "
            f"{stats['survival_rate']:>8.1%} "
            f"{stats['mine_death_rate']:>7.1%} "
            f"{stats['collision_death_rate']:>7.1%} "
            f"{stats['timeout_rate']:>7.1%} "
            f"{stats['avg_efficiency']:>10.3f} "
            f"{stats['avg_waits']:>9.1f}"
        )


# ---------------------------------------------------------------------------
# Paired comparison runner
# ---------------------------------------------------------------------------

def run_paired_comparison(config_keys: List[str], seed: int, num_rounds: int,
                          num_robots: int = 4, verbose: bool = True) -> Dict:
    """
    Run multiple experiment configurations on the SAME set of worlds (paired).

    For fair comparison:
      Round 1 seed = seed+0 for ALL configs
      Round 2 seed = seed+1 for ALL configs
      ...

    This ensures each algorithm faces the same environments, making comparisons
    statistically meaningful.

    Args:
        config_keys : list of keys from EXPERIMENT_CONFIGS (e.g. ['A', 'B', 'C'])
        seed        : base random seed
        num_rounds  : number of rounds per config
        num_robots  : number of robots
        verbose     : print progress

    Returns:
        Dict mapping config_key -> experiment result.
    """
    results = {}
    for key in config_keys:
        results[key] = run_experiment(
            key, seed=seed, num_rounds=num_rounds,
            num_robots=num_robots, verbose=verbose
        )

    # Save combined results
    os.makedirs("logs", exist_ok=True)
    keys_str = "_".join(config_keys)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"experiment_{keys_str}_seed{seed}_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Paired comparison complete. Results saved → {filename}")
        _print_comparison_table(results, config_keys)

    return results


def _print_comparison_table(results: Dict, config_keys: List[str]) -> None:
    """Print a side-by-side comparison of aggregate metrics."""
    print(f"\n{'='*60}")
    print("COMPARISON TABLE (averaged across all robots and rounds)")
    print(f"{'Metric':<25}", end="")
    for key in config_keys:
        name = EXPERIMENT_CONFIGS[key]["name"][:15]
        print(f" {key}:{name:<14}", end="")
    print()
    print("-" * (25 + 20 * len(config_keys)))

    def avg_over_robots(result, metric):
        agg = result.get("aggregate", {})
        vals = [stats[metric] for stats in agg.values() if metric in stats]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = [
        ("survival_rate",        "Survival Rate"),
        ("mine_death_rate",      "Mine Death Rate"),
        ("collision_death_rate", "Collision Death Rate"),
        ("timeout_rate",         "Timeout Rate"),
        ("avg_efficiency",       "Avg Efficiency"),
        ("avg_waits",            "Avg Waits/Round"),
    ]
    for metric_key, metric_label in metrics:
        print(f"{metric_label:<25}", end="")
        for key in config_keys:
            val = avg_over_robots(results[key], metric_key)
            print(f" {val:>19.3f}", end="")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run headless multi-agent grid-world experiments."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducible experiments (default: 42)"
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of rounds per experiment configuration (default: 10)"
    )
    parser.add_argument(
        "--robots", type=int, default=4,
        help="Number of robots per round (default: 4)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A", "B", "C"],
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment configurations to run (default: A B C)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-round progress output"
    )

    args = parser.parse_args()

    print(f"\nMulti-Agent Grid World Experiment Runner")
    print(f"  Seed    : {args.seed}")
    print(f"  Rounds  : {args.rounds}")
    print(f"  Robots  : {args.robots}")
    print(f"  Configs : {args.configs}")
    print()

    print("Available experiment configurations:")
    for key, cfg in EXPERIMENT_CONFIGS.items():
        marker = "★" if key in args.configs else " "
        print(f"  {marker} [{key}] {cfg['name']}: {cfg['description'][:60]}...")
    print()

    run_paired_comparison(
        config_keys=args.configs,
        seed=args.seed,
        num_rounds=args.rounds,
        num_robots=args.robots,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
