"""
experiments.py — Headless experiment runner for the multi-agent grid world.

Supports reproducible, paired-comparison experiments across different
planner configurations and personality sweeps.

Experiment Configurations
=========================
  A (shortest):     Shortest-path baseline — uniform cost A*, no risk weighting.
  B (risk_aware):   Risk-aware A* — mine probability + unknown penalty + collision avoidance.
  C (conservative): Conservative risk-aware A* — large unknown-cell penalty.
  D (no_collision): Risk-aware A* with collision weight = 0.
  E (risk_aware):   Risk-aware A* with collision weight enabled (same as B, default).

Personality Sweeps
==================
  --personality-sweep risk_tolerance  Varies risk tolerance (0.1–0.9)
  --personality-sweep patience        Varies patience (0.1–0.9)
  --personality-sweep speed           Varies speed (1–2)

  Sweeps control one parameter while keeping others fixed, answering:
    - Does higher risk tolerance increase speed but reduce survival?
    - Does higher patience reduce collisions?
    - Does higher speed improve completion time?

Usage
=====
  python experiments.py --seed 42 --rounds 10 --configs A B C
  python experiments.py --personality-sweep risk_tolerance --seed 42
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
# Personality sweep configurations
# ---------------------------------------------------------------------------

# Each sweep varies one robot personality parameter while keeping others fixed.
# This allows measuring the causal effect of each trait on outcomes.
PERSONALITY_SWEEPS: Dict[str, Dict] = {
    "risk_tolerance": {
        "param"     : "risk_tolerance",
        "label"     : "Risk Tolerance",
        "values"    : [0.1, 0.3, 0.5, 0.7, 0.9],
        "planner"   : "risk_aware",
        "questions" : [
            "Does higher risk tolerance increase completion speed but reduce survival?",
            "What tolerance maximises efficiency in moderate-mine environments?",
        ],
    },
    "patience": {
        "param"     : "patience",
        "label"     : "Patience",
        "values"    : [0.1, 0.3, 0.5, 0.7, 0.9],
        "planner"   : "risk_aware",
        "questions" : [
            "Does higher patience increase waiting behaviour but reduce collisions?",
            "What patience level minimises timeout rate?",
        ],
    },
    "speed": {
        "param"     : "speed",
        "label"     : "Speed",
        "values"    : [1, 2],
        "planner"   : "risk_aware",
        "questions" : [
            "Does higher speed reduce ticks-to-goal?",
            "Does higher speed increase mine death rate (less cautious movement)?",
        ],
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
    random.seed(seed)
    dummy_world = GridWorld(seed=seed)
    robots = [
        Robot(i, dummy_world, planner_name=planner_name, planner_config=planner_config)
        for i in range(num_robots)
    ]

    round_results = []

    for rnd in range(num_rounds):
        round_seed = seed + rnd
        world = GridWorld(seed=round_seed)
        world.visited = collections.defaultdict(set)

        for r in robots:
            r.world = world

        passable = [
            c for c in world._all_cells()
            if c not in world.walls and c not in world.mines
        ]
        if len(passable) < num_robots * 2:
            passable = [c for c in world._all_cells() if c not in world.walls]

        random.seed(round_seed)
        chosen = random.sample(passable, min(num_robots * 2, len(passable)))
        starts  = chosen[:num_robots]
        targets = chosen[num_robots:num_robots*2]

        for s in starts:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    world.mines.discard((s[0]+dr, s[1]+dc))
        world._compute_clues()

        for i, robot in enumerate(robots):
            robot.reset_round(
                starts[i], targets[i],
                oracle_wall_set=world.walls
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

        for robot in robots:
            robot.record_round_result()

        if verbose:
            reached = sum(1 for rr in result["robots"] if rr["reached"])
            mine_d  = sum(1 for rr in result["robots"] if rr["hit_mine"])
            coll_d  = sum(1 for rr in result["robots"] if rr["hit_robot"])
            print(f"  -> {reached} reached, {mine_d} mine, {coll_d} coll, "
                  f"ticks={result['ticks']}")

    aggregate = {}
    for robot in robots:
        total = len(robot.total_efficiency)
        aggregate[robot.name] = {
            "planner"              : planner_name,
            "speed"                : robot.speed,
            "risk_tolerance"       : round(robot.risk_tolerance, 3),
            "patience"             : round(robot.patience, 3),
            "survival_rate"        : robot.total_wins / max(1, total),
            "mine_death_rate"      : robot.total_mine_deaths / max(1, total),
            "collision_death_rate" : robot.total_collision_deaths / max(1, total),
            "timeout_rate"         : robot.total_timeouts / max(1, total),
            "completion_rate"      : robot.total_wins / max(1, total),
            "avg_efficiency"       : (sum(robot.total_efficiency) / max(1, total)),
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
    print(f"\n--- Summary: {meta['config_key']} - {meta['config_name']} ---")
    print(f"{'Robot':<12} {'Survival':>9} {'MineX':>7} {'CollX':>7} "
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

    os.makedirs("logs", exist_ok=True)
    keys_str = "_".join(config_keys)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"experiment_{keys_str}_seed{seed}_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Paired comparison complete. Results saved -> {filename}")
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
# Personality sweep runner
# ---------------------------------------------------------------------------

def run_personality_sweep(sweep_key: str, seed: int, num_rounds: int,
                          num_robots: int = 4, planner_type: str = "risk_aware",
                          verbose: bool = True) -> Dict[str, Dict]:
    """
    Run a controlled personality sweep: vary one parameter, keep others fixed.

    All robots in a run share the same personality value for the swept parameter,
    ensuring clean attribution of effect.  Robots are the same population;
    only the swept parameter differs across runs.

    Args:
        sweep_key   : key from PERSONALITY_SWEEPS
        seed        : base random seed
        num_rounds  : number of rounds per value
        num_robots  : number of robots per round
        planner_type: planner configuration for all runs
        verbose     : print progress

    Returns:
        Dict mapping value (e.g. "0.1") -> experiment result for that level.
    """
    sweep_cfg = PERSONALITY_SWEEPS[sweep_key]
    param     = sweep_cfg["param"]
    values    = sweep_cfg["values"]
    label     = sweep_cfg["label"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Personality Sweep: {label} ({param})")
        print(f"  Values  : {values}")
        print(f"  Seed    : {seed}")
        print(f"  Rounds  : {num_rounds}")
        print(f"  Planner : {planner_type}")
        print(f"  Questions:")
        for q in sweep_cfg["questions"]:
            print(f"    - {q}")
        print(f"{'='*60}")

    results: Dict[str, Dict] = {}

    for val in values:
        if verbose:
            print(f"\n  --- {label} = {val} ---")

        # All robots get the same fixed personality for this sweep value
        random.seed(seed)
        dummy_world = GridWorld(seed=seed)
        robots = [
            Robot(i, dummy_world, planner_name=planner_type)
            for i in range(num_robots)
        ]

        # Override the swept parameter for all robots
        for r in robots:
            setattr(r, param, val)

        # Run rounds
        round_results = []
        for rnd in range(num_rounds):
            round_seed = seed + rnd
            world = GridWorld(seed=round_seed)
            world.visited = collections.defaultdict(set)

            for r in robots:
                r.world = world

            passable = [
                c for c in world._all_cells()
                if c not in world.walls and c not in world.mines
            ]
            if len(passable) < num_robots * 2:
                passable = [c for c in world._all_cells() if c not in world.walls]

            random.seed(round_seed)
            chosen = random.sample(passable, min(num_robots * 2, len(passable)))
            starts  = chosen[:num_robots]
            targets = chosen[num_robots:num_robots*2]

            for s in starts:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        world.mines.discard((s[0]+dr, s[1]+dc))
            world._compute_clues()

            for i, robot in enumerate(robots):
                robot.reset_round(
                    starts[i], targets[i],
                    oracle_wall_set=world.walls
                )

            result = run_single_round(world, robots, max_ticks=MAX_TICKS, log=False)
            result["round"] = rnd + 1
            result["round_seed"] = round_seed
            result["world_mines"] = world.count_mines()
            result["world_walls"] = world.count_walls()
            round_results.append(result)

            for robot in robots:
                robot.record_round_result()

        # Compute aggregate
        aggregate = {}
        for robot in robots:
            total = len(robot.total_efficiency)
            aggregate[robot.name] = {
                param                : val,
                "survival_rate"      : robot.total_wins / max(1, total),
                "mine_death_rate"    : robot.total_mine_deaths / max(1, total),
                "collision_death_rate": robot.total_collision_deaths / max(1, total),
                "timeout_rate"       : robot.total_timeouts / max(1, total),
                "completion_rate"    : robot.total_wins / max(1, total),
                "avg_efficiency"     : (sum(robot.total_efficiency) / max(1, total)),
                "avg_waits"          : (sum(robot.total_wait_times) / max(1, total)),
            }

        results[str(val)] = {
            "meta": {
                "sweep"       : sweep_key,
                "param"       : param,
                "value"       : val,
                "planner"     : planner_type,
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
            avg_survival = sum(stats["survival_rate"] for stats in aggregate.values()) / len(aggregate)
            avg_eff      = sum(stats["avg_efficiency"] for stats in aggregate.values()) / len(aggregate)
            print(f"  -> survival={avg_survival:.1%}, efficiency={avg_eff:.3f}")

    # Save sweep results
    os.makedirs("logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"sweep_{sweep_key}_seed{seed}_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    if verbose:
        print(f"\n  Sweep results saved -> {filename}")
        _print_sweep_summary(results, label, param, verbose=True)

    return results


def _print_sweep_summary(results: Dict[str, Dict], label: str,
                         param: str, verbose: bool = False) -> None:
    """Print a summary of a personality sweep across all values."""
    values_sorted = sorted(results.keys(), key=float)
    print(f"\n{'='*60}")
    print(f"Personality Sweep: {label}")
    print(f"{'Value':<10} {'Survival':>9} {'MineX':>7} {'CollX':>7} "
          f"{'Timeout':>8} {'Efficiency':>11} {'Avg Waits':>10}")
    print("-" * 68)

    for val_str in values_sorted:
        r   = results[val_str]
        agg = r["aggregate"]
        avg_survival = sum(s["survival_rate"] for s in agg.values()) / max(1, len(agg))
        avg_mine     = sum(s["mine_death_rate"] for s in agg.values()) / max(1, len(agg))
        avg_coll     = sum(s["collision_death_rate"] for s in agg.values()) / max(1, len(agg))
        avg_timeout  = sum(s["timeout_rate"] for s in agg.values()) / max(1, len(agg))
        avg_eff      = sum(s["avg_efficiency"] for s in agg.values()) / max(1, len(agg))
        avg_waits    = sum(s["avg_waits"] for s in agg.values()) / max(1, len(agg))

        print(
            f"{val_str:<10} "
            f"{avg_survival:>8.1%} "
            f"{avg_mine:>7.1%} "
            f"{avg_coll:>7.1%} "
            f"{avg_timeout:>7.1%} "
            f"{avg_eff:>10.3f} "
            f"{avg_waits:>9.1f}"
        )


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
        "--personality-sweep",
        choices=list(PERSONALITY_SWEEPS.keys()),
        help="Run a personality sweep instead of config comparison"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-round progress output"
    )

    args = parser.parse_args()

    if args.personality_sweep:
        run_personality_sweep(
            sweep_key=args.personality_sweep,
            seed=args.seed,
            num_rounds=args.rounds,
            num_robots=args.robots,
            verbose=not args.quiet,
        )
        return

    print(f"\nMulti-Agent Grid World Experiment Runner")
    print(f"  Seed    : {args.seed}")
    print(f"  Rounds  : {args.rounds}")
    print(f"  Robots  : {args.robots}")
    print(f"  Configs : {args.configs}")
    print()

    print("Available experiment configurations:")
    for key, cfg in EXPERIMENT_CONFIGS.items():
        marker = "*" if key in args.configs else " "
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
