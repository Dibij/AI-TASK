# Multi-Agent Grid-World Pathfinding with Risk-Aware A*

A risk-aware multi-agent pathfinding system for a partially observable grid world. The system combines A* search, explicit knowledge representation, local probabilistic reasoning about hidden mines, and short-horizon multi-agent collision avoidance.

## Overview

Four robots navigate a 15×15 grid world with hidden mines and walls. Each robot can only see what it (or its teammates) have discovered. The goal is to reach the assigned target cell while avoiding mines, collisions with other robots, and walls.

**Key characteristics:**
- **Partially observable**: robots only know what they've physically discovered
- **Cooperative knowledge sharing**: all robots share one knowledge state (modelling a radio-linked swarm)
- **Risk-aware pathfinding**: A* weighted by mine probability, unknown-cell penalty, and collision prediction
- **Five-phase tick**: strict phase separation (Observe, Update, Decide, Resolve, Apply) ensures no update-order bias
- **Short-horizon deterministic trajectory prediction**: planned-path-based collision avoidance (not probabilistic)
- **Priority-based conflict resolution**: collisions force waiters based on wait-time priority to prevent starvation

## Architecture

```
game.py          Tournament management, two-phase tick, round lifecycle
grid.py          Ground-truth world generation, physics/discovery engine
robot.py         Robot agent: decision loop, personality traits, stats
knowledge.py     AgentKnowledge: explicit belief state, probability, constraint propagation
planner.py       Configurable planners: ShortestPath / RiskAware / Conservative
utils.py         A* pathfinding, BFS helpers, shortest-path oracle
experiments.py   Headless experiment runner (paired comparison + personality sweeps)
main.py          Pygame renderer with GUI controls
tests/           Pytest test suite (85+ tests)
```

### Ground Truth vs. Agent Knowledge

The system enforces a strict separation between:

| Component | Accesses | Used for |
|-----------|----------|----------|
| `GridWorld.walls` | Physics engine, oracle metrics | Determining if a step is blocked; computing shortest-path oracle |
| `GridWorld.mines` | Physics engine | Determining if stepping on a cell kills the robot |
| `AgentKnowledge` | Robot decision logic, planners, probability model | ALL AI reasoning |
| `GridWorld.discover()` | Physics engine | Bridges truth → knowledge when robot enters a cell |

Robot decision logic, planners, and the probability model **never** access `GridWorld.walls` or `GridWorld.mines` directly. They only consult `AgentKnowledge`.

## Knowledge Representation

Every cell in the agent's world model has one of these statuses:

| Status | Meaning |
|--------|---------|
| `UNKNOWN` | Robot has never seen this cell |
| `OBSERVED_SAFE` | Robot physically entered the cell; it was empty |
| `OBSERVED_WALL` | Adjacent scan revealed this cell is a wall |
| `OBSERVED_MINE` | Robot stepped on a mine here (and died) |
| `INFERRED_SAFE` | Logical constraint propagation proves no mine here |
| `INFERRED_MINE` | Logical constraint propagation proves mine here |

The probability model estimates P(mine | knowledge) using local clue constraints from observed Minesweeper-style adjacency numbers. This is a **local heuristic approximation**, not full Bayesian inference or a complete Minesweeper solver.

## Planners

Three planners are available for experimental comparison:

| Planner | Cost function | Purpose |
|---------|---------------|---------|
| **ShortestPath** (baseline) | Uniform cost=1 per cell | Control condition — ignores all risk |
| **RiskAware** (default) | movement + mine_risk*P(mine) + unknown_penalty + collision_penalty | Main algorithm |
| **Conservative** | Like RiskAware but 10x unknown penalty | Tests hypothesis: avoid unknowns → fewer deaths |

All planners use the same A* implementation from `utils.py`. The cost function and blocked-cell predicate are parameterised per planner.

### Cost Model (RiskAwarePlanner)

```
total_cost = movement_cost
           + mine_risk_weight * P(mine | knowledge)
           + collision_weight * is_predicted_occupied(cell, step=1)
           + unknown_weight   * is_unknown(cell)
           + avoid_weight     * is_in_avoid_set(cell)
```

**Weights are manually chosen heuristics, not mathematically optimal values.** Their effect should be measured experimentally (see experiments below).

| Weight | Default | Interpretation |
|--------|---------|----------------|
| movement_cost | 1.0 | Base step cost |
| mine_risk_weight | 3.0 | Cost per unit of mine probability |
| unknown_weight | 0.2 (2.0 for Conservative) | Flat penalty for entering unknown cells |
| collision_weight | 0.5 | Penalty for stepping into a predicted-occupied cell |
| avoid_weight | 100.0 | Large penalty for cells the robot is actively avoiding |

### Collision Prediction

`predict_others()` implements **short-horizon deterministic trajectory prediction** — it simulates other robots advancing along their declared planned paths for a configurable horizon (default 5 steps). This is NOT probabilistic collision prediction.

## Robot Personalities

Each robot has three personality traits that influence behaviour:

| Trait | Range | Effect |
|-------|-------|--------|
| `speed` | 1 (slow) or 2 (fast) | Cells moved per tick |
| `risk_tolerance` | 0.1–0.9 | Threshold for entering risky cells |
| `patience` | 0.1–0.9 | Wait duration before replanning when stuck |

In the default `Game`, these are randomised. Controlled personality sweeps are available via `experiments.py`.

## Five-Phase Tick Protocol

Each simulation tick follows a strict five-phase protocol that eliminates update-order bias and prevents starvation:

**Phase 1 — Observe:** All robots simultaneously read the GridWorld and accumulate observations (mines, walls, clues). No inferences are made yet.
**Phase 2 — Update Knowledge:** The shared `AgentKnowledge` processes all observations globally, propagating constraints and updating the probability model.
**Phase 3 — Decide:** Every alive, unreached robot calls `decide_action()` on the synchronized knowledge state. No state is mutated. Wait-forcing risk-rejection happens here.
**Phase 4 — Resolve Conflicts:** System detects collisions (same-cell and swap). A deterministic priority system (based on consecutive waits + starvation boost) decides the winner. Losers are forced to change their action to `WAIT`.
**Phase 5 — Apply:** Actions (Move/Wait) are applied simultaneously. Risk relaxation triggers here if a robot is stuck avoiding a cell and runs out of options.

## Death Types

The system tracks two separate death types (collisions no longer cause death, they cause waits):

| Flag | Cause |
|------|-------|
| `hit_mine` | Robot stepped on a hidden mine |
| `timed_out` | Robot didn't reach target within MAX_TICKS (300) per round |

## Metrics

### Safety
- Survival rate
- Mine death rate
- Collision death rate
- Timeout rate

### Task Performance
- Goal completion rate
- Average ticks to goal
- Path efficiency (oracle-shortest-path / actual steps)

### Decision Behaviour
- Number of waits per round
- Number of replans per round
- Number of unknown cells entered

### Environment
- Actual mine count (for context, not AI decision-making)
- Wall count
- Grid dimensions

## Experiments

### Configuration Comparison

```bash
python experiments.py --seed 42 --num-seeds 10 --configs A B C --output-dir logs/
```

This runs a **paired comparison**: all configurations face the exact same sequence of worlds (same seeds). 
If interrupted, you can resume by passing `--resume logs/experiment_A_B_C_seed42_...`.

| Config | Name | Description |
|--------|------|-------------|
| A | Shortest-Path Baseline | Uniform cost A*, no risk weighting |
| B | Risk-Aware A* | Composite cost with mine probability + unknown penalty + collision avoidance |
| C | Conservative Risk-Aware A* | Like B but 10x unknown penalty |
| D | Risk-Aware A* (No Collision) | Like B but collision_weight=0 |
| E | Risk-Aware A* (Full) | Same as B (collision enabled) |

### Personality Sweeps

```
python experiments.py --personality-sweep risk_tolerance
python experiments.py --personality-sweep patience
python experiments.py --personality-sweep speed
```

Each sweep varies one parameter while keeping others fixed, measuring the causal effect on outcomes. Designed to answer hypotheses such as:
- Does higher risk tolerance increase speed but reduce survival?
- Does higher patience reduce collisions?
- Does faster speed reduce ticks but increase mine deaths?

## Usage

### GUI Mode
```
python main.py
```

**Controls:**
- `SPACE` — toggle play/pause
- `R` — full reset (new tournament)
- `LEFT/RIGHT` — step through history (paused)
- `+/-` — speed up/slow down
- `ESC` — quit
- Mouse hover — robot tooltip
- Scroll wheel — log scrolling

### Headless Experiments
```bash
python experiments.py --seed 42 --num-seeds 30 --configs A B C D E
python experiments.py --personality-sweep risk_tolerance --seed 42 --num-seeds 10
python experiments.py --resume logs/experiment_A_B_C_D_E_seed42_...
python experiments.py --gui  # Watch the experiment run at maximum speed
python experiments.py --help
```

### Experiment Output Format
Experiments write output to a uniquely timestamped directory (e.g. `logs/experiment_A_B_C_seed42_.../`) containing:
- `meta.json`: Experiment configuration parameters.
- `events.jsonl`: Event-level JSON Lines format. Each line represents one completed round per configuration, enabling crash recovery and incremental data processing.
- `summary.json`: Final statistical summary including Mean, Median, StdDev, 95% Confidence Intervals, and Paired T-Test p-values against the baseline.

### Pre-Experiment Checklist
Before kicking off a massive overnight sweep, verify the following:
1. **Tests pass:** Run `pytest tests/test_suite.py` to ensure core systems are sound.
2. **Crash recovery:** Verify the `--resume` flag works by cancelling a short run and resuming it.
3. **Statistical significance:** Set `--num-seeds` to a statistically meaningful value (30-50 recommended) for paired comparisons.
4. **Scipy installed:** Ensure `scipy` is installed in your environment for T-Tests and CIs.

### Tests
```bash
python -m pytest tests/test_suite.py -v
```

## Limitations

The following are explicitly documented limitations (not bugs):

1. **Probability model**: Uses local clue-ratio heuristics, not full Bayesian inference or a complete Minesweeper solver. The `propagate_constraints()` method handles simple subset deductions but does not enumerate all satisfying mine assignments.

2. **Collision prediction**: Deterministic trajectory extrapolation from declared paths. Robots may replan, making predictions stale. This is not probabilistic collision prediction.

3. **Weight tuning**: Path cost weights are manually chosen heuristics, not mathematically optimal values. Their effect should be measured rather than assumed.

4. **Cooperative knowledge**: All robots share one `AgentKnowledge` instance. This models a radio-linked swarm. If independent per-robot knowledge is desired, each robot would need its own instance.

5. **Mine death is permanent**: Robots do not respawn mid-round. A mine death is final for that round.

6. **No machine learning**: The system uses explainable A* search, heuristic costs, and logical constraint propagation. There are no trained models.

7. **Grid constraints**: The grid is always 15×15. The generation algorithm ensures full connectivity (all passable cells reachable) but may create challenging maze-like corridors.

## Requirements

- Python 3.10+
- pygame 2.6.1 (for GUI mode)
- pytest (for tests)

## File Structure

```
.
├── main.py              Pygame renderer and game loop
├── game.py              Tournament management, two-phase tick
├── grid.py              GridWorld: ground truth + physics engine
├── robot.py             Robot agent: decision, personality, stats
├── knowledge.py         AgentKnowledge: belief state, probability model
├── planner.py           Configurable A* strategies (3 variants)
├── utils.py             A* pathfinding, BFS, shortest-path oracle
├── experiments.py       Headless experiment runner with personality sweeps
├── Readme.md            This file
├── requirements.txt     Python dependencies
├── assets/              Robot sprite images
├── logs/                Experiment output (JSON)
└── tests/
    ├── __init__.py
    └── test_suite.py    85+ pytest tests covering all modules
