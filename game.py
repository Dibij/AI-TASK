"""
game.py — Round and tournament management for the multi-agent grid world.

Architecture:
  - Each round creates a NEW GridWorld (new layout, mines, walls).
    This is the documented behaviour.  Shared knowledge resets each round.
  - Robots keep their personality (speed, risk_tolerance, patience) across rounds.
  - Robots reset their per-round state (position, path, stats) each round.

Two-phase tick (Phase 1: Decision, Phase 2: Resolution):
  Phase 1 — All living robots call decide_action() from the same world snapshot.
             No state is mutated.  Each robot records its intended Action.
  Phase 2 — game.py resolves all actions simultaneously:
             - Detects same-cell collisions (two robots land on the same cell)
             - Detects swap collisions (two robots swap positions in one tick)
             - Applies physics (discover mine, update position, check arrival)
             - Marks dead robots with hit_robot=True (not hit_mine)

This two-phase protocol eliminates the update-order bias of the previous
sequential approach where Robot 0 always moved first.

Seed support:
  Game(seed=N) passes N to GridWorld so experiments are reproducible.
  Pass seed=None for random behaviour.

Metrics tracked:
  SAFETY    : survival rate, mine_death_rate, collision_death_rate, timeout_rate
  TASK      : completion rate, avg ticks to goal, avg movement steps
  PATH      : shortest-path oracle length, actual steps, path efficiency
  DECISION  : avg waits, avg replans, avg unknown cells entered
  EXPERIMENT: seed, grid dims, mine count, wall count, planner type
"""

import random
import json
import os
import datetime
import collections
from typing import Optional, List, Dict

from grid import GridWorld
from robot import Robot, Action


MAX_ROUNDS  = 10
NUM_ROBOTS  = 4
MAX_TICKS   = 300  # safety timeout per round


class Game:
    """
    Manages a multi-round tournament.

    Each round:
      1. A new GridWorld is created (new random layout).
      2. Robot positions and targets are randomised.
      3. The simulation runs tick-by-tick until all robots are done or MAX_TICKS.
      4. Results are accumulated into each robot's cumulative stats.

    After MAX_ROUNDS, the tournament ends and data is exported to logs/.
    """

    def __init__(self, seed: Optional[int] = None,
                 planner_name: str = 'risk_aware',
                 num_robots: int = NUM_ROBOTS,
                 max_rounds: int = MAX_ROUNDS):
        self.seed          = seed
        self.planner_name  = planner_name
        self.num_robots    = num_robots
        self.max_rounds    = max_rounds

        # Create robots (world reference will be updated each round)
        self.world   = GridWorld(seed=seed)
        self.robots  = [
            Robot(i, self.world, planner_name=planner_name)
            for i in range(num_robots)
        ]

        # Tournament state
        self.round_number    : int  = 0
        self.tick            : int  = 0
        self.running         : bool = False
        self.round_over      : bool = False
        self.tournament_over : bool = False

        # Action log (shared with renderer, capped at 200 entries)
        self.log       : List[str] = []
        self.tick_logs : List[str] = []

        # Per-round outcome records
        self.round_summary: List[Dict] = []

        # Start first round
        self._start_round()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        self.tick_logs.append(msg)
        if len(self.log) > 200:
            self.log.pop(0)

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def _start_round(self) -> None:
        """Initialise a new round: new world, new positions, reset robot state."""
        self.round_number += 1
        self.tick        = 0
        self.running     = True
        self.round_over  = False

        # Create a fresh world each round
        # Seed is incremented so each round is deterministically different
        # but can be reproduced by replaying with the same base seed.
        round_seed = None
        if self.seed is not None:
            round_seed = self.seed + self.round_number - 1
        self.world = GridWorld(seed=round_seed)
        self.world.visited = collections.defaultdict(set)

        # Update robots' world references
        for r in self.robots:
            r.world = self.world

        self._log(f"\n--- Initialising Round {self.round_number} ---")

        # Pick starting positions and targets
        starts, targets = self._pick_positions()

        # Clear mines adjacent to start positions (safe spawn zones)
        for s in starts:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    c = (s[0] + dr, s[1] + dc)
                    if c in self.world.mines:
                        self.world.mines.discard(c)
        # Recompute clues after mine removal
        self.world._compute_clues()

        # Reset each robot ([ORACLE] passes true walls only for efficiency metric)
        for i, robot in enumerate(self.robots):
            robot.reset_round(
                starts[i], targets[i],
                oracle_wall_set=self.world.walls   # [ORACLE] — not used in AI decisions
            )

        self._log(f"━━ Round {self.round_number} / {self.max_rounds} started ━━")
        self._log(f"   World: {self.world.rows}×{self.world.cols} "
                  f"| Mines: {self.world.count_mines()} "
                  f"| Walls: {self.world.count_walls()}")
        for robot in self.robots:
            self._log(
                f"  {robot.name}: {robot.pos} → {robot.target} "
                f"(spd={robot.speed}, risk={robot.risk_tolerance:.2f}, "
                f"pat={robot.patience:.2f}, planner={self.planner_name})"
            )

    def _pick_positions(self):
        """
        Pick distinct start and target positions for all robots.
        Uses ground-truth passable cells (OK — this is placement logic, not AI).
        """
        passable = [
            c for c in self.world._all_cells()
            if c not in self.world.walls and c not in self.world.mines
        ]
        if len(passable) < self.num_robots * 2:
            passable = [c for c in self.world._all_cells()
                        if c not in self.world.walls]

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Attempt to find well-spaced starts and targets
        for attempt_limit, min_dist in [(2000, 10), (1000, 6), (200, 2)]:
            for _ in range(attempt_limit):
                cand_starts = random.sample(passable, self.num_robots)
                cand_targets = []
                valid = True
                for i in range(self.num_robots):
                    far_enough = [c for c in passable
                                  if manhattan(cand_starts[i], c) >= min_dist
                                  and c not in cand_starts]
                    if not far_enough:
                        valid = False
                        break
                    cand_targets.append(random.choice(far_enough))

                if (valid
                        and len(cand_targets) == self.num_robots
                        and len(set(cand_starts + cand_targets)) == self.num_robots * 2):
                    return cand_starts, cand_targets

        # Last resort: random placement
        chosen = random.sample(passable, self.num_robots * 2)
        return chosen[:self.num_robots], chosen[self.num_robots:]

    def _end_round(self) -> None:
        """Finalise round, record results, possibly end tournament."""
        self.running   = False
        self.round_over = True
        self._log(f"━━ Round {self.round_number} ended (tick {self.tick}) ━━")

        # Mark timed-out robots
        for robot in self.robots:
            if robot.alive and not robot.reached:
                robot.timed_out = True
                robot.alive     = False

        summary = {}
        for robot in self.robots:
            robot.record_round_result()

            # Determine display status using separate death flags
            if robot.reached:
                status = "✅ GOAL"
            elif robot.hit_mine:
                status = "💥 MINE"
            elif robot.hit_robot:
                status = "💥 COLLISION"
            elif robot.timed_out:
                status = "⏰ TIMEOUT"
            else:
                status = "❓ UNKNOWN"

            self._log(
                f"  {robot.name}: {status} | steps={robot.steps_taken} "
                f"| waits={robot.waits_taken} | replans={robot.replans_taken}"
            )
            summary[robot.id] = {
                "name"                : robot.name,
                "reached"             : robot.reached,
                "hit_mine"            : robot.hit_mine,
                "hit_robot"           : robot.hit_robot,
                "timed_out"           : robot.timed_out,
                "steps"               : robot.steps_taken,
                "waits"               : robot.waits_taken,
                "replans"             : robot.replans_taken,
                "unknown_cells_entered": robot.unknown_cells_entered,
                "shortest_oracle"     : robot._shortest_oracle,
            }
        self.round_summary.append(summary)

        if self.round_number >= self.max_rounds:
            self.tournament_over = True
            self._announce_winner()

    def _announce_winner(self) -> None:
        self._log("━━━━ TOURNAMENT OVER ━━━━")
        ranked = sorted(self.robots, key=lambda r: self.avg_efficiency(r), reverse=True)
        for rank, robot in enumerate(ranked, 1):
            eff  = self.avg_efficiency(robot)
            surv = self.survival_rate(robot)
            self._log(
                f"  #{rank} {robot.name}: eff={eff:.2f} | surv={surv:.0%} "
                f"| wins={robot.total_wins} | mine_deaths={robot.total_mine_deaths} "
                f"| collision_deaths={robot.total_collision_deaths}"
            )
        self._export_data()

    # ------------------------------------------------------------------
    # Simulation tick — two-phase protocol
    # ------------------------------------------------------------------

    def tick_simulation(self) -> None:
        """
        Advance simulation by one tick using the two-phase protocol.

        Phase 1 — Decision:
            All living robots call decide_action() with the SAME current state.
            No robot state is mutated during Phase 1.
            This eliminates update-order bias (Robot 0 no longer has priority).

        Phase 2 — Resolution:
            Collect all intended Actions.
            Detect and resolve collisions:
              - Same-cell collisions: two robots land on the same cell.
              - Swap collisions: two robots swap positions in one tick.
            Apply each robot's action via apply_action().
            Apply collision deaths AFTER movement.
        """
        self.tick_logs = []
        if not self.running:
            return

        self.tick += 1

        active = [r for r in self.robots if r.alive and not r.reached]

        # ── Phase 1: Decision (no state mutation) ──
        actions = {}
        for robot in active:
            actions[robot.id] = robot.decide_action(self.robots, log_fn=self._log)

        # ── Phase 2: Resolution ──
        # Apply each robot's action (movement + mine discovery + arrival)
        for robot in active:
            action = actions.get(robot.id, Action(Action.WAIT))
            robot.apply_action(action, log_fn=self._log)

        # ── Collision Detection (after all moves) ──
        self._resolve_collisions()

        # ── Swap Collision Detection ──
        self._resolve_swap_collisions(active, actions)

        # ── Round-end check ──
        self._check_round_over()

    def _resolve_collisions(self) -> None:
        """
        Detect same-cell collisions: multiple living robots at the same position.
        These are marked as hit_robot=True (not hit_mine).
        """
        pos_map = collections.defaultdict(list)
        for robot in self.robots:
            if robot.alive and not robot.reached:
                pos_map[robot.pos].append(robot)

        for pos, bots in pos_map.items():
            if len(bots) > 1:
                names = [b.name for b in bots]
                self._log(
                    f"💥 COLLISION at {pos}! "
                    f"{', '.join(names)} collided!"
                )
                for b in bots:
                    b.alive     = False
                    b.hit_robot = True  # correctly typed as collision, not mine

    def _resolve_swap_collisions(self, active_before, actions) -> None:
        """
        Detect swap collisions: robots A and B exchanging positions in one tick.
        (A moves A_pos→B_pos and B moves B_pos→A_pos simultaneously.)
        """
        # Map robot_id -> (old_pos, new_pos)
        moves = {}
        for robot in active_before:
            action = actions.get(robot.id)
            if action and action.kind == Action.MOVE and action.cells:
                moves[robot.id] = (robot.pos, action.cells[-1])
            # Note: robot.pos may have been updated by apply_action already
            # Use the first cell in action to detect the initial swap

        # Check each pair for swap
        ids = list(moves.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                old_a, new_a = moves[id_a]
                old_b, new_b = moves[id_b]
                # Swap: A went to B's old pos AND B went to A's old pos
                if new_a == old_b and new_b == old_a:
                    robot_a = next(r for r in self.robots if r.id == id_a)
                    robot_b = next(r for r in self.robots if r.id == id_b)
                    if robot_a.alive and robot_b.alive:
                        self._log(
                            f"💥 SWAP COLLISION! {robot_a.name} ↔ {robot_b.name} "
                            f"crossed paths between {old_a} and {old_b}!"
                        )
                        robot_a.alive     = False
                        robot_a.hit_robot = True
                        robot_b.alive     = False
                        robot_b.hit_robot = True

    def _check_round_over(self) -> None:
        """Check if all robots are done this tick."""
        active = [r for r in self.robots if r.alive and not r.reached]

        # Last-survivor default win
        alive_total = [r for r in self.robots if r.alive]
        if len(alive_total) == 1 and len(active) == 1:
            lone = active[0]
            lone.reached = True
            self._log(f"🏆 {lone.name} wins by default as last survivor!")
            active = []

        all_done  = len(active) == 0
        timed_out = self.tick >= MAX_TICKS

        if all_done or timed_out:
            self._end_round()

    # ------------------------------------------------------------------
    # Round navigation
    # ------------------------------------------------------------------

    def next_round(self) -> None:
        """Advance to the next round (called from renderer on SPACE or auto-advance)."""
        if self.round_over and not self.tournament_over:
            self._start_round()

    def reset(self) -> None:
        """Full reset: new world, same robots, round 1."""
        self.__init__(seed=self.seed, planner_name=self.planner_name,
                      num_robots=self.num_robots, max_rounds=self.max_rounds)

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def _export_data(self) -> None:
        """Write full simulation results to logs/ as JSON."""
        data = {
            "meta": {
                "seed"        : self.seed,
                "planner"     : self.planner_name,
                "rounds"      : self.max_rounds,
                "num_robots"  : self.num_robots,
                "max_ticks"   : MAX_TICKS,
                "timestamp"   : datetime.datetime.now().isoformat(),
            },
            "robots": [],
            "round_history": self.round_summary,
            "full_log": self.log,
        }

        for r in self.robots:
            total_rounds = len(r.total_efficiency)
            data["robots"].append({
                "id"                    : r.id,
                "name"                  : r.name,
                "speed"                 : r.speed,
                "risk_tolerance"        : r.risk_tolerance,
                "patience"              : r.patience,
                "planner"               : self.planner_name,
                # Safety metrics
                "total_wins"            : r.total_wins,
                "total_mine_deaths"     : r.total_mine_deaths,
                "total_collision_deaths": r.total_collision_deaths,
                "total_timeouts"        : r.total_timeouts,
                "survival_rate"         : self.survival_rate(r),
                "mine_death_rate"       : r.total_mine_deaths / max(1, total_rounds),
                "collision_death_rate"  : r.total_collision_deaths / max(1, total_rounds),
                "timeout_rate"          : r.total_timeouts / max(1, total_rounds),
                # Task performance
                "completion_rate"       : r.total_wins / max(1, total_rounds),
                "avg_efficiency"        : self.avg_efficiency(r),
                # Decision behaviour
                "avg_wait_per_round"    : self.avg_wait(r),
                "total_wait_rounds"     : r.total_wait_times,
            })

        os.makedirs("logs", exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("logs", f"tournament_{self.planner_name}_{ts}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        self._log(f"Data exported → {filename}")

    # ------------------------------------------------------------------
    # Stats helpers (used by renderer)
    # ------------------------------------------------------------------

    def avg_efficiency(self, robot: Robot) -> float:
        if not robot.total_efficiency:
            return 0.0
        return sum(robot.total_efficiency) / len(robot.total_efficiency)

    def survival_rate(self, robot: Robot) -> float:
        total = len(robot.total_efficiency)
        if total == 0:
            return 0.0
        return robot.total_wins / total  # fraction of rounds completed (not died)

    def avg_wait(self, robot: Robot) -> float:
        if not robot.total_wait_times:
            return 0.0
        return sum(robot.total_wait_times) / len(robot.total_wait_times)

    def leaderboard(self) -> List[Robot]:
        """Return robots sorted by average efficiency (descending)."""
        return sorted(self.robots, key=lambda r: self.avg_efficiency(r), reverse=True)
