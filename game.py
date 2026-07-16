"""
game.py — Round and tournament management.

Responsibilities:
  - One fixed GridWorld shared across all 10 rounds.
  - Per-round: randomise start/target positions, reset robot state.
  - Tick simulation until all robots finish or round times out.
  - Accumulate per-robot stats and produce final leaderboard.
"""

import random
import json
import os
import datetime
from grid import GridWorld
from robot import Robot


MAX_ROUNDS     = 10
NUM_ROBOTS     = 4
MAX_TICKS      = 300  # safety limit per round


class GameState:
    """Holds all mutable simulation state accessible by main.py renderer."""
    pass


class Game:
    def __init__(self):
        self.world   = GridWorld()
        self.robots  = [Robot(i, self.world) for i in range(NUM_ROBOTS)]

        self.round_number  = 0
        self.tick          = 0
        self.running       = False   # True while a round is in progress
        self.round_over    = False
        self.tournament_over = False

        # Action log — shared with renderer
        self.log = []
        self.tick_logs = []

        # Per-round outcome for display
        self.round_summary = []   # list of dicts after each round

        # Start first round
        self._start_round()

    def _log(self, msg):
        self.log.append(msg)
        self.tick_logs.append(msg)
        if len(self.log) > 200:
            self.log.pop(0)

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def _start_round(self):
        import collections
        self.round_number += 1
        self.tick         = 0
        self.running      = True
        self.round_over   = False
        
        # Create a completely new grid and update robots
        self.world = GridWorld()
        self.world.visited = collections.defaultdict(set)
        for r in self.robots:
            r.world = self.world

        self._log(f"\n--- Initialising Round {self.round_number} ---")
        # Pick distinct start and target positions for all robots
        # Exclude mine cells (robots spawn safe)
        num_positions = NUM_ROBOTS * 2
        passable = [c for c in self.world._all_cells() if c not in self.world.walls and c not in self.world.mines]
        
        if len(passable) < num_positions:
            passable = [c for c in self.world._all_cells() if c not in self.world.walls]

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        starts = []
        targets = []
        
        # Try to find starts and targets satisfying the conditions
        for _ in range(2000):
            cand_starts = random.sample(passable, NUM_ROBOTS)
            cand_targets = []
            valid = True
            for i in range(NUM_ROBOTS):
                next_start = cand_starts[(i + 1) % NUM_ROBOTS]
                # Target in proximity of next robot's start (dist <= 2)
                near = [c for c in passable if manhattan(c, next_start) <= 2]
                # Target at least 10 steps away from this robot's start
                far_enough = [c for c in near if manhattan(cand_starts[i], c) >= 10]
                
                if not far_enough:
                    valid = False
                    break
                cand_targets.append(random.choice(far_enough))
                
            if valid and len(set(cand_starts + cand_targets)) == NUM_ROBOTS * 2:
                starts = cand_starts
                targets = cand_targets
                break
                
        if not starts:
            # Fallback if too strict
            for _ in range(1000):
                cand_starts = random.sample(passable, NUM_ROBOTS)
                cand_targets = []
                valid = True
                for i in range(NUM_ROBOTS):
                    next_start = cand_starts[(i + 1) % NUM_ROBOTS]
                    near = [c for c in passable if manhattan(c, next_start) <= 3]
                    far_enough = [c for c in near if manhattan(cand_starts[i], c) >= 6]
                    if not far_enough:
                        far_enough = near
                    if not far_enough:
                        valid = False
                        break
                    cand_targets.append(random.choice(far_enough))
                    
                if valid and len(set(cand_starts + cand_targets)) == NUM_ROBOTS * 2:
                    starts = cand_starts
                    targets = cand_targets
                    break
                    
        if not starts:
            chosen = random.sample(passable, NUM_ROBOTS * 2)
            starts = chosen[:NUM_ROBOTS]
            targets = chosen[NUM_ROBOTS:]
        
        # CLEAR MINES around starts to prevent unavoidable early deaths (safe zones)
        for s in starts:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    c = (s[0] + dr, s[1] + dc)
                    if c in self.world.mines:
                        self.world.mines.remove(c)
        # Recompute clues since mines were removed
        self.world._compute_clues()

        for i, robot in enumerate(self.robots):
            robot.reset_round(starts[i], targets[i])

        self._log(f"━━ Round {self.round_number} / {MAX_ROUNDS} started ━━")
        for robot in self.robots:
            self._log(
                f"  {robot.name}: {robot.pos} → {robot.target} "
                f"(spd={robot.speed}, risk={robot.risk_tolerance:.2f}, "
                f"pat={robot.patience:.2f})"
            )

    def _end_round(self):
        self.running    = False
        self.round_over = True
        self._log(f"━━ Round {self.round_number} ended (tick {self.tick}) ━━")

        summary = {}
        for robot in self.robots:
            robot.record_round_result()
            status = "✅ WON" if robot.reached else ("💥 DEAD" if robot.hit_mine else "⏰ TIMEOUT")
            self._log(
                f"  {robot.name}: {status} | steps={robot.steps_taken} "
                f"| waits={robot.waits_taken}"
            )
            summary[robot.id] = {
                "won": robot.reached,
                "dead": robot.hit_mine,
                "steps": robot.steps_taken,
                "waits": robot.waits_taken,
            }
        self.round_summary.append(summary)

        if self.round_number >= MAX_ROUNDS:
            self.tournament_over = True
            self._announce_winner()

    def _announce_winner(self):
        self._log("━━━━ TOURNAMENT OVER ━━━━")
        ranked = sorted(self.robots, key=lambda r: self.avg_efficiency(r), reverse=True)
        for rank, robot in enumerate(ranked, 1):
            eff  = self.avg_efficiency(robot)
            surv = self.survival_rate(robot)
            self._log(
                f"  #{rank} {robot.name}: eff={eff:.2f} | surv={surv:.0%} "
                f"| wins={robot.total_wins} | deaths={robot.total_deaths}"
            )
        self._export_data()

    def _export_data(self):
        data = {
            "tournament_summary": {
                "rounds": MAX_ROUNDS,
                "robots": NUM_ROBOTS,
                "grid": {
                    "rows": self.world.rows,
                    "cols": self.world.cols,
                    "walls": len(self.world.walls),
                    "mines": len(self.world.mines)
                }
            },
            "robots": [],
            "round_history": self.round_summary,
            "full_log": self.log
        }
        for r in self.robots:
            data["robots"].append({
                "id": r.id,
                "name": getattr(r, 'name', f'Robot {r.id+1}'),
                "speed": r.speed,
                "risk_tolerance": r.risk_tolerance,
                "patience": r.patience,
                "total_wins": r.total_wins,
                "total_deaths": r.total_deaths,
                "avg_efficiency": self.avg_efficiency(r),
                "survival_rate": self.survival_rate(r)
            })
            
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("logs", f"simulation_data_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        self._log(f"Data exported to {filename}")

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def tick_simulation(self):
        """Advance simulation by one tick. Called from main.py game loop."""
        self.tick_logs = []
        if not self.running:
            return

        self.tick += 1

        # Let each robot take its turn
        for robot in self.robots:
            if robot.alive and not robot.reached:
                robot.decide(self.robots, self._log)

        # ── Collision Check ──
        import collections
        pos_map = collections.defaultdict(list)
        for robot in self.robots:
            if robot.alive and not robot.reached:
                pos_map[robot.pos].append(robot)
                
        for pos, bots in pos_map.items():
            if len(bots) > 1:
                names = [b.name for b in bots]
                self._log(f"💥 COLLISION at {pos}! {', '.join(names)} crashed into each other and exploded!")
                for b in bots:
                    b.alive = False
                    b.hit_mine = True  # treat as death for stats

        # Check if round is over
        alive_not_reached = [r for r in self.robots if r.alive and not r.reached]
        alive_total = [r for r in self.robots if r.alive]
        
        # Default win: if only one robot remains alive and hasn't reached yet
        if len(alive_total) == 1 and len(alive_not_reached) == 1:
            lone_survivor = alive_not_reached[0]
            lone_survivor.reached = True
            self._log(f"🏆 {lone_survivor.name} wins by default as the last survivor!")
            alive_not_reached = []

        all_done = len(alive_not_reached) == 0
        timed_out = self.tick >= MAX_TICKS

        if all_done or timed_out:
            self._end_round()

    # ------------------------------------------------------------------
    # Advance to next round (called from renderer on SPACE press)
    # ------------------------------------------------------------------

    def next_round(self):
        if self.round_over and not self.tournament_over:
            self._start_round()

    def reset(self):
        """Full reset: new world, new robots, round 1."""
        self.__init__()

    # ------------------------------------------------------------------
    # Stats helpers (used by renderer)
    # ------------------------------------------------------------------

    def avg_efficiency(self, robot):
        if not robot.total_efficiency:
            return 0.0
        return sum(robot.total_efficiency) / len(robot.total_efficiency)

    def survival_rate(self, robot):
        total = len(robot.total_efficiency)
        if total == 0:
            return 0.0
        survived = total - robot.total_deaths
        return survived / total

    def avg_wait(self, robot):
        if not robot.total_wait_times:
            return 0.0
        return sum(robot.total_wait_times) / len(robot.total_wait_times)

    def leaderboard(self):
        """Return robots sorted by average efficiency descending."""
        return sorted(self.robots, key=lambda r: self.avg_efficiency(r), reverse=True)
