"""
robot.py — Robot agent with:
  - A* pathfinding on partial (shared) knowledge
  - Multi-robot collision prediction (3-step horizon)
  - Risk-scored move selection using mine probability + collision probability
  - Waiting behaviour with patience-driven hold durations
  - Per-round statistics tracking
"""

import random
from utils import astar, shortest_path_length
from grid import CELL_WALL, CELL_MINE, CELL_EMPTY, CELL_UNKNOWN


ROBOT_COLORS = [
    (59,  130, 246),   # Blue
    (239, 68,  68),    # Red
    (34,  197, 94),    # Green
    (251, 191, 36),    # Amber
]


class Robot:
    def __init__(self, robot_id, world):
        self.id           = robot_id
        self.color        = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        
        # Assign color names
        color_names = ["Blue", "Red", "Green", "Yellow"]
        self.name         = color_names[robot_id % len(color_names)]
        
        self.world        = world

        # Personality traits — randomised each game (fixed per robot across rounds)
        self.speed           = random.choice([1, 2])         # cells advanced per tick
        self.risk_tolerance  = random.uniform(0.10, 0.90)   # 0=very cautious, 1=reckless
        self.patience        = random.uniform(0.10, 0.90)   # 0=impatient, 1=very patient

        # State (reset each round via reset_round)
        self.pos          = None
        self.target       = None
        self.planned_path = []   # list of (r,c) from current pos to target (excl current)
        self.wait_counter = 0    # ticks still waiting
        self.alive        = True
        self.reached      = False

        # Per-round stats (accumulated into cumulative by game.py)
        self.steps_taken  = 0
        self.waits_taken  = 0
        self.hit_mine     = False
        self._shortest_known = None  # BFS shortest at round start
        self._avoid_cells = set()    # cells temporarily avoided due to high risk

        # Cumulative stats across all rounds
        self.total_wins       = 0
        self.total_deaths      = 0
        self.total_efficiency  = []  # list of floats
        self.total_wait_times  = []  # list of ints

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def reset_round(self, start, target):
        self.pos          = start
        self.target       = target
        self.planned_path = []
        self.wait_counter = 0
        self.alive        = True
        self.reached      = False
        self.steps_taken  = 0
        self.waits_taken  = 0
        self.hit_mine     = False
        self.idle_ticks   = 0
        self._avoid_cells = set()
        self._shortest_known = shortest_path_length(
            start, target, self.world.rows, self.world.cols, self.world.walls
        )

    def record_round_result(self):
        """Called by game.py after each round ends."""
        if self.reached:
            self.total_wins += 1
            spl = self._shortest_known or 1
            eff = spl / max(1, self.steps_taken)
            self.total_efficiency.append(min(1.0, eff))
        elif self.hit_mine:
            self.total_deaths += 1
            self.total_efficiency.append(0.0)
        else:
            # Did not reach — timed out
            self.total_efficiency.append(0.0)
        self.total_wait_times.append(self.waits_taken)

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def replan(self):
        """Run A* on current shared knowledge to find path to target."""
        w = self.world

        def blocked(r, c):
            status = w.known_cells.get((r, c), CELL_UNKNOWN)
            return status in (CELL_WALL, CELL_MINE)

        def cost(r, c):
            status = w.known_cells.get((r, c), CELL_UNKNOWN)
            penalty = 100 if (r, c) in getattr(self, '_avoid_cells', set()) else 0
            if status == CELL_EMPTY:
                return 1 + penalty
            # Unknown cell: penalise proportional to mine probability
            mine_p = w.get_mine_probability(r, c)
            return 1 + mine_p * 3 + penalty

        path = astar(self.pos, self.target, w.rows, w.cols, cost, blocked)
        self.planned_path = path[1:] if len(path) > 1 else []  # exclude start

    # ------------------------------------------------------------------
    # Other-robot prediction (horizon steps ahead)
    # ------------------------------------------------------------------

    def predict_others(self, other_robots, horizon=5):
        """
        Returns dict: step (1..horizon) -> set of (r,c) cells predicted occupied by others.
        """
        by_step = {s: set() for s in range(1, horizon + 1)}
        for bot in other_robots:
            if bot.id == self.id or not bot.alive:
                continue
            # Simulate bot advancing along their planned path by their speed each tick
            path   = bot.planned_path[:]
            path_i = 0
            for step in range(1, horizon + 1):
                advance = bot.speed
                path_i  = min(path_i + advance, len(path))
                if path_i > 0:
                    by_step[step].add(path[path_i - 1])
                else:
                    by_step[step].add(bot.pos)
        return by_step

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _score_cell(self, cell, predictions):
        """
        Compute risk score [0, 1] for moving into cell at next step.
        Risk = mine_p*0.3 + collision_p*0.4 + is_unknown*0.3
        """
        r, c = cell
        w = self.world
        status = w.known_cells.get(cell, CELL_UNKNOWN)

        # Mine probability component
        if status == CELL_MINE:
            mine_p = 1.0
        elif status == CELL_WALL:
            return 1.0  # impassable
        elif status == CELL_EMPTY:
            mine_p = 0.0
        else:
            mine_p = w.get_mine_probability(r, c)

        # Collision probability: check step 1 predictions
        step1_cells = predictions.get(1, set())
        collision_p = 1.0 if cell in step1_cells else 0.0

        # Unknown penalty
        is_unknown = 1.0 if status == CELL_UNKNOWN else 0.0

        # Adjust weights to heavily penalize mines, but allow passing through other robots (low collision penalty)
        risk = mine_p * 0.7 + collision_p * 0.1 + is_unknown * 0.2
        return min(1.0, risk)

    # ------------------------------------------------------------------
    # Main decision tick
    # ------------------------------------------------------------------

    def decide(self, other_robots, log_fn=None):
        """Make one tick's worth of decisions."""
        if not self.alive or self.reached:
            return

        w = self.world
        pos_at_start_of_tick = self.pos

        # -- Scan neighbourhood (discover adjacent walls before deciding) --
        w.scan_adjacent(*self.pos)

        # -- Handle waiting --
        if self.wait_counter > 0:
            self.wait_counter -= 1
            self.waits_taken  += 1
            if log_fn:
                log_fn(f"{self.name} waiting at {self.pos} ({self.wait_counter}t left)")
            return

        # -- Replan if path is empty or next cell is now known-blocked --
        if not self.planned_path:
            self.replan()
        elif w.known_cells.get(self.planned_path[0]) in (CELL_WALL, CELL_MINE):
            self.replan()

        if not self.planned_path:
            if log_fn:
                log_fn(f"{self.name} stuck — no path from {self.pos} to {self.target}")
            # Check for idle timeout
            self.idle_ticks += 1
            if self.idle_ticks >= 10:
                self.alive = False
                if log_fn:
                    log_fn(f"💀 {self.name} disqualified for being stuck/idle for 10 ticks!")
            return

        predictions = self.predict_others(other_robots, horizon=5)
        risk_threshold = self.risk_tolerance

        # -- Advance by up to self.speed cells --
        for _ in range(self.speed):
            if not self.planned_path or self.reached:
                break

            # If path is now blocked, replan before this step
            if w.known_cells.get(self.planned_path[0]) in (CELL_WALL, CELL_MINE):
                self.replan()
                if not self.planned_path:
                    break

            next_cell = self.planned_path[0]
            risk = self._score_cell(next_cell, predictions)

            if risk > risk_threshold:
                # Too risky! Try to find a different path.
                if next_cell not in getattr(self, '_avoid_cells', set()):
                    self._avoid_cells.add(next_cell)
                    self.replan()
                    wait_turns = max(1, int(self.patience * 4))
                    self.wait_counter = wait_turns
                    self.waits_taken += 1
                    if log_fn:
                        log_fn(f"{self.name} avoids risky path and replans at {self.pos}")
                    break
                else:
                    # We already tried avoiding it and we're back here, meaning no other choice!
                    # Temporarily increase risk tolerance to force movement eventually
                    self.risk_tolerance = min(1.0, self.risk_tolerance + 0.1)
                    wait_turns = max(1, int(self.patience * 2))
                    self.wait_counter = wait_turns
                    self.waits_taken += 1
                    if log_fn:
                        log_fn(f"{self.name} waits {wait_turns}t (impatient! risk_tol={self.risk_tolerance:.2f})")
                    break

            # Step into next_cell
            old_pos = self.pos
            self.planned_path.pop(0)
            self.pos        = next_cell
            self.steps_taken += 1
            
            # Record visit for GUI trail rendering
            if hasattr(w, 'visited'):
                w.visited[self.pos].add(self.color)
            
            if log_fn:
                log_fn(f"{self.name} moved from {old_pos} to {self.pos}")

            # ── Immediate mid-tick Collision Check ──
            for other in other_robots:
                if other.id != self.id and other.alive and not other.reached and other.pos == self.pos:
                    self.alive = False
                    self.hit_mine = True
                    other.alive = False
                    other.hit_mine = True
                    if log_fn:
                        log_fn(f"💥 COLLISION at {self.pos}! {self.name} ran straight into {other.name} and exploded!")
                    return

            # Discover the cell
            is_mine = w.discover(*self.pos)

            if is_mine:
                self.alive    = False
                self.hit_mine = True
                if log_fn:
                    log_fn(f"💥 {self.name} HIT MINE at {self.pos}!")
                return

            # Check arrival
            if self.pos == self.target:
                self.reached = True
                if log_fn:
                    log_fn(f"✅ {self.name} reached target {self.target} in {self.steps_taken} steps!")
                return

            # After moving, refresh predictions offset by 1
            predictions = self.predict_others(other_robots, horizon=5)

        # Check for idle timeout
        if self.pos == pos_at_start_of_tick:
            self.idle_ticks += 1
            if self.idle_ticks >= 10:
                self.alive = False
                if log_fn:
                    log_fn(f"💀 {self.name} disqualified for being stuck/idle for 10 ticks!")
        else:
            self.idle_ticks = 0
