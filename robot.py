"""
robot.py — Robot agent for the multi-agent grid-world simulation.

Each robot:
  - Has a personality (speed, risk_tolerance, patience) — controllable for experiments.
  - Uses a configurable Planner (ShortestPath / RiskAware / Conservative).
  - Makes decisions from AgentKnowledge ONLY — no ground-truth access.
  - Exposes decide_action() and apply_action() for the two-phase tick in game.py.
  - Tracks separate death types: mine deaths vs collision deaths vs timeouts.

Two-phase tick protocol (implemented in game.py):
  Phase 1 — Decision:
      All robots call decide_action(snapshot) using the same world state.
      No state is mutated.  Each robot records its intended_action.

  Phase 2 — Resolution:
      game.py resolves all intended actions simultaneously:
        - Detects same-cell collisions and swap collisions.
        - Applies movement and triggers discover() for new cells.
        - Updates alive/reached/death flags.

Terminology note:
  predict_others() implements SHORT-HORIZON DETERMINISTIC TRAJECTORY PREDICTION.
  It simulates other robots advancing along their planned paths.
  It is NOT probabilistic collision prediction.
  Probability vocabulary is reserved for mine probability estimation.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from planner import make_planner, PlannerConfig, RiskAwarePlanner
from knowledge import AgentKnowledge, CellStatus


# ---------------------------------------------------------------------------
# Robot colors
# ---------------------------------------------------------------------------

ROBOT_COLORS = [
    (59,  130, 246),   # Blue
    (239, 68,  68),    # Red
    (34,  197, 94),    # Green
    (251, 191, 36),    # Amber / Yellow
]

COLOR_NAMES = ["Blue", "Red", "Green", "Yellow"]


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class Action:
    """Encapsulates a robot's intended action for one tick."""
    MOVE   = 'move'
    WAIT   = 'wait'
    DONE   = 'done'   # already reached or dead

    def __init__(self, kind: str, cells: Optional[List[Tuple[int, int]]] = None):
        self.kind  = kind
        self.cells = cells or []   # sequence of cells to move through this tick

    def __repr__(self):
        return f"Action({self.kind}, cells={self.cells})"


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class Robot:
    """
    An autonomous robot agent.

    Personality traits are fixed per robot across all rounds (they represent
    the robot's inherent characteristics, not its runtime state).

    Round state (reset each round via reset_round):
      - pos, target, planned_path
      - alive, reached
      - hit_mine, hit_robot, timed_out  (separate death type flags)
      - steps_taken, waits_taken, replans_taken
      - unknown_cells_entered, mines_avoided_by_rerouting

    Cumulative stats (accumulated across all rounds):
      - total_wins, total_mine_deaths, total_collision_deaths, total_timeouts
      - total_efficiency, total_wait_times
    """

    def __init__(self, robot_id: int, world, planner_name: str = 'risk_aware',
                 planner_config: Optional[PlannerConfig] = None):
        self.id     = robot_id
        self.color  = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        self.name   = COLOR_NAMES[robot_id % len(COLOR_NAMES)]
        self.world  = world   # GridWorld reference (for discover/scan_adjacent)

        # Personality traits — fixed per robot across all rounds
        # These are experimental variables — see experiments.py
        self.speed           = random.choice([1, 2])
        self.risk_tolerance  = random.uniform(0.10, 0.90)
        self.patience        = random.uniform(0.10, 0.90)

        # Planner — configurable for experiments
        self._planner_name   = planner_name
        self._planner_config = planner_config
        self.planner         = make_planner(planner_name, planner_config)

        # Round state (initialised by reset_round)
        self.pos           : Optional[Tuple[int, int]] = None
        self.target        : Optional[Tuple[int, int]] = None
        self.planned_path  : List[Tuple[int, int]] = []
        self.wait_counter  : int = 0
        self.idle_ticks    : int = 0
        self.alive         : bool = True
        self.reached       : bool = False

        # Separate death type flags (not conflated into one variable)
        self.hit_mine      : bool = False   # robot stepped on a mine
        self.hit_robot     : bool = False   # robot collided with another robot
        self.timed_out     : bool = False   # round ended without reaching target

        # Per-round stats
        self.steps_taken           : int = 0
        self.waits_taken           : int = 0
        self.replans_taken         : int = 0
        self.unknown_cells_entered : int = 0

        # Cells temporarily avoided during replanning
        self._avoid_cells: Set[Tuple[int, int]] = set()

        # Shortest-path oracle length (computed at round start using true walls)
        # [ORACLE] — ground truth, used only for post-hoc efficiency metrics
        self._shortest_oracle: Optional[int] = None

        # Cumulative stats across all rounds
        self.total_wins             : int   = 0
        self.total_mine_deaths      : int   = 0
        self.total_collision_deaths : int   = 0
        self.total_timeouts         : int   = 0
        self.total_efficiency       : List[float] = []
        self.total_wait_times       : List[int]   = []

        # Legacy alias for compatibility with game.py / renderer
        @property
        def total_deaths(self):
            return self.total_mine_deaths + self.total_collision_deaths

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def reset_round(self, start: Tuple[int, int], target: Tuple[int, int],
                    oracle_wall_set=None) -> None:
        """
        Reset per-round state for a new round.

        Args:
            start           : starting cell (ground truth used for placement only)
            target          : target cell
            oracle_wall_set : [ORACLE] true wall set, used ONLY to compute the
                              optimal shortest-path length for efficiency metrics.
                              NOT passed to the planner or probability model.
        """
        self.pos           = start
        self.target        = target
        self.planned_path  = []
        self.wait_counter  = 0
        self.idle_ticks    = 0
        self.alive         = True
        self.reached       = False

        # Clear separate death flags
        self.hit_mine      = False
        self.hit_robot     = False
        self.timed_out     = False

        # Clear per-round stats
        self.steps_taken           = 0
        self.waits_taken           = 0
        self.replans_taken         = 0
        self.unknown_cells_entered = 0
        self._avoid_cells          = set()

        # Re-initialise planner (fresh instance each round)
        self.planner = make_planner(self._planner_name, self._planner_config)

        # [ORACLE] Compute ground-truth shortest path for efficiency reporting only
        if oracle_wall_set is not None:
            from utils import shortest_path_length
            self._shortest_oracle = shortest_path_length(
                start, target, self.world.rows, self.world.cols, oracle_wall_set
            )
        else:
            self._shortest_oracle = None

    def record_round_result(self) -> None:
        """Called by game.py after each round ends to accumulate cumulative stats."""
        if self.reached:
            self.total_wins += 1
            spl = self._shortest_oracle or 1
            eff = spl / max(1, self.steps_taken)
            self.total_efficiency.append(min(1.0, eff))
        elif self.hit_mine:
            self.total_mine_deaths += 1
            self.total_efficiency.append(0.0)
        elif self.hit_robot:
            self.total_collision_deaths += 1
            self.total_efficiency.append(0.0)
        elif self.timed_out:
            self.total_timeouts += 1
            self.total_efficiency.append(0.0)
        else:
            self.total_efficiency.append(0.0)

        self.total_wait_times.append(self.waits_taken)

    @property
    def total_deaths(self) -> int:
        """Total deaths across all rounds (mine + collision)."""
        return self.total_mine_deaths + self.total_collision_deaths

    # ------------------------------------------------------------------
    # Path planning (uses AgentKnowledge only)
    # ------------------------------------------------------------------

    def replan(self) -> None:
        """
        Re-run A* using current AgentKnowledge to find a path to target.
        No ground-truth information is used.
        """
        knowledge = self.world.knowledge
        path = self.planner.plan(
            self.pos,
            self.target,
            knowledge,
            avoid_cells=self._avoid_cells,
            predictions=None,   # predictions computed separately per tick
        )
        self.planned_path = path[1:] if len(path) > 1 else []
        self.replans_taken += 1

    # ------------------------------------------------------------------
    # Short-horizon deterministic trajectory prediction for other robots
    # ------------------------------------------------------------------

    def predict_others(self, other_robots, horizon: int = 5) -> Dict[int, Set]:
        """
        Predict future positions of other robots using their planned paths.

        This is SHORT-HORIZON DETERMINISTIC TRAJECTORY PREDICTION.
        It is NOT probabilistic — it simulates each robot advancing along its
        declared planned_path.  It does not account for dynamic replanning.

        The prediction horizon used here (default 5) must match documentation.

        Args:
            other_robots : list of all Robot objects
            horizon      : number of steps to look ahead

        Returns:
            dict: step (1..horizon) -> set of (r,c) predicted positions
        """
        by_step: Dict[int, Set[Tuple[int, int]]] = {s: set() for s in range(1, horizon + 1)}
        for bot in other_robots:
            if bot.id == self.id or not bot.alive:
                continue
            path   = bot.planned_path[:]
            path_i = 0
            for step in range(1, horizon + 1):
                path_i = min(path_i + bot.speed, len(path))
                if path_i > 0:
                    by_step[step].add(path[path_i - 1])
                else:
                    by_step[step].add(bot.pos)
        return by_step

    # ------------------------------------------------------------------
    # Phase 1: decide_action — pure decision, no state mutation
    # ------------------------------------------------------------------

    def decide_action(self, other_robots, log_fn=None) -> Action:
        """
        Phase 1 of the two-phase tick: determine intended action.

        This method MUST NOT mutate self.pos, self.alive, self.reached,
        or any game state.  It only reads state and returns an Action.

        The robot consults AgentKnowledge (via self.world.knowledge) only.
        No GridWorld.walls, GridWorld.mines are accessed here.

        Returns:
            Action describing what the robot intends to do this tick.
        """
        if not self.alive or self.reached:
            return Action(Action.DONE)

        knowledge = self.world.knowledge

        # Scan neighbourhood BEFORE deciding (read-only: just marks knowledge)
        self.world.scan_adjacent(*self.pos)

        # Handle waiting (robot is in a wait state from previous tick)
        if self.wait_counter > 0:
            return Action(Action.WAIT)

        # Replan if path is empty or next cell is now known-blocked
        if not self.planned_path:
            self.replan()
        elif knowledge.is_known_blocked(*self.planned_path[0]):
            self.replan()

        if not self.planned_path:
            return Action(Action.WAIT)  # stuck — game.py handles idle timeout

        # Compute trajectory predictions for collision awareness
        predictions = self.predict_others(other_robots, horizon=5)

        # Determine up to self.speed cells to move through this tick
        intended_cells: List[Tuple[int, int]] = []
        path_copy = self.planned_path[:]

        for _ in range(self.speed):
            if not path_copy:
                break

            # Replan if next cell is now blocked
            if knowledge.is_known_blocked(*path_copy[0]):
                # Can't replan within decide_action without mutating state;
                # signal a WAIT and let apply_action trigger a replan next tick
                break

            next_cell = path_copy[0]
            risk = self.planner.score_cell(next_cell, knowledge, predictions)

            if risk > self.risk_tolerance:
                # Too risky — will wait and replan (handled in apply_action)
                break

            intended_cells.append(path_copy.pop(0))

        if intended_cells:
            return Action(Action.MOVE, cells=intended_cells)
        return Action(Action.WAIT)

    # ------------------------------------------------------------------
    # Phase 2: apply_action — state mutation happens here
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, log_fn=None) -> Optional[str]:
        """
        Phase 2 of the two-phase tick: apply the previously decided action.

        Called by game.py after ALL robots have decided (simultaneous resolution).
        This method mutates self.pos, waits, path, etc.

        Collision detection is handled by game.py (it sees all positions at once).
        This method handles: movement, mine discovery, arrival, wait timers,
        replanning, and idle-timeout.

        Args:
            action : Action returned by decide_action() this tick
            log_fn : optional logging callback

        Returns:
            'mine'    if the robot hit a mine and died
            'reached' if the robot reached its target
            None      otherwise
        """
        if not self.alive or self.reached:
            return None

        pos_at_start = self.pos
        knowledge    = self.world.knowledge

        # ── Handle WAIT ──
        if action.kind == Action.WAIT:
            if self.wait_counter > 0:
                self.wait_counter -= 1
                self.waits_taken  += 1
                if log_fn:
                    log_fn(f"{self.name} waiting at {self.pos} ({self.wait_counter}t left)")
            else:
                # Stuck — no path available or avoiding risk
                # Initiate a wait + replan cycle
                self._attempt_replan_with_wait(knowledge, log_fn)

            # Idle timeout check
            self.idle_ticks += 1
            if self.idle_ticks >= 10:
                self.alive     = False
                self.timed_out = True
                if log_fn:
                    log_fn(f"💀 {self.name} idle-timeout at {self.pos} after 10 ticks!")
            return None

        # ── Handle MOVE ──
        if action.kind == Action.MOVE:
            for next_cell in action.cells:
                if not self.alive or self.reached:
                    break

                # Re-check: is the cell now known-blocked (updated since decide_action)?
                if knowledge.is_known_blocked(*next_cell):
                    self.replan()
                    break  # stop moving this tick; will replan next tick

                old_pos = self.pos

                # Re-check risk against updated knowledge
                risk = self.planner.score_cell(next_cell, knowledge)
                if risk > self.risk_tolerance:
                    # Risk is now too high — avoid and replan
                    if next_cell not in self._avoid_cells:
                        self._avoid_cells.add(next_cell)
                        self.replan()
                        wait_turns = max(1, int(self.patience * 4))
                        self.wait_counter = wait_turns
                        self.waits_taken += 1
                        if log_fn:
                            log_fn(f"{self.name} avoids risky cell {next_cell}, replanning")
                    else:
                        # Already tried avoiding — gradually increase tolerance
                        self.risk_tolerance = min(1.0, self.risk_tolerance + 0.1)
                        wait_turns = max(1, int(self.patience * 2))
                        self.wait_counter = wait_turns
                        self.waits_taken += 1
                        if log_fn:
                            log_fn(
                                f"{self.name} forced to accept risk "
                                f"(tolerance raised to {self.risk_tolerance:.2f})"
                            )
                    break

                # Check if cell is unknown (track for metrics)
                was_unknown = knowledge.is_unknown(*next_cell)

                # ── Advance into cell ──
                if next_cell in self.planned_path:
                    self.planned_path.remove(next_cell)
                self.pos        = next_cell
                self.steps_taken += 1
                if was_unknown:
                    self.unknown_cells_entered += 1

                # Record for trail rendering
                self.world.visited[self.pos].add(self.color)

                if log_fn:
                    log_fn(f"{self.name} moved {old_pos} → {self.pos}")

                # ── Discover cell (physics: reads ground truth internally) ──
                is_mine = self.world.discover(*self.pos)
                if is_mine:
                    self.alive    = False
                    self.hit_mine = True
                    if log_fn:
                        log_fn(f"💥 {self.name} HIT MINE at {self.pos}!")
                    return 'mine'

                # ── Arrival check ──
                if self.pos == self.target:
                    self.reached = True
                    if log_fn:
                        log_fn(
                            f"✅ {self.name} reached target {self.target} "
                            f"in {self.steps_taken} steps!"
                        )
                    return 'reached'

            # Reset idle counter if we moved at all
            if self.pos != pos_at_start:
                self.idle_ticks = 0
            else:
                self.idle_ticks += 1
                if self.idle_ticks >= 10:
                    self.alive     = False
                    self.timed_out = True
                    if log_fn:
                        log_fn(f"💀 {self.name} idle-timeout at {self.pos}!")

        return None

    def _attempt_replan_with_wait(self, knowledge, log_fn) -> None:
        """Helper: try to replan and set a wait if replanning fails."""
        old_len = len(self.planned_path)
        self.replan()
        if not self.planned_path:
            # Still stuck
            wait_turns = max(1, int(self.patience * 3))
            self.wait_counter = wait_turns
            self.waits_taken += 1
            if log_fn:
                log_fn(f"{self.name} stuck — no path found, waiting {wait_turns}t")
        else:
            # Found a new path — check if it's risky
            next_cell = self.planned_path[0]
            risk = self.planner.score_cell(next_cell, knowledge)
            if risk > self.risk_tolerance:
                if next_cell not in self._avoid_cells:
                    self._avoid_cells.add(next_cell)
                    self.replan()
                wait_turns = max(1, int(self.patience * 4))
                self.wait_counter = wait_turns
                self.waits_taken += 1
                if log_fn:
                    log_fn(f"{self.name} replanned but risky — waiting {wait_turns}t")

    # ------------------------------------------------------------------
    # Legacy decide() — kept for backward compatibility if needed
    # (Not used in the two-phase tick; use decide_action + apply_action)
    # ------------------------------------------------------------------

    def decide(self, other_robots, log_fn=None) -> None:
        """
        Legacy single-phase decision method.
        Preserved for backward compatibility but NOT used in the two-phase tick.
        Use decide_action() + apply_action() in game.py instead.
        """
        action = self.decide_action(other_robots, log_fn)
        self.apply_action(action, log_fn)
