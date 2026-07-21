"""
robot.py — Robot agent for the multi-agent grid-world simulation.

Each robot:
  - Has a personality (speed, risk_tolerance, patience) — controllable for experiments.
  - Uses a configurable Planner (ShortestPath / RiskAware / variants).
  - Makes decisions from AgentKnowledge ONLY — no ground-truth access.
  - Exposes observe(), decide_action() and apply_action() for the phased tick in game.py.
  - Tracks separate death types: mine deaths vs collision deaths vs timeouts.
  - Tracks risk relaxation events explicitly.
  - Tracks conflict wins/losses for fairness analysis.

Five-phase tick protocol (implemented in game.py):
  Phase 1 — Observation:
      All robots call observe(). All observations complete before any decisions.
  Phase 2 — Knowledge Update:
      Constraint propagation runs once after all observations.
  Phase 3 — Decision:
      All robots call decide_action(snapshot) using the same knowledge state.
      No state is mutated. Each robot records its intended_action.
  Phase 4 — Conflict Resolution:
      game.py resolves all intended actions simultaneously using priority policy.
  Phase 5 — Action Application:
      game.py applies approved actions, triggers discovery, checks hazards.

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
from config import (
    DEFAULT_RISK_RELAXATION, DEFAULT_SIM_CONFIG,
    RiskRelaxationConfig,
)


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

    def __init__(self, kind: str, cells: Optional[List[Tuple[int, int]]] = None,
                 reason: str = ''):
        self.kind   = kind
        self.cells  = cells or []   # sequence of cells to move through this tick
        self.reason = reason        # human-readable explanation

    def __repr__(self):
        return f"Action({self.kind}, cells={self.cells}, reason={self.reason!r})"


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
      - initial_risk_tolerance, effective_risk_tolerance, risk_relaxation_count
      - conflicts_won, conflicts_lost, consecutive_waits, max_consecutive_waits
      - priority_boost (for fairness)

    Cumulative stats (accumulated across all rounds):
      - total_wins, total_mine_deaths, total_collision_deaths, total_timeouts
      - total_efficiency, total_wait_times
    """

    def __init__(self, robot_id: int, world, planner_name: str = 'risk_aware',
                 planner_config: Optional[PlannerConfig] = None,
                 risk_relaxation_config: Optional[RiskRelaxationConfig] = None,
                 rng: Optional[random.Random] = None):
        self.id     = robot_id
        self.color  = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        self.name   = COLOR_NAMES[robot_id % len(COLOR_NAMES)]
        self.world  = world   # GridWorld reference (for discover/scan_adjacent)

        # Use isolated RNG to avoid polluting global random state
        _rng = rng or random.Random()

        # Personality traits — fixed per robot across all rounds
        # These are experimental variables — see experiments.py
        self.speed           = _rng.choice([1, 2])
        self.risk_tolerance  = _rng.uniform(0.10, 0.90)
        self.patience        = _rng.uniform(0.10, 0.90)

        # Risk relaxation config
        self._risk_relaxation_config = risk_relaxation_config or DEFAULT_RISK_RELAXATION

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

        # Risk relaxation tracking
        self.initial_risk_tolerance   : float = self.risk_tolerance
        self.effective_risk_tolerance : float = self.risk_tolerance
        self.risk_relaxation_count    : int = 0
        self.risk_relaxation_log      : List[Dict] = []  # log of relaxation events

        # Conflict tracking
        self.conflicts_won            : int = 0
        self.conflicts_lost           : int = 0
        self.consecutive_waits        : int = 0
        self.max_consecutive_waits    : int = 0
        self.priority_boost           : float = 0.0  # fairness boost

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

        # Reset risk relaxation state (restore original tolerance)
        self.effective_risk_tolerance = self.risk_tolerance
        self.risk_relaxation_count    = 0
        self.risk_relaxation_log      = []

        # Reset conflict tracking
        self.conflicts_won         = 0
        self.conflicts_lost        = 0
        self.consecutive_waits     = 0
        self.max_consecutive_waits = 0
        self.priority_boost        = 0.0

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
    # Phase 1: Observation — separated from decision-making
    # ------------------------------------------------------------------

    def observe(self) -> None:
        """
        Phase 1: Scan the robot's local environment.

        This method is called for ALL robots before ANY robot makes a decision.
        It updates shared AgentKnowledge with wall observations only
        (mines are only discovered by physically entering a cell).

        This method MUST be called before decide_action().
        """
        if not self.alive or self.reached:
            return
        self.world.scan_adjacent(*self.pos)

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
    # Phase 3: decide_action — pure decision, no state mutation
    # ------------------------------------------------------------------

    def decide_action(self, other_robots, log_fn=None) -> Action:
        """
        Phase 3 of the five-phase tick: determine intended action.

        This method MUST NOT mutate self.pos, self.alive, self.reached,
        or any game state.  It only reads state and returns an Action.

        The robot consults AgentKnowledge (via self.world.knowledge) only.
        No GridWorld.walls, GridWorld.mines are accessed here.

        IMPORTANT: observe() must have been called for ALL robots before
        this method is called for ANY robot. This is enforced by game.py.

        Returns:
            Action describing what the robot intends to do this tick.
        """
        if not self.alive or self.reached:
            return Action(Action.DONE, reason='already done')

        knowledge = self.world.knowledge

        # NOTE: scan_adjacent is NO LONGER called here.
        # It is called in observe() during Phase 1, before any decisions.

        # Handle waiting (robot is in a wait state from previous tick)
        if self.wait_counter > 0:
            return Action(Action.WAIT, reason=f'waiting ({self.wait_counter}t left)')

        # Replan if path is empty or next cell is now known-blocked
        if not self.planned_path:
            self.replan()
        elif knowledge.is_known_blocked(*self.planned_path[0]):
            self.replan()

        if not self.planned_path:
            return Action(Action.WAIT, reason='no path found')

        # Compute trajectory predictions for collision awareness
        horizon = self._planner_config.collision_horizon if self._planner_config else 5
        predictions = self.predict_others(other_robots, horizon=horizon)

        # Determine up to self.speed cells to move through this tick
        intended_cells: List[Tuple[int, int]] = []
        path_copy = self.planned_path[:]

        for _ in range(self.speed):
            if not path_copy:
                break

            # Replan if next cell is now blocked
            if knowledge.is_known_blocked(*path_copy[0]):
                break

            next_cell = path_copy[0]
            risk = self.planner.score_cell(next_cell, knowledge, predictions)

            if risk > self.effective_risk_tolerance:
                # Too risky — will wait and replan (handled in apply_action)
                break

            intended_cells.append(path_copy.pop(0))

        if intended_cells:
            return Action(Action.MOVE, cells=intended_cells,
                          reason=f'planned move through {len(intended_cells)} cells')
        return Action(Action.WAIT, reason='risk too high or path blocked')

    # ------------------------------------------------------------------
    # Phase 5: apply_action — state mutation happens here
    # ------------------------------------------------------------------

    def apply_action(self, action: Action, log_fn=None) -> Optional[str]:
        """
        Phase 5 of the five-phase tick: apply the previously decided action.

        Called by game.py after conflict resolution (Phase 4).
        This method mutates self.pos, waits, path, etc.

        Collision detection is handled by game.py (it sees all positions at once).
        This method handles: movement, mine discovery, arrival, wait timers,
        replanning, and idle-timeout.

        Args:
            action : Action returned by decide_action() this tick
                     (possibly modified by conflict resolution)
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

            # Track consecutive waits
            self.consecutive_waits += 1
            self.max_consecutive_waits = max(
                self.max_consecutive_waits, self.consecutive_waits
            )

            # Idle timeout check
            self.idle_ticks += 1
            max_idle = DEFAULT_SIM_CONFIG.MAX_CONSECUTIVE_IDLE_TICKS
            if self.idle_ticks >= max_idle:
                self.alive     = False
                self.timed_out = True
                if log_fn:
                    log_fn(f"💀 {self.name} idle-timeout at {self.pos} after {max_idle} ticks!")
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
                if risk > self.effective_risk_tolerance:
                    # Risk is now too high — avoid and replan with explicit logging
                    self._handle_risk_rejection(next_cell, risk, knowledge, log_fn)
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

                # Reset consecutive waits on successful movement
                self.consecutive_waits = 0

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
                # Decay fairness priority boost on successful movement
                self.priority_boost *= DEFAULT_SIM_CONFIG.FAIRNESS.decay_rate
            else:
                self.idle_ticks += 1
                max_idle = DEFAULT_SIM_CONFIG.MAX_CONSECUTIVE_IDLE_TICKS
                if self.idle_ticks >= max_idle:
                    self.alive     = False
                    self.timed_out = True
                    if log_fn:
                        log_fn(f"💀 {self.name} idle-timeout at {self.pos}!")

        return None

    def _handle_risk_rejection(self, next_cell, risk, knowledge, log_fn):
        """
        Handle the case where the robot rejects a cell due to high risk.
        """
        cfg = self._risk_relaxation_config

        if next_cell not in self._avoid_cells:
            # First rejection: avoid and replan
            self._avoid_cells.add(next_cell)
            self.replan()
            
            if not self.planned_path:
                # No alternative path exists, must relax risk
                self._apply_risk_relaxation(next_cell, risk, log_fn)
                self._avoid_cells.discard(next_cell)
                self.replan()
            else:
                wait_turns = max(1, int(self.patience * 4))
                self.wait_counter = wait_turns
                self.waits_taken += 1
                if log_fn:
                    log_fn(f"{self.name} avoids risky cell {next_cell} "
                           f"(risk={risk:.3f} > tol={self.effective_risk_tolerance:.3f}), replanning")
        else:
            # Already tried avoiding — apply risk relaxation
            self._apply_risk_relaxation(next_cell, risk, log_fn)

    def _apply_risk_relaxation(self, next_cell, risk, log_fn):
        cfg = self._risk_relaxation_config
        if self.risk_relaxation_count < cfg.max_relaxations:
            old_tol = self.effective_risk_tolerance
            self.effective_risk_tolerance = min(
                cfg.max_tolerance,
                self.effective_risk_tolerance + cfg.relax_step
            )
            self.risk_relaxation_count += 1

            # Record relaxation event
            event = {
                'initial_tolerance': self.risk_tolerance,
                'previous_tolerance': old_tol,
                'new_tolerance': self.effective_risk_tolerance,
                'relaxation_number': self.risk_relaxation_count,
                'reason': f'cell {next_cell} risk={risk:.3f}, no alternative path',
                'cell': next_cell,
            }
            self.risk_relaxation_log.append(event)

            wait_turns = max(1, int(self.patience * 2))
            self.wait_counter = wait_turns
            self.waits_taken += 1
            if log_fn:
                log_fn(
                    f"{self.name} risk relaxation #{self.risk_relaxation_count}: "
                    f"tolerance {old_tol:.3f} → {self.effective_risk_tolerance:.3f} "
                    f"(initial={self.risk_tolerance:.3f})"
                )
        else:
            # Max relaxations reached — force accept
            wait_turns = max(1, int(self.patience * 2))
            self.wait_counter = wait_turns
            self.waits_taken += 1
            if log_fn:
                log_fn(
                    f"{self.name} max relaxations reached ({cfg.max_relaxations}), "
                    f"forced to wait at {self.pos}"
                )

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
            if risk > self.effective_risk_tolerance:
                self._handle_risk_rejection(next_cell, risk, knowledge, log_fn)

    # ------------------------------------------------------------------
    # Conflict tracking (called by game.py conflict resolution)
    # ------------------------------------------------------------------

    def record_conflict_win(self):
        """Record that this robot won a conflict resolution."""
        self.conflicts_won += 1

    def record_conflict_loss(self):
        """Record that this robot lost a conflict resolution."""
        self.conflicts_lost += 1
        self.consecutive_waits += 1
        self.max_consecutive_waits = max(
            self.max_consecutive_waits, self.consecutive_waits
        )
        # Check if starvation prevention should kick in
        fairness = DEFAULT_SIM_CONFIG.FAIRNESS
        if self.consecutive_waits >= fairness.max_consecutive_losses:
            self.priority_boost += fairness.priority_boost

    def get_priority_score(self) -> float:
        """
        Compute priority score for conflict resolution.

        Higher score = higher priority (gets to move).
        Components:
            - Base: consecutive wait time
            - Boost: accumulated fairness boost from repeated losses
        """
        return float(self.consecutive_waits) + self.priority_boost

    # ------------------------------------------------------------------
    # Legacy decide() — kept for backward compatibility if needed
    # (Not used in the five-phase tick; use observe + decide_action + apply_action)
    # ------------------------------------------------------------------

    def decide(self, other_robots, log_fn=None) -> None:
        """
        Legacy single-phase decision method.
        Preserved for backward compatibility but NOT used in the five-phase tick.
        Use observe() + decide_action() + apply_action() in game.py instead.
        """
        self.observe()
        action = self.decide_action(other_robots, log_fn)
        self.apply_action(action, log_fn)
