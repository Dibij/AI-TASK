"""
planner.py — Configurable planners for the multi-agent grid world.

Three distinct planning strategies are implemented:

  ShortestPathPlanner (BASELINE):
      Uses A* with uniform cost=1 for every traversable cell.
      Treats all unknown cells as traversable with normal cost.
      Does NOT use mine-risk information.
      Purpose: experimental baseline for comparison.

  RiskAwarePlanner (DEFAULT / EXPERIMENT B):
      Uses A* with a composite cost function:
          cost = movement_cost
               + mine_risk_weight   * P(mine)
               + unknown_weight     * unknown_penalty
               + avoid_weight       * is_in_avoid_set
      Weights are configurable named parameters, not magic numbers.
      Uses only AgentKnowledge — no ground-truth access.

  ConservativeRiskPlanner (EXPERIMENT C):
      Like RiskAwarePlanner but with a significantly larger unknown_weight.
      Strongly prefers known-safe paths even if longer.

All planners use the SAME A* implementation from utils.py.
The blocked() predicate and cost_fn() are parameterised per planner.
This makes them easy to swap for experiments.

Collision risk is kept separate from mine risk in RiskAwarePlanner.
It is computed from a caller-supplied predictions dict (see Robot.decide).

Expected cost interpretation (for RiskAwarePlanner):
    expected_cost = movement_cost
                  + P(mine)      * mine_penalty
                  + P(collision) * collision_penalty
                  + unknown_penalty (if cell is unknown)

    Weights are manually chosen heuristics, NOT mathematically optimal values.
    Their effect should be measured experimentally (see experiments.py).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple, List

from utils import astar
from knowledge import AgentKnowledge, BLOCKED_STATUSES, CellStatus


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    """
    Named weight configuration for a planner.

    All weights are heuristic values — they are not mathematically proven
    to be optimal.  Vary them between experiments to measure their effect.

    Attributes:
        movement_cost      : base cost for stepping into any cell (default 1.0)
        mine_risk_weight   : multiplier on P(mine) for risk-aware planning
        unknown_weight     : flat penalty for stepping into an UNKNOWN cell
        collision_weight   : multiplier on P(collision) for collision avoidance
        avoid_weight       : large penalty for cells in the avoid-set
        mine_penalty       : expected-loss weight per mine event
        collision_penalty  : expected-loss weight per collision event
        planner_type       : one of 'shortest', 'risk_aware', 'conservative'
    """
    movement_cost      : float = 1.0
    mine_risk_weight   : float = 3.0
    unknown_weight     : float = 0.2
    collision_weight   : float = 0.5
    avoid_weight       : float = 100.0
    mine_penalty       : float = 10.0   # used in expected_cost calculation
    collision_penalty  : float = 5.0    # used in expected_cost calculation
    planner_type       : str   = 'risk_aware'  # 'shortest', 'risk_aware', 'conservative'


# Predefined configurations for experiments
CONFIGS = {
    'shortest': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=0.0,    # ignores mine risk
        unknown_weight=0.0,      # treats unknowns as normal
        collision_weight=0.0,    # ignores collision risk
        avoid_weight=0.0,
        planner_type='shortest',
    ),
    'risk_aware': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=3.0,
        unknown_weight=0.2,
        collision_weight=0.5,
        avoid_weight=100.0,
        planner_type='risk_aware',
    ),
    'conservative': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=5.0,
        unknown_weight=2.0,      # much larger: strongly avoids unknowns
        collision_weight=1.0,
        avoid_weight=100.0,
        planner_type='conservative',
    ),
}


# ---------------------------------------------------------------------------
# Planner classes
# ---------------------------------------------------------------------------

class ShortestPathPlanner:
    """
    Baseline planner: A* with uniform cost, ignoring mine risk.

    Treats every non-blocked cell as having cost=1 regardless of mine
    probability or unknown status.  This is the control condition for
    experiments — it shows the system's performance WITHOUT risk awareness.

    blocked() predicate: cells known to be walls or confirmed mines.
    Unknown cells are treated as traversable (cost 1).
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or CONFIGS['shortest']

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        knowledge: AgentKnowledge,
        avoid_cells: Optional[Set[Tuple[int, int]]] = None,
        predictions: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Plan a path from start to goal using uniform cost A*.

        Args:
            start      : current robot position
            goal       : target position
            knowledge  : agent's current knowledge state
            avoid_cells: cells currently flagged for avoidance (ignored by baseline)
            predictions: collision predictions (ignored by baseline)

        Returns:
            List of (r,c) tuples from start to goal inclusive, or [] if unreachable.
        """
        cfg = self.config

        def blocked(r, c):
            return knowledge.is_known_blocked(r, c)

        def cost_fn(r, c):
            return cfg.movement_cost

        path = astar(start, goal, knowledge.rows, knowledge.cols, cost_fn, blocked)
        return path

    def score_cell(
        self,
        cell: Tuple[int, int],
        knowledge: AgentKnowledge,
        predictions: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    ) -> float:
        """Risk score for a cell.  Baseline always returns a uniform low score."""
        if knowledge.is_known_blocked(*cell):
            return 1.0
        return 0.0   # no risk differentiation in baseline


class RiskAwarePlanner:
    """
    Risk-aware planner: A* with mine probability + collision + unknown-penalty cost.

    Cost function (expected cost interpretation):
        cost(cell) = movement_cost
                   + mine_risk_weight   * P(mine | knowledge)
                   + collision_weight   * P(collision | predictions)
                   + unknown_weight     * is_unknown(cell)
                   + avoid_weight       * is_in_avoid_set(cell)

    The weights are manually chosen heuristics documented in PlannerConfig.
    They are NOT mathematically optimal values.  Their relative effect should
    be measured via controlled experiments (see experiments.py).

    Information sources:
        - P(mine) comes from AgentKnowledge.get_mine_probability()
        - P(collision) approximated as 1.0 if the cell is predicted occupied
          at step 1, else 0.0  (binary, deterministic trajectory prediction)
        - Terminology: this is short-horizon deterministic trajectory
          prediction, NOT probabilistic collision prediction.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or CONFIGS['risk_aware']

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        knowledge: AgentKnowledge,
        avoid_cells: Optional[Set[Tuple[int, int]]] = None,
        predictions: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Plan a path from start to goal using risk-weighted A*.

        Args:
            start      : current robot position
            goal       : target position
            knowledge  : agent's current knowledge state (NO ground truth)
            avoid_cells: cells temporarily flagged for avoidance (high penalty)
            predictions: step->set(cells) from predict_others(), used for
                         collision cost at step 1

        Returns:
            List of (r,c) tuples from start to goal inclusive, or [] if unreachable.
        """
        cfg = self.config
        avoid = avoid_cells or set()
        step1 = (predictions or {}).get(1, set())

        def blocked(r, c):
            return knowledge.is_known_blocked(r, c)

        def cost_fn(r, c):
            cell = (r, c)
            status = knowledge.get_status(r, c)

            # Base movement cost
            total = cfg.movement_cost

            # Mine risk (uses only agent knowledge probability)
            if status == CellStatus.OBSERVED_SAFE or status == CellStatus.INFERRED_SAFE:
                mine_p = 0.0
            elif status == CellStatus.OBSERVED_MINE or status == CellStatus.INFERRED_MINE:
                mine_p = 1.0
            else:
                mine_p = knowledge.get_mine_probability(r, c)
            total += cfg.mine_risk_weight * mine_p

            # Unknown-cell uncertainty penalty
            if status == CellStatus.UNKNOWN:
                total += cfg.unknown_weight

            # Collision risk: deterministic step-1 trajectory prediction
            # (binary: 0 or 1 — not a true probability)
            if cell in step1:
                total += cfg.collision_weight

            # Avoid-set penalty (used when replanning around a risky cell)
            if cell in avoid:
                total += cfg.avoid_weight

            return total

        path = astar(start, goal, knowledge.rows, knowledge.cols, cost_fn, blocked)
        return path

    def score_cell(
        self,
        cell: Tuple[int, int],
        knowledge: AgentKnowledge,
        predictions: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    ) -> float:
        """
        Compute a risk score in [0, 1] for moving into cell at the next step.

        This is used by the robot to decide whether to wait/replan before
        committing to the next move.

        Components:
            mine_component      : P(mine) from agent knowledge only
            collision_component : 1.0 if predicted occupied at step 1, else 0.0
            unknown_component   : 1.0 if cell is UNKNOWN, else 0.0

        The final score is a weighted sum normalised to [0,1].
        Weights reflect relative severity: mine > unknown > collision
        because a mine = immediate death, unknown = uncertain, collision = bad
        but robots currently respawn each round.

        NOTE: These weights are heuristics and should be treated as experimental
        parameters, not ground-truth optimal values.
        """
        cfg = self.config
        status = knowledge.get_status(*cell)

        if status == CellStatus.OBSERVED_WALL or status == CellStatus.INFERRED_MINE:
            return 1.0  # definitely impassable

        # Mine probability
        if status == CellStatus.OBSERVED_MINE:
            mine_p = 1.0
        elif status in (CellStatus.OBSERVED_SAFE, CellStatus.INFERRED_SAFE):
            mine_p = 0.0
        else:
            mine_p = knowledge.get_mine_probability(*cell)

        # Collision (binary deterministic prediction)
        step1 = (predictions or {}).get(1, set())
        collision_p = 1.0 if cell in step1 else 0.0

        # Unknown penalty
        unknown_penalty = 1.0 if status == CellStatus.UNKNOWN else 0.0

        # Weighted sum with explicit named weights:
        # mine is most dangerous, unknown second, collision third
        # (collision weight is kept small because avoiding all collisions
        #  can cause deadlocks; mine avoidance is life-critical)
        risk = (
            0.7 * mine_p
            + 0.1 * collision_p
            + 0.2 * unknown_penalty
        )
        return min(1.0, risk)


class ConservativeRiskPlanner(RiskAwarePlanner):
    """
    Conservative variant: like RiskAwarePlanner but strongly penalises unknowns.

    This planner will prefer longer routes through known-safe territory
    over shorter routes through uncharted cells.
    Used in Experiment C.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        super().__init__(config or CONFIGS['conservative'])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_planner(name: str, config: Optional[PlannerConfig] = None):
    """
    Create a planner by name.

    Args:
        name   : 'shortest', 'risk_aware', or 'conservative'
        config : override PlannerConfig; if None, uses the predefined defaults

    Returns:
        A planner object with .plan() and .score_cell() methods.
    """
    name = name.lower()
    if name == 'shortest':
        return ShortestPathPlanner(config or CONFIGS['shortest'])
    elif name == 'risk_aware':
        return RiskAwarePlanner(config or CONFIGS['risk_aware'])
    elif name == 'conservative':
        return ConservativeRiskPlanner(config or CONFIGS['conservative'])
    else:
        raise ValueError(f"Unknown planner name: {name!r}. "
                         f"Choose from: 'shortest', 'risk_aware', 'conservative'")
