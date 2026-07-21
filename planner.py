"""
planner.py — Configurable planners for the multi-agent grid world.

Five distinct planning strategies are implemented for ablation study:

  ShortestPathPlanner (CONFIG A — BASELINE):
      Uses A* with uniform cost=1 for every traversable cell.
      Treats all unknown cells as traversable with normal cost.
      Does NOT use mine-risk information, unknown penalty, or collision avoidance.
      Purpose: experimental baseline for comparison.

  RiskAwarePlanner (CONFIG B — FULL RISK-AWARE):
      Uses A* with a composite cost function:
          cost = movement_cost
               + mine_risk_weight   * P(mine)
               + unknown_weight     * unknown_penalty
               + collision_weight   * collision_prediction_penalty
               + avoid_weight       * is_in_avoid_set
      Weights are configurable named parameters, not magic numbers.
      Uses only AgentKnowledge — no ground-truth access.

  RiskAwareNoCollisionPlanner (CONFIG C — NO COLLISION PREDICTION):
      Like RiskAwarePlanner but with collision_weight = 0.
      Allows evaluation of whether collision prediction contributes to performance.

  RiskAwareNoHazardPlanner (CONFIG D — NO HAZARD REASONING):
      Uses movement cost + collision prediction only.
      Disables mine-risk and unknown-cell penalties.
      Helps isolate the effect of hazard inference.

  FullSystemPlanner (CONFIG E — FULL SYSTEM):
      RiskAwarePlanner with all features enabled, plus conservative unknown weight.
      Knowledge representation + hazard inference + risk-aware A*
      + unknown-cell penalty + collision prediction + fair conflict resolution.

All planners use the SAME A* implementation from utils.py.
The blocked() predicate and cost_fn() are parameterised per planner.
This makes them easy to swap for experiments.

Cost interpretation (for RiskAwarePlanner):
    expected_cost = movement_cost
                  + P(mine)      * mine_risk_weight
                  + P(collision) * collision_weight  (decayed over horizon)
                  + unknown_penalty (if cell is unknown)

    Weights are manually chosen heuristics, NOT mathematically optimal values.
    Their effect should be measured experimentally (see experiments.py).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple, List

from utils import astar
from knowledge import AgentKnowledge, BLOCKED_STATUSES, CellStatus
from config import (
    DEFAULT_RISK_CONFIG, DEFAULT_COLLISION_DECAY,
    CollisionDecayConfig, RiskConfig,
)


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
        movement_cost         : base cost for stepping into any cell (default 1.0)
        mine_risk_weight      : multiplier on P(mine) for risk-aware planning
        unknown_weight        : flat penalty for stepping into an UNKNOWN cell
        collision_weight      : multiplier on collision prediction penalty
        avoid_weight          : large penalty for cells in the avoid-set
        mine_penalty          : expected-loss weight per mine event
        collision_penalty     : expected-loss weight per collision event
        planner_type          : one of the registered planner names
        collision_horizon     : how many prediction steps to use in cost
        score_mine_weight     : weight on mine probability in score_cell
        score_collision_weight: weight on collision prediction in score_cell
        score_unknown_weight  : weight on unknown status in score_cell
        collision_decay       : time-decaying factors for collision prediction
        risk_config           : explicit risk values for cell statuses
    """
    movement_cost         : float = 1.0
    mine_risk_weight      : float = 3.0
    unknown_weight        : float = 0.2
    collision_weight      : float = 0.5
    avoid_weight          : float = 100.0
    mine_penalty          : float = 10.0   # used in expected_cost calculation
    collision_penalty     : float = 5.0    # used in expected_cost calculation
    planner_type          : str   = 'risk_aware'
    collision_horizon     : int   = 5      # steps to look ahead for collisions
    # score_cell weights (previously hardcoded as 0.7/0.1/0.2)
    score_mine_weight     : float = 0.7
    score_collision_weight: float = 0.1
    score_unknown_weight  : float = 0.2
    # Collision decay and risk config
    collision_decay       : CollisionDecayConfig = field(default_factory=CollisionDecayConfig)
    risk_config           : RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict:
        """Serialize to dict for experiment metadata logging."""
        return {
            'movement_cost': self.movement_cost,
            'mine_risk_weight': self.mine_risk_weight,
            'unknown_weight': self.unknown_weight,
            'collision_weight': self.collision_weight,
            'avoid_weight': self.avoid_weight,
            'mine_penalty': self.mine_penalty,
            'collision_penalty': self.collision_penalty,
            'planner_type': self.planner_type,
            'collision_horizon': self.collision_horizon,
            'score_mine_weight': self.score_mine_weight,
            'score_collision_weight': self.score_collision_weight,
            'score_unknown_weight': self.score_unknown_weight,
            'collision_decay_factors': self.collision_decay.decay_factors,
        }


# Predefined configurations for experiments (A–E ablation study)
CONFIGS = {
    # A — Shortest-Path Baseline: uniform cost, ignores all risk
    'shortest': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=0.0,    # ignores mine risk
        unknown_weight=0.0,      # treats unknowns as normal
        collision_weight=0.0,    # ignores collision risk
        avoid_weight=0.0,
        planner_type='shortest',
        score_mine_weight=0.0,
        score_collision_weight=0.0,
        score_unknown_weight=0.0,
    ),
    # B — Risk-Aware A*: full composite cost
    'risk_aware': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=3.0,
        unknown_weight=0.2,
        collision_weight=0.5,
        avoid_weight=100.0,
        planner_type='risk_aware',
    ),
    # C — Risk-Aware WITHOUT collision prediction
    'risk_no_collision': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=3.0,
        unknown_weight=0.2,
        collision_weight=0.0,   # disabled
        avoid_weight=100.0,
        planner_type='risk_no_collision',
        score_collision_weight=0.0,
    ),
    # D — Risk-Aware WITHOUT hazard reasoning (movement + collision only)
    'risk_no_hazard': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=0.0,   # disabled
        unknown_weight=0.0,     # disabled
        collision_weight=0.5,
        avoid_weight=100.0,
        planner_type='risk_no_hazard',
        score_mine_weight=0.0,
        score_unknown_weight=0.0,
    ),
    # E — Full System: all components enabled with moderate conservatism
    'full_system': PlannerConfig(
        movement_cost=1.0,
        mine_risk_weight=3.0,
        unknown_weight=0.5,      # moderate unknown penalty
        collision_weight=0.5,
        avoid_weight=100.0,
        planner_type='full_system',
    ),
    # Legacy: conservative (kept for backward compatibility)
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
                   + collision_weight   * collision_prediction_penalty(cell)
                   + unknown_weight     * is_unknown(cell)
                   + avoid_weight       * is_in_avoid_set(cell)

    The collision prediction penalty uses a time-decaying scheme:
        For each prediction step within the horizon, if the cell is predicted
        occupied, the penalty is: collision_weight * decay_factor(step).
        The total is the sum across all steps.

    The weights are manually chosen heuristics documented in PlannerConfig.
    They are NOT mathematically optimal values.  Their relative effect should
    be measured via controlled experiments (see experiments.py).

    Information sources:
        - P(mine) comes from AgentKnowledge.get_mine_probability()
        - Collision predictions from predict_others() (deterministic trajectory)
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
                         collision cost across prediction horizon

        Returns:
            List of (r,c) tuples from start to goal inclusive, or [] if unreachable.
        """
        cfg = self.config
        avoid = avoid_cells or set()
        preds = predictions or {}

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

            # Collision risk: time-decaying across prediction horizon
            collision_penalty = 0.0
            horizon = min(cfg.collision_horizon, len(cfg.collision_decay.decay_factors))
            for step in range(1, horizon + 1):
                step_cells = preds.get(step, set())
                if cell in step_cells:
                    collision_penalty += cfg.collision_decay.get_factor(step)
            total += cfg.collision_weight * collision_penalty

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
            collision_component : decayed prediction across horizon
            unknown_component   : 1.0 if cell is UNKNOWN, else 0.0

        The final score is a weighted sum normalised to [0,1].
        Weights are configurable via PlannerConfig (score_mine_weight etc.)

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

        # Collision prediction (decayed across horizon)
        preds = predictions or {}
        collision_p = 0.0
        horizon = min(cfg.collision_horizon, len(cfg.collision_decay.decay_factors))
        for step in range(1, horizon + 1):
            step_cells = preds.get(step, set())
            if cell in step_cells:
                collision_p += cfg.collision_decay.get_factor(step)
        collision_p = min(1.0, collision_p)  # cap at 1.0

        # Unknown penalty
        unknown_penalty = 1.0 if status == CellStatus.UNKNOWN else 0.0

        # Weighted sum with configurable named weights
        risk = (
            cfg.score_mine_weight * mine_p
            + cfg.score_collision_weight * collision_p
            + cfg.score_unknown_weight * unknown_penalty
        )
        return min(1.0, risk)


class ConservativeRiskPlanner(RiskAwarePlanner):
    """
    Conservative variant: like RiskAwarePlanner but strongly penalises unknowns.

    This planner will prefer longer routes through known-safe territory
    over shorter routes through uncharted cells.
    Kept for backward compatibility.
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
        name   : planner type name (see CONFIGS keys)
        config : override PlannerConfig; if None, uses the predefined defaults

    Returns:
        A planner object with .plan() and .score_cell() methods.
    """
    name = name.lower()
    if name == 'shortest':
        return ShortestPathPlanner(config or CONFIGS['shortest'])
    elif name in ('risk_aware', 'risk_no_collision', 'risk_no_hazard', 'full_system'):
        return RiskAwarePlanner(config or CONFIGS.get(name, CONFIGS['risk_aware']))
    elif name == 'conservative':
        return ConservativeRiskPlanner(config or CONFIGS['conservative'])
    else:
        raise ValueError(f"Unknown planner name: {name!r}. "
                         f"Choose from: {list(CONFIGS.keys())}")
