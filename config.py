"""
config.py — Central configuration for the multi-agent grid-world system.

All configurable parameters live here, eliminating hardcoded magic numbers.
Import and use these values throughout the codebase instead of local constants.

Categories:
    RiskConfig      — Risk values for cell statuses (mine probability mapping)
    SimConfig       — Simulation parameters (tick limits, timeouts, horizons)
    ConflictPolicy  — Conflict resolution strategies
    FairnessConfig  — Starvation prevention parameters
    ExperimentConfig— Experiment runner defaults
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


# ---------------------------------------------------------------------------
# Risk Configuration — explicit risk values per cell status
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Explicit risk values for different cell knowledge states.

    These map cell statuses to risk estimates used by planners.
    All values are in [0.0, 1.0].

    The distinction between categories is important:
        KNOWN_MINE    : cell confirmed as mine (observed or inferred) → maximum risk
        KNOWN_SAFE    : cell confirmed safe (observed or inferred) → zero risk
        INFERRED_MINE : logical deduction says mine → very high risk
        INFERRED_SAFE : logical deduction says safe → very low risk
        UNKNOWN       : no information → estimated from prior or context
    """
    RISK_KNOWN_MINE    : float = 1.0
    RISK_KNOWN_SAFE    : float = 0.0
    RISK_INFERRED_MINE : float = 0.95
    RISK_INFERRED_SAFE : float = 0.02
    RISK_UNKNOWN       : float = 0.05   # default prior, matches ~5% mine density
    RISK_OBSERVED_WALL : float = 0.0    # walls are blocked, not risky


# ---------------------------------------------------------------------------
# Conflict Resolution Policy
# ---------------------------------------------------------------------------

class ConflictPolicy(Enum):
    """
    Determines how same-cell and swap conflicts are resolved.

    BY_ROBOT_ID          : lower robot ID gets priority (deterministic, simple)
    BY_REMAINING_DISTANCE: robot closer to goal gets priority (progress-oriented)
    BY_WAIT_TIME         : robot that has waited longer gets priority (fairness)
    ROTATING             : round-robin priority based on tick number
    """
    BY_ROBOT_ID           = auto()
    BY_REMAINING_DISTANCE = auto()
    BY_WAIT_TIME          = auto()
    ROTATING              = auto()


# ---------------------------------------------------------------------------
# Fairness / Starvation Prevention
# ---------------------------------------------------------------------------

@dataclass
class FairnessConfig:
    """
    Parameters for starvation prevention.

    A robot that repeatedly loses conflicts will have its priority boosted
    after losing too many consecutive conflicts.
    """
    max_consecutive_losses : int   = 5     # boost priority after this many losses
    priority_boost         : float = 100.0 # amount to boost priority score
    decay_rate             : float = 0.5   # boost decays by this factor each tick with movement


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Core simulation parameters.

    These control the physics of the simulation, not the AI strategy.
    """
    MAX_TICKS                   : int   = 300   # safety timeout per round
    MAX_CONSECUTIVE_IDLE_TICKS  : int   = 10    # idle-timeout threshold
    COLLISION_PREDICTION_HORIZON: int   = 5     # how many steps ahead to predict collisions
    GRID_ROWS                   : int   = 15
    GRID_COLS                   : int   = 15
    MINE_DENSITY                : float = 0.05  # ~5% of passable cells
    WALL_DENSITY                : float = 0.15  # ~15% of all cells
    MIN_MINES                   : int   = 4     # minimum mines regardless of density
    CONFLICT_POLICY             : ConflictPolicy = ConflictPolicy.BY_WAIT_TIME
    FAIRNESS                    : FairnessConfig = field(default_factory=FairnessConfig)


# ---------------------------------------------------------------------------
# Risk Relaxation Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskRelaxationConfig:
    """
    Parameters for risk-tolerance relaxation when a robot is stuck.

    If no path exists under the current risk threshold:
        1. Wait for `wait_before_relax` ticks.
        2. Relax threshold by `relax_step`.
        3. Log the relaxation event.
        4. Repeat up to `max_relaxations` times.
    """
    wait_before_relax : int   = 3     # ticks to wait before relaxing
    relax_step        : float = 0.1   # amount to relax risk tolerance
    max_relaxations   : int   = 8     # maximum number of relaxation steps
    max_tolerance     : float = 1.0   # ceiling for relaxed tolerance


# ---------------------------------------------------------------------------
# Collision Prediction Decay
# ---------------------------------------------------------------------------

@dataclass
class CollisionDecayConfig:
    """
    Time-decaying collision penalty across prediction horizon.

    step 1: penalty * decay_factors[0]
    step 2: penalty * decay_factors[1]
    ...

    The decay factors control how much future collision risk matters
    relative to immediate collision risk.
    """
    decay_factors: List[float] = field(
        default_factory=lambda: [1.0, 0.7, 0.4, 0.2, 0.1]
    )

    def get_factor(self, step: int) -> float:
        """Get decay factor for a given prediction step (1-indexed)."""
        idx = step - 1
        if 0 <= idx < len(self.decay_factors):
            return self.decay_factors[idx]
        return 0.0  # beyond horizon


# ---------------------------------------------------------------------------
# Experiment Runner Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Default configuration for the overnight experiment runner.
    """
    default_num_seeds     : int  = 30    # seeds per configuration
    default_num_robots    : int  = 4
    default_max_rounds    : int  = 10    # rounds per seed (kept for Game compat)
    checkpoint_interval   : int  = 1     # save checkpoint every N completed runs
    risk_tolerance_sweep  : List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9]
    )


# ---------------------------------------------------------------------------
# Global default instances
# ---------------------------------------------------------------------------

DEFAULT_RISK_CONFIG        = RiskConfig()
DEFAULT_SIM_CONFIG         = SimConfig()
DEFAULT_RISK_RELAXATION    = RiskRelaxationConfig()
DEFAULT_COLLISION_DECAY    = CollisionDecayConfig()
DEFAULT_EXPERIMENT_CONFIG  = ExperimentConfig()
