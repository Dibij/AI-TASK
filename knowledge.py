"""
knowledge.py — Explicit agent knowledge representation for the multi-agent grid world.

This module provides a clean separation between:

A. Ground-truth environment state (in GridWorld):
   - true walls, true mines, true clue counts, robot positions

B. Agent knowledge state (in AgentKnowledge):
   - cells directly observed (safe, wall, mine)
   - clue values observed by stepping on numbered cells
   - inferred cell status from logical constraint propagation
   - estimated mine probabilities from local clue constraints

Design notes:
    - All robots currently share one AgentKnowledge instance (cooperative shared knowledge).
      This is an intentional design choice modelling a robot swarm with a shared radio link.
      It is documented here so it is not mistaken for silent ground-truth leakage.
    - The probability model never accesses GridWorld.mines or GridWorld.walls.
      It reasons solely from the data in AgentKnowledge.
    - Ground-truth access is permitted ONLY in GridWorld itself (for physics/rendering)
      and in oracle evaluation metrics (clearly labelled in game.py / utils.py).
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Cell status enum
# ---------------------------------------------------------------------------

class CellStatus(Enum):
    """
    Every cell in the agent's world model has exactly one of these statuses.

    UNKNOWN        – robot has never seen this cell; anything could be there.
    OBSERVED_SAFE  – robot (or teammate) physically entered this cell and it was empty.
    OBSERVED_WALL  – adjacent scan or line-of-sight revealed this cell is a wall.
    OBSERVED_MINE  – robot stepped on a mine here (and died); cell known to be a mine.
    INFERRED_SAFE  – logical constraint propagation proves this cell cannot be a mine.
    INFERRED_MINE  – logical constraint propagation proves this cell must be a mine.
    """
    UNKNOWN        = auto()
    OBSERVED_SAFE  = auto()
    OBSERVED_WALL  = auto()
    OBSERVED_MINE  = auto()
    INFERRED_SAFE  = auto()
    INFERRED_MINE  = auto()


# Convenience aliases for readability
UNKNOWN       = CellStatus.UNKNOWN
OBSERVED_SAFE = CellStatus.OBSERVED_SAFE
OBSERVED_WALL = CellStatus.OBSERVED_WALL
OBSERVED_MINE = CellStatus.OBSERVED_MINE
INFERRED_SAFE = CellStatus.INFERRED_SAFE
INFERRED_MINE = CellStatus.INFERRED_MINE

# Cells the agent treats as definitely blocked for pathfinding purposes
BLOCKED_STATUSES = frozenset({OBSERVED_WALL, OBSERVED_MINE, INFERRED_MINE})

# Cells confirmed safe for movement (no mine risk)
SAFE_STATUSES = frozenset({OBSERVED_SAFE, INFERRED_SAFE})

# Cells that are unknown — may or may not be traversable
UNKNOWN_STATUSES = frozenset({UNKNOWN})


# ---------------------------------------------------------------------------
# AgentKnowledge
# ---------------------------------------------------------------------------

class AgentKnowledge:
    """
    Maintains everything the agent (or cooperative swarm) knows about the grid.

    This object is the ONLY source of information the planner and probability
    model should consult.  It must never receive direct references to
    GridWorld.mines or GridWorld.walls.

    Attributes:
        rows, cols     : grid dimensions (known at construction; not secret)
        cells          : dict (r,c) -> CellStatus
        clues          : dict (r,c) -> int  Minesweeper-style adjacency count
                         only populated when cell status is OBSERVED_SAFE
        mine_prior     : float, prior probability assigned to a completely
                         unknown cell when no clue information is available.
                         Default 0.05 (5%) which matches the world generation
                         density (~5% of passable cells are mines).
                         This is a configurable design parameter, NOT a hidden
                         mine count.  Set it to match world configuration if
                         the mine density is known to the robots as a "rule".
    """

    def __init__(self, rows: int, cols: int, mine_prior: float = 0.05):
        self.rows = rows
        self.cols = cols
        self.mine_prior: float = mine_prior  # heuristic prior, not hidden count

        # Core knowledge stores
        self.cells: Dict[Tuple[int, int], CellStatus] = {}
        self.clues: Dict[Tuple[int, int], int] = {}

        # Cache for mine probabilities (invalidated on any update)
        self._prob_cache: Dict[Tuple[int, int], float] = {}
        self._prob_dirty: bool = True

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        return [(r + dr, c + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if self._in_bounds(r + dr, c + dc)]

    def _neighbors8(self, r: int, c: int) -> List[Tuple[int, int]]:
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0) and self._in_bounds(r + dr, c + dc)]

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_status(self, r: int, c: int) -> CellStatus:
        return self.cells.get((r, c), UNKNOWN)

    def is_known_blocked(self, r: int, c: int) -> bool:
        """True if the agent knows this cell is definitely impassable."""
        return self.cells.get((r, c), UNKNOWN) in BLOCKED_STATUSES

    def is_known_safe(self, r: int, c: int) -> bool:
        """True if the agent knows this cell is safe (no mine)."""
        return self.cells.get((r, c), UNKNOWN) in SAFE_STATUSES

    def is_unknown(self, r: int, c: int) -> bool:
        return self.cells.get((r, c), UNKNOWN) == UNKNOWN

    # ------------------------------------------------------------------
    # Knowledge updates (called by GridWorld.discover / scan_adjacent)
    # ------------------------------------------------------------------

    def mark_observed_safe(self, r: int, c: int, clue: Optional[int] = None) -> None:
        """Record that cell (r,c) was entered and was safe."""
        cell = (r, c)
        # Only upgrade status — never downgrade an INFERRED_MINE to OBSERVED_SAFE
        # (that would indicate a logic error in the inference engine)
        if self.cells.get(cell) not in BLOCKED_STATUSES:
            self.cells[cell] = OBSERVED_SAFE
        if clue is not None:
            self.clues[cell] = clue
        self._prob_dirty = True

    def mark_observed_wall(self, r: int, c: int) -> None:
        """Record that cell (r,c) is a wall (seen by adjacent scan)."""
        self.cells[(r, c)] = OBSERVED_WALL
        self._prob_dirty = True

    def mark_observed_mine(self, r: int, c: int) -> None:
        """Record that cell (r,c) exploded — confirmed mine."""
        self.cells[(r, c)] = OBSERVED_MINE
        self._prob_dirty = True

    def mark_inferred_safe(self, r: int, c: int) -> None:
        """Logical deduction: this cell cannot be a mine."""
        if self.cells.get((r, c), UNKNOWN) == UNKNOWN:
            self.cells[(r, c)] = INFERRED_SAFE
            self._prob_dirty = True

    def mark_inferred_mine(self, r: int, c: int) -> None:
        """Logical deduction: this cell must be a mine."""
        if self.cells.get((r, c), UNKNOWN) in (UNKNOWN, INFERRED_SAFE):
            self.cells[(r, c)] = INFERRED_MINE
            self._prob_dirty = True

    # ------------------------------------------------------------------
    # Constraint propagation
    # ------------------------------------------------------------------

    def propagate_constraints(self) -> int:
        """
        Simple subset-constraint propagation (Minesweeper-style logic).

        For each pair of clue cells A and B where the unknown-neighbour set of A
        is a subset of the unknown-neighbour set of B:

            unknowns_A ⊆ unknowns_B
            clue_A - known_mines_A == clue_B - known_mines_B - len(unknowns_B \ unknowns_A)

        Deductions:
            If remaining_A == len(unknowns_A): every unknown of A is a mine.
            If remaining_A == 0: every unknown of A is safe.
            If remaining_B - remaining_A == len(unknowns_B) - len(unknowns_A):
                all cells in unknowns_B \ unknowns_A are mines.
            If remaining_B - remaining_A == 0:
                all cells in unknowns_B \ unknowns_A are safe.

        Returns the number of new inferences made.
        """
        new_inferences = 0

        def _unknowns(r, c):
            return frozenset(
                nb for nb in self._neighbors8(r, c)
                if self.cells.get(nb, UNKNOWN) == UNKNOWN
            )

        def _known_mine_count(r, c):
            return sum(
                1 for nb in self._neighbors8(r, c)
                if self.cells.get(nb) in (OBSERVED_MINE, INFERRED_MINE)
            )

        clue_cells = [(r, c) for (r, c), v in self.clues.items()
                      if self.cells.get((r, c)) == OBSERVED_SAFE]

        # Build constraint list: (remaining_mines, frozenset of unknowns)
        constraints = []
        for (r, c) in clue_cells:
            remaining = self.clues[(r, c)] - _known_mine_count(r, c)
            unknowns = _unknowns(r, c)
            if unknowns:  # only add if there are still unknowns to constrain
                constraints.append((remaining, unknowns))

        # Single-constraint deductions
        for remaining, unknowns in constraints:
            if remaining < 0:
                continue  # clue already satisfied (can happen if inference is wrong)
            if remaining == len(unknowns):
                # All unknowns must be mines
                for cell in unknowns:
                    if self.cells.get(cell, UNKNOWN) == UNKNOWN:
                        self.mark_inferred_mine(*cell)
                        new_inferences += 1
            elif remaining == 0:
                # All unknowns are safe
                for cell in unknowns:
                    if self.cells.get(cell, UNKNOWN) == UNKNOWN:
                        self.mark_inferred_safe(*cell)
                        new_inferences += 1

        # Subset constraint propagation: A ⊆ B => B\A gives extra deductions
        for i in range(len(constraints)):
            rem_a, unk_a = constraints[i]
            for j in range(len(constraints)):
                if i == j:
                    continue
                rem_b, unk_b = constraints[j]
                if unk_a < unk_b:  # strict subset
                    diff = unk_b - unk_a
                    diff_rem = rem_b - rem_a
                    if diff_rem == len(diff):
                        # All cells in diff are mines
                        for cell in diff:
                            if self.cells.get(cell, UNKNOWN) == UNKNOWN:
                                self.mark_inferred_mine(*cell)
                                new_inferences += 1
                    elif diff_rem == 0:
                        # All cells in diff are safe
                        for cell in diff:
                            if self.cells.get(cell, UNKNOWN) == UNKNOWN:
                                self.mark_inferred_safe(*cell)
                                new_inferences += 1

        if new_inferences > 0:
            self._prob_dirty = True
        return new_inferences

    # ------------------------------------------------------------------
    # Mine probability estimation
    # ------------------------------------------------------------------

    def get_mine_probability(self, r: int, c: int) -> float:
        """
        Estimate P(cell (r,c) contains a mine | current agent knowledge).

        Uses ONLY information in this AgentKnowledge object.
        Never accesses GridWorld ground truth.

        Algorithm:
          1. If cell status is known definitively → return 0.0 or 1.0.
          2. Collect local clue constraints from 8-adjacent observed cells.
          3. For each clue: p_i = (clue - known_mines) / |unknown_neighbours|
             This is the fraction of remaining mines assigned uniformly over
             this clue's unknown neighbourhood.  This is a LOCAL approximation,
             not exact Bayesian inference.
          4. Final estimate = max(mine_prior, max(constraint_probs))
             (conservative: take the worst-case local constraint estimate)

        Limitation:
            This is a local heuristic approximation.  It does NOT implement
            full exact Minesweeper inference (which would require enumeration
            of all satisfying mine assignments).  Constraint propagation
            (propagate_constraints) provides additional logical deductions
            but does not replace a full solver.

        Returns float in [0.0, 1.0].
        """
        cell = (r, c)
        status = self.cells.get(cell, UNKNOWN)

        # Definitive answers
        if status in (OBSERVED_MINE, INFERRED_MINE):
            return 1.0
        if status in (OBSERVED_SAFE, OBSERVED_WALL, INFERRED_SAFE):
            return 0.0

        # Gather constraint-based estimates from neighbouring clue cells
        constraint_probs: List[float] = []
        for nr, nc in self._neighbors8(r, c):
            nb = (nr, nc)
            clue = self.clues.get(nb)
            if clue is None:
                continue
            if self.cells.get(nb) != OBSERVED_SAFE:
                continue  # only use clues from confirmed safe cells

            # Count already-known mines in the clue's neighbourhood
            nb_known_mines = sum(
                1 for xr, xc in self._neighbors8(nr, nc)
                if self.cells.get((xr, xc)) in (OBSERVED_MINE, INFERRED_MINE)
            )
            # Count remaining unknowns in the clue's neighbourhood
            nb_unknowns = [
                (xr, xc) for xr, xc in self._neighbors8(nr, nc)
                if self.cells.get((xr, xc), UNKNOWN) == UNKNOWN
            ]
            mines_left = max(0, clue - nb_known_mines)
            if nb_unknowns:
                p = mines_left / len(nb_unknowns)
                constraint_probs.append(min(1.0, p))

        if constraint_probs:
            # Conservative estimate: take the maximum local constraint
            return max(self.mine_prior, max(constraint_probs))

        return self.mine_prior  # no local clue — fall back to prior

    # ------------------------------------------------------------------
    # Helpers for planners
    # ------------------------------------------------------------------

    def all_unknown_cells(self) -> List[Tuple[int, int]]:
        return [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.cells.get((r, c), UNKNOWN) == UNKNOWN
        ]

    def known_safe_cells(self) -> List[Tuple[int, int]]:
        return [cell for cell, st in self.cells.items() if st in SAFE_STATUSES]

    def known_mine_cells(self) -> List[Tuple[int, int]]:
        return [cell for cell, st in self.cells.items()
                if st in (OBSERVED_MINE, INFERRED_MINE)]

    def known_wall_cells(self) -> List[Tuple[int, int]]:
        return [cell for cell, st in self.cells.items() if st == OBSERVED_WALL]

    def summary(self) -> str:
        counts = {s: 0 for s in CellStatus}
        for st in self.cells.values():
            counts[st] += 1
        unknown_total = self.rows * self.cols - len(self.cells)
        return (
            f"AgentKnowledge({self.rows}x{self.cols}): "
            f"unknown={unknown_total} "
            f"safe={counts[OBSERVED_SAFE]+counts[INFERRED_SAFE]} "
            f"wall={counts[OBSERVED_WALL]} "
            f"mine={counts[OBSERVED_MINE]+counts[INFERRED_MINE]}"
        )
