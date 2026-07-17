"""
grid.py — GridWorld: terrain generation (ground truth only).

ARCHITECTURE NOTE — Ground Truth vs. Agent Knowledge
=====================================================
GridWorld holds and manages ONLY ground-truth state:
  - self.walls      : set of true wall cells
  - self.mines      : set of true mine cells
  - self._mine_clues: hidden adjacency counts per empty cell

Agent-observable state lives in AgentKnowledge (see knowledge.py).
GridWorld exposes a single AgentKnowledge instance (self.knowledge)
that is updated whenever a robot discovers or scans a cell.

ALL ROBOTS SHARE ONE AgentKnowledge INSTANCE.
This models a cooperative swarm with shared radio communication.
This is an intentional design decision, not accidental ground-truth leakage.
If independent per-robot knowledge is desired, each Robot would hold its own
AgentKnowledge and only update it via its own discover() calls.

Ground-truth methods (is_wall, is_mine, is_passable_true) must NOT be called
from robot decision logic, planners, or the probability model.
They may only be called by:
  - The physics engine (does this step kill the robot? → discover())
  - The renderer (dimly showing hidden mines for debugging/display)
  - Oracle evaluation metrics (labelled [ORACLE] in code comments)

Generation pipeline:
  1. Start with all cells empty.
  2. Attempt to add wall cells with BFS connectivity check (rejection sampling).
     Target: ~15% wall density.
  3. Scatter mines on ~5% of remaining passable cells.
  4. Pre-compute hidden mine-adjacency clue numbers for every empty cell.
"""

import random
import collections
from typing import Optional, Set, Tuple

from utils import is_fully_connected, bfs_all_reachable
from knowledge import AgentKnowledge, CellStatus


# ---------------------------------------------------------------------------
# String constants kept for backwards compatibility with renderer
# (main.py uses these to draw cells by string status from known_cells)
# ---------------------------------------------------------------------------

CELL_UNKNOWN = 'unknown'
CELL_EMPTY   = 'empty'
CELL_WALL    = 'wall'
CELL_MINE    = 'mine'

# Mapping from CellStatus enum → string for renderer compatibility
_STATUS_TO_STR = {
    CellStatus.UNKNOWN       : CELL_UNKNOWN,
    CellStatus.OBSERVED_SAFE : CELL_EMPTY,
    CellStatus.OBSERVED_WALL : CELL_WALL,
    CellStatus.OBSERVED_MINE : CELL_MINE,
    CellStatus.INFERRED_SAFE : CELL_EMPTY,   # treat as empty for rendering
    CellStatus.INFERRED_MINE : CELL_MINE,    # treat as mine for rendering
}


class GridWorld:
    """
    Ground-truth grid environment.

    Public attributes (ground truth — DO NOT READ from robot logic):
        walls       : set of (r,c) cells that are walls
        mines       : set of (r,c) cells that are mines
        rows, cols  : grid dimensions

    Public attributes (agent knowledge — safe to use everywhere):
        knowledge   : AgentKnowledge instance shared by all robots

    Backward-compatible properties (used by renderer / game.py):
        known_cells : dict view of knowledge.cells as legacy string values
        known_clues : dict view of knowledge.clues
    """

    ROWS = 15
    COLS = 15

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.rows = self.ROWS
        self.cols = self.COLS

        # ── Ground truth (NEVER accessed by robot decision logic) ──
        self.walls: Set[Tuple[int, int]] = set()
        self.mines: Set[Tuple[int, int]] = set()
        self._mine_clues = {}  # (r,c) -> int, hidden adjacency count

        # ── Agent knowledge (the ONLY source for robot reasoning) ──
        # mine_prior = 0.05 matches the ~5% mine density used in _place_mines().
        # This is an explicit configurable environment rule, not a hidden count.
        self.knowledge = AgentKnowledge(self.rows, self.cols, mine_prior=0.05)

        # Trail rendering support (populated during simulation)
        self.visited = collections.defaultdict(set)  # (r,c) -> set of robot colors

        self._generate_walls()
        self._place_mines()
        self._compute_clues()

    # ------------------------------------------------------------------
    # Backward-compatible properties for renderer / game.py
    # ------------------------------------------------------------------

    @property
    def known_cells(self):
        """
        Dict view: (r,c) -> legacy string status.
        Used by main.py renderer which predates the CellStatus enum.
        Always reflects the AgentKnowledge state.
        """
        return {cell: _STATUS_TO_STR[st]
                for cell, st in self.knowledge.cells.items()}

    @property
    def known_clues(self):
        """Dict view of knowledge.clues for renderer compatibility."""
        return self.knowledge.clues

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _all_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def _passable_set(self):
        return {cell for cell in self._all_cells() if cell not in self.walls}

    def _generate_walls(self):
        """
        Rejection-sampling wall placement.
        Each candidate wall is rejected if it would disconnect the passability graph.
        Target: ~15% wall density, creating natural corridors.
        """
        target = int(self.rows * self.cols * 0.15)
        candidates = self._all_cells()[:]
        random.shuffle(candidates)

        attempts = 0
        for cell in candidates:
            if len(self.walls) >= target:
                break
            if attempts > 2000:
                break
            attempts += 1

            self.walls.add(cell)
            passable = self._passable_set()
            if not is_fully_connected(passable, self.rows, self.cols):
                self.walls.discard(cell)  # reject — would disconnect grid

        passable = self._passable_set()
        assert is_fully_connected(passable, self.rows, self.cols), \
            "Wall generation produced disconnected grid — should not happen."

    def _place_mines(self):
        """
        Place mines on ~5% of passable cells.
        Mine density is an explicit environment parameter (not a secret).
        Robots are given the mine_prior=0.05 in AgentKnowledge which matches this.
        """
        passable = [cell for cell in self._all_cells() if cell not in self.walls]
        mine_count = max(4, int(len(passable) * 0.05))
        self.mines = set(random.sample(passable, min(mine_count, len(passable))))

    def _compute_clues(self):
        """Pre-compute hidden mine-adjacency numbers for all non-mine, non-wall cells."""
        self._mine_clues = {}
        for r, c in self._all_cells():
            if (r, c) in self.walls or (r, c) in self.mines:
                continue
            count = sum(
                1 for nr, nc in self._neighbors8(r, c)
                if (nr, nc) in self.mines
            )
            if count > 0:
                self._mine_clues[(r, c)] = count

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _neighbors4(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield (nr, nc)

    def _neighbors8(self, r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield (nr, nc)

    def neighbors4(self, r, c):
        return list(self._neighbors4(r, c))

    # ------------------------------------------------------------------
    # Ground-truth queries — ONLY for physics/rendering/oracle metrics
    # ------------------------------------------------------------------

    def is_wall(self, r, c) -> bool:
        """[GROUND TRUTH] True if (r,c) is a wall. DO NOT call from robot logic."""
        return (r, c) in self.walls

    def is_mine(self, r, c) -> bool:
        """[GROUND TRUTH] True if (r,c) is a mine. DO NOT call from robot logic."""
        return (r, c) in self.mines

    def is_passable_true(self, r, c) -> bool:
        """[GROUND TRUTH] True if cell is passable. Use only for stats/oracle."""
        return (r, c) not in self.walls and (r, c) not in self.mines

    def get_cell_status(self, r, c) -> str:
        """Legacy string status for a cell.  Delegates to AgentKnowledge."""
        st = self.knowledge.get_status(r, c)
        return _STATUS_TO_STR[st]

    # ------------------------------------------------------------------
    # Discovery — physics engine updates AgentKnowledge here
    # ------------------------------------------------------------------

    def discover(self, r, c) -> bool:
        """
        Reveal what is at cell (r,c) when a robot physically enters it.

        This method is allowed to read ground truth (self.walls, self.mines)
        because it is the physics simulation — it determines what actually
        happens when a robot steps on a cell.

        Updates AgentKnowledge accordingly.

        Returns:
            True  if the cell is a mine (robot is killed).
            False otherwise.
        """
        cell = (r, c)

        if (r, c) in self.walls:
            # Robots shouldn't enter walls, but handle it gracefully
            self.knowledge.mark_observed_wall(r, c)
            return False

        if (r, c) in self.mines:
            self.knowledge.mark_observed_mine(r, c)
            # Reveal clue numbers for all 8 neighbours (standard Minesweeper rule)
            for nr, nc in self._neighbors8(r, c):
                nb = (nr, nc)
                if nb not in self.walls and nb not in self.mines:
                    if nb in self._mine_clues:
                        self.knowledge.clues[nb] = self._mine_clues[nb]
            # Run constraint propagation in case we can deduce new info
            self.knowledge.propagate_constraints()
            return True  # mine hit — robot dies

        # Normal safe cell
        clue = self._mine_clues.get(cell)
        self.knowledge.mark_observed_safe(r, c, clue=clue)

        # Line-of-sight: reveal adjacent walls so robots don't walk into them
        for nr, nc in self._neighbors4(r, c):
            nb = (nr, nc)
            if nb in self.walls:
                self.knowledge.mark_observed_wall(nr, nc)

        # Run constraint propagation after each discovery
        self.knowledge.propagate_constraints()
        return False

    def scan_adjacent(self, r, c):
        """
        Scan 4-neighbours of (r,c) and reveal their wall status.
        Called at start of each robot turn for local awareness.
        Does NOT reveal mines — only physical entry reveals a mine.
        """
        for nr, nc in self._neighbors4(r, c):
            if (nr, nc) in self.walls:
                self.knowledge.mark_observed_wall(nr, nc)

    # ------------------------------------------------------------------
    # Mine probability (delegates to AgentKnowledge)
    # ------------------------------------------------------------------

    def get_mine_probability(self, r, c) -> float:
        """
        Estimate P(mine at (r,c) | agent knowledge).

        Delegates entirely to AgentKnowledge.get_mine_probability().
        No ground-truth information is used here.
        """
        return self.knowledge.get_mine_probability(r, c)

    # ------------------------------------------------------------------
    # Robot starting positions
    # ------------------------------------------------------------------

    def random_passable_cells(self, n, exclude=None):
        """
        Return n random cells that are passable (not wall, not mine, not excluded).
        [GROUND TRUTH] — used only for placing robots at round start, not for AI.
        """
        exclude = set(exclude or [])
        candidates = [
            cell for cell in self._all_cells()
            if cell not in self.walls and cell not in self.mines and cell not in exclude
        ]
        if len(candidates) < n:
            raise ValueError(
                f"Not enough passable cells ({len(candidates)}) to place {n} agents."
            )
        return random.sample(candidates, n)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def count_mines(self) -> int:
        return len(self.mines)

    def count_walls(self) -> int:
        return len(self.walls)
