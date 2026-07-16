"""
grid.py — GridWorld: terrain generation, mine placement, shared knowledge.

Generation pipeline:
  1. Start with all cells empty.
  2. Attempt to add wall cells one-by-one with BFS connectivity check (rejection sampling).
     Target: 12% wall density, but walls tend to cluster into corridors naturally.
  3. Scatter mines on ~8% of remaining passable cells.
  4. Pre-compute hidden mine-adjacency clue numbers for every empty cell.

Shared Knowledge (updated as robots explore):
  - known_cells: dict (r,c) -> 'empty' | 'wall' | 'mine'
  - known_clues: dict (r,c) -> int  (Minesweeper-style adjacency numbers)
  - mine_probability_cache: updated after each new discovery
"""

import random
import collections
from utils import is_fully_connected, bfs_all_reachable

CELL_UNKNOWN = 'unknown'
CELL_EMPTY   = 'empty'
CELL_WALL    = 'wall'
CELL_MINE    = 'mine'


class GridWorld:
    ROWS = 15
    COLS = 15

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.rows = self.ROWS
        self.cols = self.COLS

        # Hidden ground truth
        self.walls  = set()   # (r,c) cells that are walls
        self.mines  = set()   # (r,c) cells that are mines
        self._mine_clues = {} # (r,c) -> int (hidden adjacency count)

        # Shared knowledge — updated by discover()
        self.known_cells  = {}  # (r,c) -> 'empty' | 'wall' | 'mine'
        self.known_clues  = {}  # (r,c) -> int
        self._prob_cache  = {}  # (r,c) -> float, rebuilt on demand
        self._prob_dirty  = True

        self._generate_walls()
        self._place_mines()
        self._compute_clues()

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
        Tries to add walls to create natural bottlenecks and corridors.
        Each candidate is rejected if it disconnects the passability graph.
        """
        target = int(self.rows * self.cols * 0.15)
        all_cells = self._all_cells()

        # Shuffle candidate cells
        candidates = all_cells[:]
        random.shuffle(candidates)

        attempts = 0
        for cell in candidates:
            if len(self.walls) >= target:
                break
            if attempts > 2000:
                break
            attempts += 1

            r, c = cell
            self.walls.add(cell)
            passable = self._passable_set()
            if not is_fully_connected(passable, self.rows, self.cols):
                self.walls.discard(cell)  # reject — would disconnect grid

        # Verify connectivity one final time
        passable = self._passable_set()
        assert is_fully_connected(passable, self.rows, self.cols), \
            "Wall generation produced disconnected grid — should not happen."

    def _place_mines(self):
        passable = [cell for cell in self._all_cells() if cell not in self.walls]
        mine_count = max(4, int(len(passable) * 0.05))
        self.mines = set(random.sample(passable, min(mine_count, len(passable))))

    def _compute_clues(self):
        """Pre-compute mine-adjacency numbers for all non-mine, non-wall cells."""
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
    # Knowledge query (uses hidden ground truth)
    # ------------------------------------------------------------------

    def is_wall(self, r, c):
        return (r, c) in self.walls

    def is_mine(self, r, c):
        return (r, c) in self.mines

    def is_passable_true(self, r, c):
        """Ground truth — only used internally or for stats."""
        return (r, c) not in self.walls and (r, c) not in self.mines

    def get_cell_status(self, r, c):
        """Return known status of cell, or CELL_UNKNOWN."""
        return self.known_cells.get((r, c), CELL_UNKNOWN)

    # ------------------------------------------------------------------
    # Discovery  (called when a robot enters/scans a cell)
    # ------------------------------------------------------------------

    def discover(self, r, c):
        """
        Reveal cell (r,c) and its immediately adjacent walls to shared knowledge.
        Returns True if the cell is a mine (triggers explosion).
        """
        cell = (r, c)

        if (r, c) in self.walls:
            self.known_cells[cell] = CELL_WALL
            return False

        if (r, c) in self.mines:
            self.known_cells[cell] = CELL_MINE
            # Reveal clues for all 8 neighbours immediately
            for nr, nc in self._neighbors8(r, c):
                nb = (nr, nc)
                if nb not in self.walls and nb not in self.mines:
                    if nb in self._mine_clues:
                        self.known_clues[nb] = self._mine_clues[nb]
                    if nb not in self.known_cells:
                        pass  # don't auto-reveal, just flag the clue
            self._prob_dirty = True
            return True  # mine hit!

        # Normal empty cell
        self.known_cells[cell] = CELL_EMPTY
        if cell in self._mine_clues:
            self.known_clues[cell] = self._mine_clues[cell]

        # Line-of-sight: discover adjacent walls so robots don't walk into them
        for nr, nc in self._neighbors4(r, c):
            nb = (nr, nc)
            if nb in self.walls and nb not in self.known_cells:
                self.known_cells[nb] = CELL_WALL

        self._prob_dirty = True
        return False

    def scan_adjacent(self, r, c):
        """
        Scan all 4-neighbours of (r,c) and reveal their wall/mine status.
        Called at the start of each robot turn so they get good local info.
        """
        for nr, nc in self._neighbors4(r, c):
            nb = (nr, nc)
            if nb in self.walls and nb not in self.known_cells:
                self.known_cells[nb] = CELL_WALL
            # Do NOT automatically reveal mines — only when stepped on

    # ------------------------------------------------------------------
    # Mine probability estimation (Minesweeper-style)
    # ------------------------------------------------------------------

    def get_mine_probability(self, r, c):
        """
        Estimate probability that (r,c) is a mine using:
          - If cell is known → definitive answer.
          - If neighbour clues constrain it → use clue-based reasoning.
          - Otherwise → base rate: remaining_mines / unrevealed_cells.
        """
        cell = (r, c)
        status = self.known_cells.get(cell)
        if status == CELL_MINE:
            return 1.0
        if status in (CELL_WALL, CELL_EMPTY):
            return 0.0  # walls are blocked; empty cells are confirmed safe

        # Count known mines and unrevealed cells
        all_cells = self._all_cells()
        known_mines = sum(1 for s in self.known_cells.values() if s == CELL_MINE)
        remaining_mines = max(0, len(self.mines) - known_mines)
        unrevealed = [c2 for c2 in all_cells
                      if self.known_cells.get(c2, CELL_UNKNOWN) == CELL_UNKNOWN
                      and c2 not in self.walls]
        if not unrevealed:
            return 0.0

        base_prob = remaining_mines / len(unrevealed)

        # Refine using adjacent clue constraints
        # If any revealed neighbour has a clue and this cell is one of its unknowns:
        constraint_probs = []
        for nr, nc in self._neighbors8(r, c):
            nb = (nr, nc)
            clue = self.known_clues.get(nb)
            if clue is None:
                continue
            # Count how many unknown neighbours THIS clue cell has
            nb_unknowns = [
                (xr, xc) for xr, xc in self._neighbors8(nr, nc)
                if self.known_cells.get((xr, xc), CELL_UNKNOWN) == CELL_UNKNOWN
                and (xr, xc) not in self.walls
            ]
            nb_known_mines = sum(
                1 for xr, xc in self._neighbors8(nr, nc)
                if self.known_cells.get((xr, xc)) == CELL_MINE
            )
            mines_left = max(0, clue - nb_known_mines)
            if nb_unknowns:
                p = mines_left / len(nb_unknowns)
                constraint_probs.append(min(1.0, p))

        if constraint_probs:
            # Take max of constraint-based estimates (conservative)
            return max(base_prob, max(constraint_probs))

        return base_prob

    # ------------------------------------------------------------------
    # Helpers for robot starting positions
    # ------------------------------------------------------------------

    def random_passable_cells(self, n, exclude=None):
        """
        Return n random cells that are passable (not wall, not mine, not in exclude).
        """
        exclude = set(exclude or [])
        candidates = [
            cell for cell in self._all_cells()
            if cell not in self.walls and cell not in self.mines and cell not in exclude
        ]
        if len(candidates) < n:
            raise ValueError(f"Not enough passable cells ({len(candidates)}) to place {n} agents.")
        return random.sample(candidates, n)

    def count_mines(self):
        return len(self.mines)

    def count_walls(self):
        return len(self.walls)
