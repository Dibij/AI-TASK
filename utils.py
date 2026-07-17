"""
utils.py — Core algorithmic helpers.

Contents:
  astar()                       : A* pathfinding with pluggable cost/blocked functions
  bfs_all_reachable()           : BFS reachability from a start cell
  is_fully_connected()          : connectivity check for grid validation
  shortest_path_length()        : [ORACLE] ground-truth BFS for efficiency metrics
  shortest_path_length_knowledge(): knowledge-based BFS for agent planning
"""

import heapq
import collections
from typing import Callable, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------

def astar(start: Tuple[int, int], goal: Tuple[int, int],
          rows: int, cols: int,
          cost_fn: Callable, blocked_fn: Callable):
    """
    A* pathfinder on a 4-connected grid.

    Args:
        start      : (r, c) start position
        goal       : (r, c) goal position
        rows/cols  : grid dimensions
        cost_fn    : (r, c) -> numeric cost to ENTER cell (called only on non-blocked cells)
        blocked_fn : (r, c) -> bool  True = cannot enter this cell

    Returns:
        List of (r, c) tuples from start to goal (inclusive), or [] if unreachable.

    Notes:
        - The heuristic is Manhattan distance (admissible for 4-connectivity).
        - cost_fn is responsible for all penalties (mine risk, unknown penalty,
          avoid-set penalty, collision penalty).  The planner supplies cost_fn.
        - blocked_fn should return True for OBSERVED_WALL, OBSERVED_MINE,
          and INFERRED_MINE cells.  Unknown cells are NOT blocked by default —
          they are given a cost penalty by cost_fn instead.
    """
    if start == goal:
        return [start]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score   = {start: 0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        if g > g_score.get(current, float('inf')):
            continue  # stale entry

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            neighbour = (nr, nc)
            if blocked_fn(nr, nc):
                continue
            tentative_g = g + cost_fn(nr, nc)
            if tentative_g < g_score.get(neighbour, float('inf')):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                f_new = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, neighbour))

    return []  # unreachable


# ---------------------------------------------------------------------------
# BFS helpers
# ---------------------------------------------------------------------------

def bfs_all_reachable(start: Tuple[int, int], rows: int, cols: int,
                      passable_fn: Callable) -> Set[Tuple[int, int]]:
    """
    BFS from start; returns the set of all reachable cells using 4-connectivity.

    Args:
        start       : (r, c) starting cell
        rows/cols   : grid dimensions
        passable_fn : (r, c) -> bool  True if cell can be entered

    Returns:
        Set of (r, c) tuples reachable from start.
    """
    visited = {start}
    queue   = collections.deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if (0 <= nr < rows and 0 <= nc < cols
                    and nb not in visited and passable_fn(nr, nc)):
                visited.add(nb)
                queue.append(nb)
    return visited


def is_fully_connected(passable_cells: Set[Tuple[int, int]],
                       rows: int, cols: int) -> bool:
    """
    Returns True if all passable cells form a single connected component.

    Used during world generation to validate wall placement.

    Args:
        passable_cells : set of (r, c) cells that are not walls
        rows/cols      : grid dimensions
    """
    if not passable_cells:
        return True
    start     = next(iter(passable_cells))
    reachable = bfs_all_reachable(
        start, rows, cols, lambda r, c: (r, c) in passable_cells
    )
    return len(reachable) == len(passable_cells)


# ---------------------------------------------------------------------------
# Shortest-path helpers
# ---------------------------------------------------------------------------

def shortest_path_length(start: Tuple[int, int], goal: Tuple[int, int],
                         rows: int, cols: int,
                         wall_set: Set[Tuple[int, int]]) -> Optional[int]:
    """
    [ORACLE] BFS shortest path length on the TRUE ground-truth grid.

    This function uses the actual wall set (ground truth).
    It must ONLY be called for post-hoc evaluation metrics, not for robot
    decision-making.  Callers must be labelled [ORACLE] in comments.

    Ignores mines (path length is measured as if mines were absent),
    giving the theoretical minimum steps for efficiency comparison.

    Args:
        start    : (r, c) start position
        goal     : (r, c) goal position
        rows/cols: grid dimensions
        wall_set : ground-truth set of wall cells (from GridWorld.walls)

    Returns:
        Integer path length in steps, or None if goal is unreachable.
    """
    if start == goal:
        return 0
    visited = {start}
    queue   = collections.deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if (0 <= nr < rows and 0 <= nc < cols
                    and nb not in visited and nb not in wall_set):
                if nb == goal:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
    return None


def shortest_path_length_knowledge(start: Tuple[int, int], goal: Tuple[int, int],
                                   rows: int, cols: int,
                                   known_blocked_fn: Callable) -> Optional[int]:
    """
    BFS shortest path using only the AGENT'S KNOWLEDGE of blocked cells.

    This version is safe to call from robot decision logic because it uses
    a known_blocked_fn derived from AgentKnowledge, not GridWorld ground truth.

    Unknown cells are treated as traversable (the agent hopes they are passable).

    Args:
        start            : (r, c) start position
        goal             : (r, c) goal position
        rows/cols        : grid dimensions
        known_blocked_fn : (r, c) -> bool derived from AgentKnowledge.is_known_blocked

    Returns:
        Integer path length in steps, or None if goal unreachable under current knowledge.
    """
    if start == goal:
        return 0
    visited = {start}
    queue   = collections.deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if (0 <= nr < rows and 0 <= nc < cols
                    and nb not in visited
                    and not known_blocked_fn(nr, nc)):
                if nb == goal:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
    return None
