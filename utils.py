"""
utils.py — Core algorithmic helpers:
  - A* pathfinding with pluggable cost/blocked functions
  - BFS connectivity checker for grid validation
"""
import heapq
import collections



def astar(start, goal, rows, cols, cost_fn, blocked_fn):
    """
    A* pathfinder on a grid.

    Args:
        start      : (r, c) tuple
        goal       : (r, c) tuple
        rows/cols  : grid dimensions
        cost_fn    : (r, c) -> numeric cost to ENTER cell (already confirmed passable)
        blocked_fn : (r, c) -> bool  (True = cannot enter)

    Returns:
        List of (r, c) tuples from start to goal (inclusive), or [] if unreachable.
    """
    if start == goal:
        return [start]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
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


def bfs_all_reachable(start, rows, cols, passable_fn):
    """
    BFS from start; returns set of all reachable cells using 4-connectivity.
    passable_fn(r, c) -> bool
    """
    visited = {start}
    queue = collections.deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nb not in visited and passable_fn(nr, nc):
                visited.add(nb)
                queue.append(nb)
    return visited


def is_fully_connected(passable_cells, rows, cols):
    """
    Returns True if all passable cells form a single connected component.
    passable_cells: set of (r, c)
    """
    if not passable_cells:
        return True
    start = next(iter(passable_cells))
    reachable = bfs_all_reachable(start, rows, cols, lambda r, c: (r, c) in passable_cells)
    return len(reachable) == len(passable_cells)


def shortest_path_length(start, goal, rows, cols, wall_set):
    """
    BFS shortest path length on true grid (ignoring mines — used for efficiency stats).
    Returns int steps, or None if unreachable.
    """
    if start == goal:
        return 0
    visited = {start}
    queue = collections.deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nb not in visited and nb not in wall_set:
                if nb == goal:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
    return None
