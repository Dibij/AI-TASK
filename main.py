"""
main.py — Pygame renderer and game loop for the Multi-Agent Pathfinding Simulation.

Layout: 1100 × 700 px
  Left  680 px  : Grid panel (15×15 grid, 40px cells)
  Right 420 px  : Stats / log sidebar

Controls:
  SPACE      : toggle play/pause
  R          : full reset (new world, new robots)
  LEFT/RIGHT : step through history (while paused)
  +  / =     : speed up simulation
  -          : slow down simulation
  ESC        : quit

Architecture notes:
  - The renderer reads from game.world.known_cells (AgentKnowledge string view)
    which contains only what the robots have discovered.
  - Hidden mines are rendered as faint circles for debugging purposes;
    this visual aid is clearly marked as a debug overlay.
  - Trajectory prediction circles show SHORT-HORIZON DETERMINISTIC predictions,
    not probabilistic collision probabilities.
  - Sidebar stats correctly separate mine deaths from collision deaths.
"""

import sys
import io
# Force UTF-8 stdout on Windows so emoji in log strings don't crash the terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import math
import pygame

from game import Game, MAX_ROUNDS, NUM_ROBOTS
from grid import CELL_WALL, CELL_MINE, CELL_EMPTY, CELL_UNKNOWN


# ── Window & Layout ─────────────────────────────────────────────────────────
WIN_W, WIN_H    = 1100, 700
GRID_PANEL_W    = 680
SIDEBAR_W       = WIN_W - GRID_PANEL_W

GRID_ROWS, GRID_COLS = 15, 15
CELL_PX         = 40   # pixels per grid cell
GRID_MARGIN_X   = (GRID_PANEL_W - GRID_COLS * CELL_PX) // 2
GRID_MARGIN_Y   = (WIN_H        - GRID_ROWS * CELL_PX) // 2

# ── Colour Palette ───────────────────────────────────────────────────────────
BG              = (10,  15,  28)
SIDEBAR_BG      = (18,  26,  46)
DIVIDER         = (40,  55,  80)
PANEL_BG        = (12,  20,  38)

C_UNKNOWN       = (42,  56,  78)    # unexplored cell
C_EMPTY         = (235, 240, 248)   # discovered safe
C_WALL          = (22,  30,  50)    # confirmed wall (near-black blue)
C_MINE          = (146, 64,  14)    # amber-brown — revealed mine
C_GRID_LINE     = (30,  42,  62)

TEXT_LIGHT      = (230, 238, 255)
TEXT_MUTED      = (110, 130, 165)
TEXT_CONSOLE    = (52,  211, 153)   # emerald green terminal
TEXT_WARNING    = (251, 191, 36)    # amber
TEXT_DANGER     = (239, 68,  68)    # red
TEXT_SUCCESS    = (34,  197, 94)    # green

# Minesweeper clue colours
CLUE_COLORS     = {
    1: (37,  99,  235),
    2: (22,  163, 74),
    3: (220, 38,  38),
    4: (147, 51,  234),
    5: (180, 83,  9),
    6: (13,  148, 136),
    7: (15,  23,  42),
    8: (100, 116, 139),
}

# Default ticks per second
DEFAULT_TPS     = 4
MIN_TPS         = 1
MAX_TPS         = 20


# ── Helper ───────────────────────────────────────────────────────────────────

def cell_rect(r, c):
    """Return pygame.Rect for grid cell (r, c)."""
    x = GRID_MARGIN_X + c * CELL_PX
    y = GRID_MARGIN_Y + r * CELL_PX
    return pygame.Rect(x, y, CELL_PX, CELL_PX)


def cell_center(r, c):
    rect = cell_rect(r, c)
    return rect.centerx, rect.centery


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_rounded_rect(surf, color, rect, radius=6, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


# ── Main renderer class ───────────────────────────────────────────────────────

class Renderer:
    def __init__(self, screen):
        self.screen     = screen
        pygame.font.init()
        self.f_title    = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.f_sub      = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.f_ui       = pygame.font.SysFont("Segoe UI", 13)
        self.f_cell     = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.f_clue     = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.f_console  = pygame.font.SysFont("Consolas", 11)
        self.f_big      = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.log_scroll = 0

        # Load robot sprites
        self.sprites = {}
        colors = ["blue", "red", "green", "yellow"]
        radius = CELL_PX // 2 - 5
        import os
        for i, c in enumerate(colors):
            path = os.path.join("assets", f"{c}.png")
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    img_surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                    img_surface.fill((255, 255, 255))
                    img.set_colorkey((255, 255, 255))
                    img = pygame.transform.smoothscale(img, (radius * 2, radius * 2))
                    self.sprites[i] = img
                except Exception as e:
                    print(f"Error loading {path}: {e}")

    # ── Grid Panel ─────────────────────────────────────────────────────

    def draw_grid(self, game):
        world  = game.world
        robots = game.robots

        # known_cells is the backward-compatible string-view of AgentKnowledge
        kc = world.known_cells

        for row in range(world.rows):
            for col in range(world.cols):
                cell   = (row, col)
                status = kc.get(cell, CELL_UNKNOWN)
                rect   = cell_rect(row, col)

                # ── Cell background ──
                if status == CELL_WALL:
                    bg = C_WALL
                elif status == CELL_MINE:
                    bg = C_MINE
                elif status == CELL_EMPTY:
                    bg = C_EMPTY
                else:
                    bg = C_UNKNOWN

                pygame.draw.rect(self.screen, bg, rect)

                # ── Visited Path Rendering ──
                if status == CELL_EMPTY and hasattr(world, 'visited') and cell in world.visited:
                    visitors = list(world.visited[cell])
                    n = len(visitors)
                    if n > 0:
                        w = CELL_PX / n
                        for i, color in enumerate(visitors):
                            sub_rect = pygame.Rect(
                                rect.left + int(i * w), rect.top, int(w), CELL_PX
                            )
                            s = pygame.Surface((sub_rect.width, sub_rect.height), pygame.SRCALPHA)
                            s.fill((color[0], color[1], color[2], 100))
                            self.screen.blit(s, sub_rect.topleft)

                pygame.draw.rect(self.screen, C_GRID_LINE, rect, 1)

                # ── Coordinate Label ──
                coord_surf = self.f_console.render(f"{row},{col}", True, (80, 90, 110))
                self.screen.blit(
                    coord_surf,
                    (rect.right - coord_surf.get_width() - 2,
                     rect.bottom - coord_surf.get_height() - 2)
                )

                # ── Mine probability heat on unknown cells ──
                if status == CELL_UNKNOWN:
                    p = world.get_mine_probability(row, col)
                    if p > 0.05:
                        heat  = lerp_color((42, 56, 78), (220, 38, 38), min(1, p * 2))
                        dot_r = max(3, int(CELL_PX * 0.18 * p))
                        pygame.draw.circle(self.screen, heat, rect.center, dot_r)

                # ── Minesweeper clue number ──
                clue = world.known_clues.get(cell)
                if clue and status == CELL_EMPTY:
                    clue_surf = self.f_clue.render(
                        str(clue), True, CLUE_COLORS.get(clue, TEXT_MUTED)
                    )
                    clue_rect = clue_surf.get_rect(center=rect.center)
                    self.screen.blit(clue_surf, clue_rect)

                # ── Mine icon for revealed mines ──
                if status == CELL_MINE:
                    cx, cy = rect.center
                    r2 = CELL_PX // 5
                    pygame.draw.circle(self.screen, (30, 15, 5), (cx, cy), r2)
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                        pygame.draw.line(self.screen, (30, 15, 5),
                                         (cx, cy), (cx + dx*r2, cy + dy*r2), 2)

        # ── Debug: Draw Hidden Mines (Dimly) ──
        # This is a VISUAL DEBUG OVERLAY ONLY.  The robots cannot see these.
        # They are drawn faintly so the player can evaluate robot decisions.
        for mr, mc in world.mines:
            status = kc.get((mr, mc))
            if status != CELL_MINE:
                m_rect = cell_rect(mr, mc)
                s = pygame.Surface((CELL_PX, CELL_PX), pygame.SRCALPHA)
                pygame.draw.circle(
                    s, (200, 30, 30, 40),
                    (CELL_PX // 2, CELL_PX // 2), CELL_PX // 4
                )
                self.screen.blit(s, m_rect.topleft)

        # ── Target markers ──
        for robot in robots:
            if robot.target:
                tr, tc  = robot.target
                t_rect  = cell_rect(tr, tc)
                pulse   = abs(math.sin(time.time() * 3)) * 3
                ring_r  = int(CELL_PX // 2 - 4 + pulse)
                pygame.draw.circle(self.screen, robot.color, t_rect.center, ring_r, 2)
                cx, cy = t_rect.center
                d = 5
                pygame.draw.polygon(self.screen, robot.color, [
                    (cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)
                ])

        # ── Planned paths (dotted lines) ──
        for robot in robots:
            if not robot.alive:
                continue
            path = [robot.pos] + robot.planned_path
            if len(path) > 1:
                for i in range(len(path) - 1):
                    ax, ay = cell_center(*path[i])
                    bx, by = cell_center(*path[i + 1])
                    dx, dy = bx - ax, by - ay
                    dist   = max(1, math.hypot(dx, dy))
                    steps  = max(1, int(dist / 6))
                    for j in range(steps):
                        if j % 2 == 0:
                            px = int(ax + dx * j / steps)
                            py = int(ay + dy * j / steps)
                            alpha_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                            pygame.draw.circle(alpha_surf, (*robot.color, 120), (2, 2), 2)
                            self.screen.blit(alpha_surf, (px - 2, py - 2))

        # ── Predicted positions (faded circles, 3-step horizon) ──
        # NOTE: These are SHORT-HORIZON DETERMINISTIC TRAJECTORY PREDICTIONS,
        # not probabilistic collision probabilities.
        for robot in robots:
            if not robot.alive:
                continue
            preds = robot.predict_others(robots, horizon=3)
            for step, cells in preds.items():
                alpha = max(20, 80 - step * 20)
                for (pr, pc) in cells:
                    cx2, cy2 = cell_center(pr, pc)
                    s = pygame.Surface((CELL_PX, CELL_PX), pygame.SRCALPHA)
                    pygame.draw.circle(
                        s, (*robot.color, alpha),
                        (CELL_PX // 2, CELL_PX // 2), CELL_PX // 4
                    )
                    self.screen.blit(s, (cx2 - CELL_PX // 2, cy2 - CELL_PX // 2))

        # ── Robots ──
        for robot in robots:
            if robot.pos is None:
                continue
            cr, cc = robot.pos
            cx, cy = cell_center(cr, cc)
            radius = CELL_PX // 2 - 5

            if not robot.alive:
                pygame.draw.line(self.screen, TEXT_DANGER,
                                 (cx - radius, cy - radius), (cx + radius, cy + radius), 3)
                pygame.draw.line(self.screen, TEXT_DANGER,
                                 (cx + radius, cy - radius), (cx - radius, cy + radius), 3)
                continue

            # Shadow
            shadow_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 60), (radius + 2, radius + 3), radius)
            self.screen.blit(shadow_surf, (cx - radius - 2, cy - radius - 2))

            # Body
            sprite = getattr(self, "sprites", {}).get(robot.id)
            if sprite:
                self.screen.blit(sprite, (cx - radius, cy - radius))
            else:
                pygame.draw.circle(self.screen, robot.color, (cx, cy), radius)
                pygame.draw.circle(self.screen, TEXT_LIGHT, (cx, cy), radius, 2)

            # ID label
            id_surf = self.f_cell.render(robot.name[0], True, TEXT_LIGHT)
            id_rect = id_surf.get_rect(center=(cx, cy))
            self.screen.blit(id_surf, id_rect)

            # Waiting indicator
            if robot.wait_counter > 0:
                wait_surf = self.f_sub.render("...", True, TEXT_WARNING)
                self.screen.blit(wait_surf, (cx - 8, cy - radius - 18))

            # Reached indicator
            if robot.reached:
                check_surf = self.f_sub.render("✓", True, TEXT_SUCCESS)
                self.screen.blit(check_surf, (cx - 5, cy - radius - 18))

    # ── Sidebar ─────────────────────────────────────────────────────────

    def draw_sidebar(self, game, tps, paused=False):
        sb_x    = GRID_PANEL_W
        sb_rect = pygame.Rect(sb_x, 0, SIDEBAR_W, WIN_H)
        pygame.draw.rect(self.screen, SIDEBAR_BG, sb_rect)
        pygame.draw.line(self.screen, DIVIDER, (sb_x, 0), (sb_x, WIN_H), 2)

        x = sb_x + 16
        y = 14

        # ── Title ──
        t = self.f_title.render("Multi-Agent Pathfinding", True, TEXT_LIGHT)
        self.screen.blit(t, (x, y))
        y += 26
        sub = self.f_sub.render("Risk-Aware · Partial Obs. · A* Planner", True, TEXT_MUTED)
        self.screen.blit(sub, (x, y))
        y += 22

        pygame.draw.line(self.screen, DIVIDER, (sb_x + 10, y), (WIN_W - 10, y), 1)
        y += 10

        # ── Round info ──
        round_str = f"Round {game.round_number} / {game.max_rounds}"
        if game.tournament_over:
            round_str = "Tournament Complete!"
        r_surf = self.f_sub.render(round_str, True, TEXT_WARNING)
        self.screen.blit(r_surf, (x, y))

        bar_w    = SIDEBAR_W - 32
        bar_rect = pygame.Rect(x, y + 18, bar_w, 8)
        pygame.draw.rect(self.screen, PANEL_BG, bar_rect, border_radius=4)
        prog = min(1.0, game.round_number / game.max_rounds)
        if prog > 0:
            fill = pygame.Rect(x, y + 18, int(bar_w * prog), 8)
            pygame.draw.rect(self.screen, (59, 130, 246), fill, border_radius=4)
        y += 36

        status_str = f"Tick {game.tick}"
        if game.round_over and not game.tournament_over:
            status_str += "  — SPACE for next round"
        elif game.tournament_over:
            status_str += "  — R to reset"
        tick_surf = self.f_ui.render(status_str, True, TEXT_MUTED)
        self.screen.blit(tick_surf, (x, y))
        spd_surf = self.f_ui.render(f"Speed: {tps} tps (+/-)", True, TEXT_MUTED)
        self.screen.blit(spd_surf, (x + bar_w - spd_surf.get_width(), y))
        y += 22

        # Play/Pause button
        self.play_pause_rect = pygame.Rect(x, y, 100, 24)
        btn_color = TEXT_SUCCESS if paused else TEXT_WARNING
        btn_text  = "▶ PLAY" if paused else "⏸ PAUSE"
        draw_rounded_rect(self.screen, btn_color, self.play_pause_rect, radius=4)
        t_surf = self.f_sub.render(btn_text, True, BG)
        t_rect = t_surf.get_rect(center=self.play_pause_rect.center)
        self.screen.blit(t_surf, t_rect)
        y += 34

        pygame.draw.line(self.screen, DIVIDER, (sb_x + 10, y), (WIN_W - 10, y), 1)
        y += 10

        # ── Leaderboard ──
        lb_surf = self.f_sub.render("LEADERBOARD", True, TEXT_MUTED)
        self.screen.blit(lb_surf, (x, y))
        y += 18

        for robot in game.leaderboard():
            eff  = game.avg_efficiency(robot)
            surv = game.survival_rate(robot)

            row_rect = pygame.Rect(x - 4, y - 2, SIDEBAR_W - 28, 52)
            draw_rounded_rect(self.screen, PANEL_BG, row_rect, radius=5,
                              border=1, border_color=DIVIDER)

            pygame.draw.circle(self.screen, robot.color, (x + 12, y + 12), 10)
            id_surf = self.f_sub.render(robot.name[0], True, TEXT_LIGHT)
            self.screen.blit(id_surf, id_surf.get_rect(center=(x + 12, y + 12)))

            name_surf = self.f_sub.render(
                f"{robot.name}  "
                f"spd={'fast' if robot.speed == 2 else 'slow'}  "
                f"risk={robot.risk_tolerance:.0%}",
                True, TEXT_LIGHT
            )
            self.screen.blit(name_surf, (x + 28, y + 2))

            eff_bar_w = SIDEBAR_W - 80
            eff_bg    = pygame.Rect(x + 28, y + 20, eff_bar_w, 7)
            pygame.draw.rect(self.screen, DIVIDER, eff_bg, border_radius=3)
            eff_fill  = pygame.Rect(x + 28, y + 20, int(eff_bar_w * eff), 7)
            eff_color = TEXT_SUCCESS if eff > 0.6 else TEXT_WARNING if eff > 0.3 else TEXT_DANGER
            if eff > 0:
                pygame.draw.rect(self.screen, eff_color, eff_fill, border_radius=3)

            # Show separate death types: mine vs collision
            mine_d  = robot.total_mine_deaths
            coll_d  = robot.total_collision_deaths
            stats_str = (
                f"eff={eff:.2f}  surv={surv:.0%}  "
                f"wins={robot.total_wins}  "
                f"mine✗={mine_d}  coll✗={coll_d}"
            )
            st_surf = self.f_console.render(stats_str, True, TEXT_MUTED)
            self.screen.blit(st_surf, (x + 28, y + 32))

            y += 58

        pygame.draw.line(self.screen, DIVIDER, (sb_x + 10, y), (WIN_W - 10, y), 1)
        y += 10

        # ── Robot trait display ──
        trait_surf = self.f_sub.render("ROBOT TRAITS", True, TEXT_MUTED)
        self.screen.blit(trait_surf, (x, y))
        y += 16
        for robot in game.robots:
            tr_str = (
                f"{robot.name}: spd={robot.speed} "
                f"risk={robot.risk_tolerance:.0%} "
                f"pat={robot.patience:.0%}"
            )
            tr_surf = self.f_console.render(tr_str, True, robot.color)
            self.screen.blit(tr_surf, (x, y))
            y += 14
        y += 4
        pygame.draw.line(self.screen, DIVIDER, (sb_x + 10, y), (WIN_W - 10, y), 1)
        y += 8

        # ── Action Log ──
        log_label = self.f_sub.render("SOLVER DECISION LOG", True, TEXT_MUTED)
        self.screen.blit(log_label, (x, y))
        y += 18

        log_area_h = WIN_H - y - 36
        log_rect   = pygame.Rect(x - 4, y, SIDEBAR_W - 28, log_area_h)
        draw_rounded_rect(self.screen, PANEL_BG, log_rect, radius=5,
                          border=1, border_color=DIVIDER)

        self.screen.set_clip(log_rect)
        line_h    = 15
        max_lines = log_area_h // line_h
        log       = game.log

        max_scroll = max(0, len(log) - max_lines)
        self.log_scroll = min(self.log_scroll, max_scroll)
        start_i   = max(0, len(log) - max_lines - self.log_scroll)
        end_i     = max(0, len(log) - self.log_scroll)
        visible   = log[start_i:end_i]

        for i, msg in enumerate(visible):
            if "MINE" in msg or "💥" in msg:
                color = TEXT_DANGER
            elif "✅" in msg or "reached" in msg:
                color = TEXT_SUCCESS
            elif "wait" in msg.lower():
                color = TEXT_WARNING
            else:
                color = TEXT_CONSOLE
            l_surf = self.f_console.render(msg[:62], True, color)
            self.screen.blit(l_surf, (x, y + i * line_h + 4))

        self.screen.set_clip(None)

        if len(log) > max_lines:
            hint = self.f_console.render(
                f"{start_i+1}–{end_i}/{len(log)} (scroll wheel)", True, TEXT_MUTED
            )
            self.screen.blit(hint, (x, WIN_H - 30))

        ctrl = self.f_console.render(
            "SPACE=play/pause  R=reset  +/-=speed  ESC=quit", True, TEXT_MUTED
        )
        self.screen.blit(ctrl, (x, WIN_H - 16))

    def handle_scroll(self, direction, log_len):
        if direction > 0:
            self.log_scroll = min(max(0, log_len - 1), self.log_scroll + 2)
        else:
            self.log_scroll = max(0, self.log_scroll - 2)

    def draw_tooltip(self, robot, mx, my, all_robots):
        oracle = getattr(robot, '_shortest_oracle', '?')
        lines = [
            f"{robot.name}  [{robot._planner_name}]",
            f"Speed: {robot.speed}   Risk tol: {robot.risk_tolerance:.0%}",
            f"Steps: {robot.steps_taken}  /  Oracle: {oracle}",
            f"Waits: {robot.waits_taken}  Replans: {robot.replans_taken}",
            f"Unknown cells entered: {robot.unknown_cells_entered}",
            f"Pos: {robot.pos} → Target: {robot.target}",
            f"State: {'Alive' if robot.alive else 'Dead (mine)' if robot.hit_mine else 'Dead (collision)' if robot.hit_robot else 'Dead'} "
            f"{'(Reached)' if robot.reached else ''}"
        ]

        # Short-horizon trajectory predictions (not probabilistic)
        preds = robot.predict_others(all_robots, horizon=3)
        if any(preds.values()):
            lines.append("Trajectory predictions (horizon 3):")
            for step, cells in preds.items():
                if cells:
                    lines.append(f"  T+{step}: {list(cells)}")

        surfs = [self.f_ui.render(line, True, TEXT_LIGHT) for line in lines]
        w = max(s.get_width() for s in surfs) + 16
        h = sum(s.get_height() for s in surfs) + 16

        rect = pygame.Rect(mx + 12, my + 12, w, h)
        if rect.right > WIN_W:
            rect.right = mx - w - 12
        if rect.bottom > WIN_H:
            rect.bottom = my - h - 12

        bg_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(bg_surf, (20, 25, 40, 240), bg_surf.get_rect(), border_radius=6)
        pygame.draw.rect(bg_surf, DIVIDER, bg_surf.get_rect(), 1, border_radius=6)
        self.screen.blit(bg_surf, rect)

        cy = rect.y + 8
        for s in surfs:
            self.screen.blit(s, (rect.x + 8, cy))
            cy += s.get_height()

    # ── Tournament over overlay ──────────────────────────────────────────

    def draw_tournament_overlay(self, game):
        overlay = pygame.Surface((GRID_PANEL_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((10, 15, 28, 200))
        self.screen.blit(overlay, (0, 0))

        winner    = game.leaderboard()[0]
        title_surf = self.f_big.render("🏆 Tournament Complete!", True, TEXT_WARNING)
        self.screen.blit(title_surf,
                         title_surf.get_rect(center=(GRID_PANEL_W // 2, WIN_H // 2 - 60)))

        win_surf = self.f_title.render(
            f"Winner: {winner.name}  "
            f"(eff={game.avg_efficiency(winner):.2f})",
            True, winner.color
        )
        self.screen.blit(win_surf,
                         win_surf.get_rect(center=(GRID_PANEL_W // 2, WIN_H // 2)))

        r_surf = self.f_sub.render("Press R to reset", True, TEXT_MUTED)
        self.screen.blit(r_surf,
                         r_surf.get_rect(center=(GRID_PANEL_W // 2, WIN_H // 2 + 50)))


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.display.set_caption(
        "Multi-Agent Pathfinding Simulation — Risk-Aware A* with Partial Observability"
    )
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    game     = Game()  # default: risk_aware planner, random seed
    renderer = Renderer(screen)
    tps      = DEFAULT_TPS

    paused         = True
    last_tick_time = time.time()
    round_end_time = None

    import copy
    history       = [copy.deepcopy(game)]
    history_index = 0

    print("[MAPF] Simulation started.")
    print(f"[MAPF] World: {game.world.rows}×{game.world.cols}  "
          f"Walls: {game.world.count_walls()}  "
          f"Mines: {game.world.count_mines()}")
    for r in game.robots:
        print(f"[MAPF] {r.name}: speed={r.speed}  "
              f"risk={r.risk_tolerance:.2f}  patience={r.patience:.2f}  "
              f"planner={r._planner_name}")

    while True:
        # ── Events ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_r:
                    game          = Game()
                    history       = [copy.deepcopy(game)]
                    history_index = 0
                    renderer.log_scroll = 0
                    last_tick_time = time.time()
                    round_end_time = None
                    print("[MAPF] Full reset.")

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_LEFT:
                    paused        = True
                    history_index = max(0, history_index - 1)
                    game          = history[history_index]

                elif event.key == pygame.K_RIGHT:
                    paused        = True
                    history_index = min(len(history) - 1, history_index + 1)
                    game          = history[history_index]

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    tps = min(MAX_TPS, tps + 1)

                elif event.key == pygame.K_MINUS:
                    tps = max(MIN_TPS, tps - 1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if (hasattr(renderer, "play_pause_rect")
                            and renderer.play_pause_rect.collidepoint(event.pos)):
                        paused = not paused
                elif event.button == 4:
                    renderer.handle_scroll(1, len(game.log))
                elif event.button == 5:
                    renderer.handle_scroll(-1, len(game.log))

        # ── Simulation Tick ───────────────────────────────────────────
        now = time.time()
        if game.running and not paused and (now - last_tick_time) >= 1.0 / tps:
            if history_index < len(history) - 1:
                history = history[:history_index + 1]

            game.tick_simulation()
            history.append(copy.deepcopy(game))
            history_index = len(history) - 1
            last_tick_time = now

            # Print notable events to terminal
            if hasattr(game, 'tick_logs') and game.tick_logs:
                for latest in game.tick_logs:
                    if any(k in latest for k in [
                        "MINE", "reached", "Round", "waiting", "stuck",
                        "moved", "avoids", "impatient", "COLLISION", "SWAP"
                    ]):
                        safe = latest.encode('ascii', errors='replace').decode('ascii')
                        color_code = "\033[0m"
                        if "Blue" in safe:
                            color_code = "\033[94m"
                        elif "Red" in safe:
                            color_code = "\033[91m"
                        elif "Green" in safe:
                            color_code = "\033[92m"
                        elif "Yellow" in safe:
                            color_code = "\033[93m"
                        if "COLLISION" in safe or "MINE" in safe:
                            color_code = "\033[1;91m"
                        print(f"{color_code}[MAPF] {safe}\033[0m")

        # Auto-advance after round ends
        if game.round_over and not game.tournament_over:
            if round_end_time is None:
                round_end_time = time.time()
            elif time.time() - round_end_time > 2.5:
                game.next_round()
                renderer.log_scroll = 0
                round_end_time      = None

        # ── Render ───────────────────────────────────────────────────
        screen.fill(BG)
        renderer.draw_grid(game)
        renderer.draw_sidebar(game, tps, paused=paused)
        if game.tournament_over:
            renderer.draw_tournament_overlay(game)

        # Hover tooltips
        mx, my = pygame.mouse.get_pos()
        for robot in game.robots:
            if robot.pos:
                cx, cy = cell_center(*robot.pos)
                r = CELL_PX // 2
                if math.hypot(mx - cx, my - cy) <= r:
                    renderer.draw_tooltip(robot, mx, my, game.robots)
                    break

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
