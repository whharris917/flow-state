"""
Icons - Procedural Icon Drawing for UI Buttons

Each icon function takes (surface, rect, color) and draws the icon
centered within the given rect. Icons are drawn using pygame primitives
for crisp, scalable rendering that matches the app's visual style.

Usage:
    button = Button(x, y, w, h, icon=icons.ICON_LINE)
"""

import pygame
import math


def draw_icon_brush(surface, rect, color):
    """Brush tool - circle with spray dots."""
    cx, cy = rect.center
    r = min(rect.width, rect.height) // 2 - 4
    
    # Outer circle (brush area)
    pygame.draw.circle(surface, color, (cx, cy), r, 1)
    
    # Spray dots inside
    dots = [(0, 0), (-3, -4), (4, -2), (-2, 4), (3, 3), (-4, 1)]
    for dx, dy in dots:
        pygame.draw.circle(surface, color, (cx + dx, cy + dy), 2)


def draw_icon_select(surface, rect, color):
    """Select tool - arrow cursor."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Arrow cursor pointing up-left
    points = [
        (cx - size//2, cy - size//2),      # Tip
        (cx - size//2, cy + size//2),      # Bottom left
        (cx - size//6, cy + size//6),      # Inner corner
        (cx + size//3, cy + size//2 + 2),  # Arrow tail end
        (cx + size//6, cy + size//6),      # Inner corner 2
        (cx + size//2, cy - size//2),      # Right edge
    ]
    pygame.draw.polygon(surface, color, points, 0)
    pygame.draw.polygon(surface, (255, 255, 255), points, 1)


def draw_icon_line(surface, rect, color):
    """Line tool - diagonal line with endpoints."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    p1 = (cx - size, cy + size)
    p2 = (cx + size, cy - size)
    
    pygame.draw.line(surface, color, p1, p2, 2)
    pygame.draw.circle(surface, color, p1, 3)
    pygame.draw.circle(surface, color, p2, 3)


def draw_icon_rect(surface, rect, color):
    """Rectangle tool - rectangle outline."""
    cx, cy = rect.center
    w = min(rect.width, rect.height) - 10
    h = w * 0.7
    
    r = pygame.Rect(cx - w//2, cy - h//2, w, int(h))
    pygame.draw.rect(surface, color, r, 2)


def draw_icon_circle(surface, rect, color):
    """Circle tool - circle outline."""
    cx, cy = rect.center
    r = min(rect.width, rect.height) // 2 - 5
    
    pygame.draw.circle(surface, color, (cx, cy), r, 2)


def draw_icon_point(surface, rect, color):
    """Point tool - dot with crosshair."""
    cx, cy = rect.center
    
    # Crosshair
    size = 6
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 1)
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 1)
    
    # Center dot
    pygame.draw.circle(surface, color, (cx, cy), 3)


def draw_icon_ref_line(surface, rect, color):
    """Reference line tool - dashed diagonal line."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    # Dashed line
    p1 = (cx - size, cy + size)
    p2 = (cx + size, cy - size)
    
    # Draw dashes
    num_dashes = 4
    for i in range(num_dashes):
        t1 = i / num_dashes
        t2 = (i + 0.5) / num_dashes
        x1 = p1[0] + (p2[0] - p1[0]) * t1
        y1 = p1[1] + (p2[1] - p1[1]) * t1
        x2 = p1[0] + (p2[0] - p1[0]) * t2
        y2 = p1[1] + (p2[1] - p1[1]) * t2
        pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), 2)


# =============================================================================
# Constraint Icons
# =============================================================================

def draw_icon_coincident(surface, rect, color):
    """Coincident constraint - two arrows pointing at center dot."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Center dot
    pygame.draw.circle(surface, color, (cx, cy), 3)
    
    # Left arrow pointing right
    arrow_len = size - 2
    pygame.draw.line(surface, color, (cx - arrow_len, cy), (cx - 5, cy), 2)
    pygame.draw.line(surface, color, (cx - 8, cy - 3), (cx - 5, cy), 2)
    pygame.draw.line(surface, color, (cx - 8, cy + 3), (cx - 5, cy), 2)
    
    # Right arrow pointing left
    pygame.draw.line(surface, color, (cx + arrow_len, cy), (cx + 5, cy), 2)
    pygame.draw.line(surface, color, (cx + 8, cy - 3), (cx + 5, cy), 2)
    pygame.draw.line(surface, color, (cx + 8, cy + 3), (cx + 5, cy), 2)


def draw_icon_collinear(surface, rect, color):
    """Collinear constraint - three dots on a diagonal line."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    # Diagonal line
    p1 = (cx - size, cy + size//2)
    p2 = (cx + size, cy - size//2)
    pygame.draw.line(surface, color, p1, p2, 1)
    
    # Three dots along the line
    for t in [0.15, 0.5, 0.85]:
        px = int(p1[0] + (p2[0] - p1[0]) * t)
        py = int(p1[1] + (p2[1] - p1[1]) * t)
        pygame.draw.circle(surface, color, (px, py), 3)


def draw_icon_midpoint(surface, rect, color):
    """Midpoint constraint - line with dot at center."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Horizontal line
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 2)
    
    # Endpoints
    pygame.draw.circle(surface, color, (cx - size, cy), 2)
    pygame.draw.circle(surface, color, (cx + size, cy), 2)
    
    # Center dot (larger, highlighted)
    pygame.draw.circle(surface, (255, 200, 100), (cx, cy), 4)
    pygame.draw.circle(surface, color, (cx, cy), 4, 1)


def draw_icon_length(surface, rect, color):
    """Length constraint - line with dimension arrows."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Main line
    y_line = cy + 2
    pygame.draw.line(surface, color, (cx - size, y_line), (cx + size, y_line), 2)
    
    # Dimension arrows (left)
    pygame.draw.line(surface, color, (cx - size, y_line - 4), (cx - size, y_line + 4), 1)
    pygame.draw.line(surface, color, (cx - size, y_line), (cx - size + 4, y_line - 3), 1)
    pygame.draw.line(surface, color, (cx - size, y_line), (cx - size + 4, y_line + 3), 1)
    
    # Dimension arrows (right)
    pygame.draw.line(surface, color, (cx + size, y_line - 4), (cx + size, y_line + 4), 1)
    pygame.draw.line(surface, color, (cx + size, y_line), (cx + size - 4, y_line - 3), 1)
    pygame.draw.line(surface, color, (cx + size, y_line), (cx + size - 4, y_line + 3), 1)
    
    # "L" label
    pygame.draw.line(surface, color, (cx - 3, cy - 6), (cx - 3, cy - 1), 2)
    pygame.draw.line(surface, color, (cx - 3, cy - 1), (cx + 3, cy - 1), 2)


def draw_icon_equal(surface, rect, color):
    """Equal length constraint - equals sign."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    # Two horizontal lines
    pygame.draw.line(surface, color, (cx - size, cy - 4), (cx + size, cy - 4), 3)
    pygame.draw.line(surface, color, (cx - size, cy + 4), (cx + size, cy + 4), 3)


def draw_icon_parallel(surface, rect, color):
    """Parallel constraint - two parallel diagonal lines."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    offset = 4
    
    # Two parallel lines
    pygame.draw.line(surface, color, 
                     (cx - size + offset, cy + size), 
                     (cx + size + offset, cy - size), 2)
    pygame.draw.line(surface, color, 
                     (cx - size - offset, cy + size), 
                     (cx + size - offset, cy - size), 2)


def draw_icon_perpendicular(surface, rect, color):
    """Perpendicular constraint - T or ⊥ symbol."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Vertical line
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 2)
    
    # Horizontal line at bottom
    pygame.draw.line(surface, color, (cx - size, cy + size), (cx + size, cy + size), 2)
    
    # Small square to indicate right angle
    sq_size = 5
    pygame.draw.line(surface, color, (cx, cy + size - sq_size), (cx + sq_size, cy + size - sq_size), 1)
    pygame.draw.line(surface, color, (cx + sq_size, cy + size - sq_size), (cx + sq_size, cy + size), 1)


def draw_icon_horizontal(surface, rect, color):
    """Horizontal constraint - horizontal line with arrows."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Horizontal line
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 2)
    
    # Left arrow
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size + 5, cy - 4), 2)
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size + 5, cy + 4), 2)
    
    # Right arrow
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy - 4), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy + 4), 2)


def draw_icon_vertical(surface, rect, color):
    """Vertical constraint - vertical line with arrows."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Vertical line
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 2)
    
    # Top arrow
    pygame.draw.line(surface, color, (cx, cy - size), (cx - 4, cy - size + 5), 2)
    pygame.draw.line(surface, color, (cx, cy - size), (cx + 4, cy - size + 5), 2)
    
    # Bottom arrow
    pygame.draw.line(surface, color, (cx, cy + size), (cx - 4, cy + size - 5), 2)
    pygame.draw.line(surface, color, (cx, cy + size), (cx + 4, cy + size - 5), 2)


def draw_icon_angle(surface, rect, color):
    """Angle constraint - angle symbol with arc."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Two lines forming an angle
    origin = (cx - size//2, cy + size//2)
    p1 = (cx + size, cy + size//2)  # Horizontal
    p2 = (cx + size//3, cy - size)   # Angled up
    
    pygame.draw.line(surface, color, origin, p1, 2)
    pygame.draw.line(surface, color, origin, p2, 2)
    
    # Arc to show angle
    arc_rect = pygame.Rect(origin[0] - 8, origin[1] - 8, 16, 16)
    pygame.draw.arc(surface, color, arc_rect, 0, math.pi/3 + 0.2, 2)


def draw_icon_extend(surface, rect, color):
    """Extend/Infinite line - line with arrows extending outward."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Center line segment
    pygame.draw.line(surface, color, (cx - size//2, cy), (cx + size//2, cy), 2)
    
    # Dashed extensions
    for sign in [-1, 1]:
        start = cx + sign * size//2
        for i in range(2):
            x1 = start + sign * (4 + i * 6)
            x2 = x1 + sign * 3
            pygame.draw.line(surface, color, (x1, cy), (x2, cy), 1)
    
    # Arrow heads
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size + 4, cy - 3), 1)
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size + 4, cy + 3), 1)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 4, cy - 3), 1)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 4, cy + 3), 1)


# =============================================================================
# Action Icons
# =============================================================================

def draw_icon_play(surface, rect, color):
    """Play button - triangle."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    points = [
        (cx - size//2, cy - size),
        (cx - size//2, cy + size),
        (cx + size, cy)
    ]
    pygame.draw.polygon(surface, color, points)


def draw_icon_pause(surface, rect, color):
    """Pause button - two vertical bars."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    bar_w = size // 2
    
    pygame.draw.rect(surface, color, (cx - size//2 - bar_w//2, cy - size, bar_w, size * 2))
    pygame.draw.rect(surface, color, (cx + size//2 - bar_w//2, cy - size, bar_w, size * 2))


def draw_icon_undo(surface, rect, color):
    """Undo - curved arrow pointing left."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Curved arrow (approximated with arc and lines)
    arc_rect = pygame.Rect(cx - size, cy - size//2, size * 2, size)
    pygame.draw.arc(surface, color, arc_rect, 0.5, math.pi, 2)
    
    # Arrow head
    ax, ay = cx - size, cy
    pygame.draw.line(surface, color, (ax, ay), (ax + 5, ay - 5), 2)
    pygame.draw.line(surface, color, (ax, ay), (ax + 5, ay + 5), 2)


def draw_icon_redo(surface, rect, color):
    """Redo - curved arrow pointing right."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Curved arrow (approximated with arc and lines)
    arc_rect = pygame.Rect(cx - size, cy - size//2, size * 2, size)
    pygame.draw.arc(surface, color, arc_rect, 0, math.pi - 0.5, 2)
    
    # Arrow head
    ax, ay = cx + size, cy
    pygame.draw.line(surface, color, (ax, ay), (ax - 5, ay - 5), 2)
    pygame.draw.line(surface, color, (ax, ay), (ax - 5, ay + 5), 2)


def draw_icon_delete(surface, rect, color):
    """Delete/Clear - X mark."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    pygame.draw.line(surface, color, (cx - size, cy - size), (cx + size, cy + size), 3)
    pygame.draw.line(surface, color, (cx + size, cy - size), (cx - size, cy + size), 3)


def draw_icon_atomize(surface, rect, color):
    """Atomize - scattered dots pattern."""
    cx, cy = rect.center
    
    # Central shape outline
    size = min(rect.width, rect.height) // 2 - 5
    pygame.draw.rect(surface, color, (cx - size, cy - size//2, size*2, size), 1)
    
    # Scattered dots
    dots = [(-6, -3), (-2, 2), (3, -4), (6, 1), (0, 5), (-4, 4), (5, 4)]
    for dx, dy in dots:
        pygame.draw.circle(surface, color, (cx + dx, cy + dy), 2)


def draw_icon_ghost(surface, rect, color):
    """Ghost/Physical toggle - ghost shape."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Ghost body (rounded top, wavy bottom)
    pygame.draw.arc(surface, color, (cx - size, cy - size, size*2, size*2), 0, math.pi, 2)
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size, cy + size//2), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size, cy + size//2), 2)
    
    # Wavy bottom
    wave_y = cy + size//2
    pygame.draw.line(surface, color, (cx - size, wave_y), (cx - size//2, wave_y + 4), 2)
    pygame.draw.line(surface, color, (cx - size//2, wave_y + 4), (cx, wave_y), 2)
    pygame.draw.line(surface, color, (cx, wave_y), (cx + size//2, wave_y + 4), 2)
    pygame.draw.line(surface, color, (cx + size//2, wave_y + 4), (cx + size, wave_y), 2)
    
    # Eyes
    pygame.draw.circle(surface, color, (cx - 4, cy - 2), 2)
    pygame.draw.circle(surface, color, (cx + 4, cy - 2), 2)


def draw_icon_save(surface, rect, color):
    """Save - floppy disk icon."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Outer square
    pygame.draw.rect(surface, color, (cx - size, cy - size, size*2, size*2), 2)
    
    # Top notch (save slot)
    pygame.draw.rect(surface, color, (cx - size//2, cy - size, size, size//2 + 2), 0)
    pygame.draw.rect(surface, color, (cx - size//2, cy - size, size, size//2 + 2), 1)
    
    # Label area at bottom
    pygame.draw.rect(surface, color, (cx - size + 3, cy + 2, size*2 - 6, size - 4), 1)


def draw_icon_exit(surface, rect, color):
    """Exit - door with arrow."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Door frame
    pygame.draw.rect(surface, color, (cx - size, cy - size, size + 2, size*2), 2)
    
    # Arrow pointing out
    ax = cx + size//2
    pygame.draw.line(surface, color, (cx - 2, cy), (cx + size, cy), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy - 4), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy + 4), 2)


# =============================================================================
# Icon Registry - Maps names to drawing functions
# =============================================================================

ICONS = {
    # Tools
    'brush': draw_icon_brush,
    'select': draw_icon_select,
    'line': draw_icon_line,
    'rect': draw_icon_rect,
    'circle': draw_icon_circle,
    'point': draw_icon_point,
    'ref_line': draw_icon_ref_line,
    
    # Constraints
    'coincident': draw_icon_coincident,
    'collinear': draw_icon_collinear,
    'midpoint': draw_icon_midpoint,
    'length': draw_icon_length,
    'equal': draw_icon_equal,
    'parallel': draw_icon_parallel,
    'perpendicular': draw_icon_perpendicular,
    'horizontal': draw_icon_horizontal,
    'vertical': draw_icon_vertical,
    'angle': draw_icon_angle,
    'extend': draw_icon_extend,
    
    # Actions
    'play': draw_icon_play,
    'pause': draw_icon_pause,
    'undo': draw_icon_undo,
    'redo': draw_icon_redo,
    'delete': draw_icon_delete,
    'atomize': draw_icon_atomize,
    'ghost': draw_icon_ghost,
    'save': draw_icon_save,
    'exit': draw_icon_exit,
}


def get_icon(name):
    """Get an icon drawing function by name."""
    return ICONS.get(name, None)
