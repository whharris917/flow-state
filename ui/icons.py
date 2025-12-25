"""
Icons - Hybrid Icon System for UI Buttons

Loads professional PNG icons from assets/icons/ with dynamic color tinting.
Falls back to procedural pygame drawing for icons without PNG assets.

Each icon is accessed via get_icon(name), which returns a callable:
    icon_func(surface, rect, color) -> draws icon centered in rect

PNG icons are white-on-transparent and tinted to match the requested color.
"""

import pygame
import math
import os

# =============================================================================
# PNG Icon Loader with Color Tinting
# =============================================================================

class IconCache:
    """
    Manages loading and tinting of PNG icons.
    
    PNG icons are expected to be white (255,255,255) on transparent background.
    Color tinting multiplies the white pixels by the target color.
    """
    
    def __init__(self, icon_dir="assets/icons", default_size=32):
        self.icon_dir = icon_dir
        self.default_size = default_size
        self.base_icons = {}  # name -> pygame.Surface (white on transparent)
        self.tinted_cache = {}  # (name, color_tuple) -> pygame.Surface
        self._load_all()
    
    def _load_all(self):
        """Load all PNG icons from the icon directory."""
        if not os.path.exists(self.icon_dir):
            print(f"[Icons] Warning: Icon directory '{self.icon_dir}' not found")
            return
        
        for filename in os.listdir(self.icon_dir):
            if filename.endswith('.png'):
                name = filename[:-4]  # Remove .png extension
                path = os.path.join(self.icon_dir, filename)
                try:
                    surf = pygame.image.load(path).convert_alpha()
                    self.base_icons[name] = surf
                except pygame.error as e:
                    print(f"[Icons] Failed to load {filename}: {e}")
        
        print(f"[Icons] Loaded {len(self.base_icons)} PNG icons from {self.icon_dir}")
    
    def get_tinted(self, name, color, size=None):
        """
        Get a tinted version of the icon.
        
        Args:
            name: Icon name (without .png extension)
            color: RGB tuple (r, g, b) to tint the icon
            size: Optional (width, height) to scale to. Defaults to icon's native size.
        
        Returns:
            pygame.Surface with tinted icon, or None if not found
        """
        if name not in self.base_icons:
            return None
        
        # Normalize color to tuple for cache key
        color_key = tuple(color[:3]) if hasattr(color, '__iter__') else (color, color, color)
        size_key = size if size else 'native'
        cache_key = (name, color_key, size_key)
        
        if cache_key in self.tinted_cache:
            return self.tinted_cache[cache_key]
        
        # Create tinted version
        base = self.base_icons[name]
        
        # Scale if needed
        if size and size != (base.get_width(), base.get_height()):
            base = pygame.transform.smoothscale(base, size)
        
        # Create tinted surface
        tinted = base.copy()
        
        # Apply color tint using BLEND_RGBA_MULT
        # This multiplies each pixel's RGB by the color (normalized to 0-1)
        tint_surface = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
        tint_surface.fill((*color_key, 255))
        tinted.blit(tint_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Cache and return
        self.tinted_cache[cache_key] = tinted
        return tinted
    
    def has_icon(self, name):
        """Check if a PNG icon exists for the given name."""
        return name in self.base_icons
    
    def clear_cache(self):
        """Clear the tinted icon cache (call if memory is a concern)."""
        self.tinted_cache.clear()


# Global icon cache instance (initialized lazily)
_icon_cache = None

def _get_cache():
    """Get or create the global icon cache."""
    global _icon_cache
    if _icon_cache is None:
        _icon_cache = IconCache()
    return _icon_cache


# =============================================================================
# PNG Icon Drawing Functions
# =============================================================================

def _make_png_icon_func(png_name):
    """
    Factory function that creates a draw function for a PNG icon.
    
    Returns a callable: func(surface, rect, color) that draws the icon.
    """
    def draw_func(surface, rect, color):
        cache = _get_cache()
        # Calculate size with padding
        padding = 6
        size = (rect.width - padding, rect.height - padding)
        
        tinted = cache.get_tinted(png_name, color, size)
        if tinted:
            # Center the icon in the rect
            x = rect.x + (rect.width - tinted.get_width()) // 2
            y = rect.y + (rect.height - tinted.get_height()) // 2
            surface.blit(tinted, (x, y))
    
    return draw_func


# =============================================================================
# Procedural Icon Drawing (Fallbacks)
# =============================================================================

def draw_icon_ref_line(surface, rect, color):
    """Reference line tool - dashed diagonal line."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 4
    
    p1 = (cx - size, cy + size)
    p2 = (cx + size, cy - size)
    
    # Draw dashed line
    segments = 5
    for i in range(segments):
        if i % 2 == 0:
            t1 = i / segments
            t2 = (i + 1) / segments
            x1 = int(p1[0] + (p2[0] - p1[0]) * t1)
            y1 = int(p1[1] + (p2[1] - p1[1]) * t1)
            x2 = int(p1[0] + (p2[0] - p1[0]) * t2)
            y2 = int(p1[1] + (p2[1] - p1[1]) * t2)
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)
    
    # Endpoints
    pygame.draw.circle(surface, color, p1, 3)
    pygame.draw.circle(surface, color, p2, 3)


def draw_icon_collinear(surface, rect, color):
    """Collinear constraint - three points on a line."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Diagonal line
    p1 = (cx - size, cy + size // 2)
    p2 = (cx + size, cy - size // 2)
    pygame.draw.line(surface, color, p1, p2, 2)
    
    # Three points on the line
    for t in [0.15, 0.5, 0.85]:
        px = int(p1[0] + (p2[0] - p1[0]) * t)
        py = int(p1[1] + (p2[1] - p1[1]) * t)
        pygame.draw.circle(surface, color, (px, py), 3)


def draw_icon_horizontal(surface, rect, color):
    """Horizontal constraint - horizontal line with indicator."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Horizontal line
    pygame.draw.line(surface, color, (cx - size, cy), (cx + size, cy), 2)
    
    # Endpoints
    pygame.draw.circle(surface, color, (cx - size, cy), 3)
    pygame.draw.circle(surface, color, (cx + size, cy), 3)
    
    # "H" indicator below
    pygame.draw.line(surface, color, (cx - 4, cy + 6), (cx - 4, cy + 12), 1)
    pygame.draw.line(surface, color, (cx + 4, cy + 6), (cx + 4, cy + 12), 1)
    pygame.draw.line(surface, color, (cx - 4, cy + 9), (cx + 4, cy + 9), 1)


def draw_icon_vertical(surface, rect, color):
    """Vertical constraint - vertical line with indicator."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Vertical line
    pygame.draw.line(surface, color, (cx, cy - size), (cx, cy + size), 2)
    
    # Endpoints
    pygame.draw.circle(surface, color, (cx, cy - size), 3)
    pygame.draw.circle(surface, color, (cx, cy + size), 3)
    
    # "V" indicator to the right
    pygame.draw.line(surface, color, (cx + 6, cy - 4), (cx + 9, cy + 4), 1)
    pygame.draw.line(surface, color, (cx + 12, cy - 4), (cx + 9, cy + 4), 1)


def draw_icon_angle(surface, rect, color):
    """Angle constraint - two lines forming an angle with arc."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 2
    
    # Two lines forming an angle
    origin = (cx - size // 2, cy + size // 2)
    p1 = (cx + size, cy + size // 2)  # Horizontal
    p2 = (cx + size // 3, cy - size)   # Angled up
    
    pygame.draw.line(surface, color, origin, p1, 2)
    pygame.draw.line(surface, color, origin, p2, 2)
    
    # Arc to show angle
    arc_rect = pygame.Rect(origin[0] - 8, origin[1] - 8, 16, 16)
    pygame.draw.arc(surface, color, arc_rect, 0, math.pi / 3 + 0.2, 2)


def draw_icon_undo(surface, rect, color):
    """Undo - curved arrow pointing left."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Curved arrow (approximated with arc and lines)
    arc_rect = pygame.Rect(cx - size, cy - size // 2, size * 2, size)
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
    arc_rect = pygame.Rect(cx - size, cy - size // 2, size * 2, size)
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


def draw_icon_ghost(surface, rect, color):
    """Ghost/Physical toggle - ghost shape."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Ghost body (rounded top, wavy bottom)
    pygame.draw.arc(surface, color, (cx - size, cy - size, size * 2, size * 2), 0, math.pi, 2)
    pygame.draw.line(surface, color, (cx - size, cy), (cx - size, cy + size // 2), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size, cy + size // 2), 2)
    
    # Wavy bottom
    wave_y = cy + size // 2
    pygame.draw.line(surface, color, (cx - size, wave_y), (cx - size // 2, wave_y + 4), 2)
    pygame.draw.line(surface, color, (cx - size // 2, wave_y + 4), (cx, wave_y), 2)
    pygame.draw.line(surface, color, (cx, wave_y), (cx + size // 2, wave_y + 4), 2)
    pygame.draw.line(surface, color, (cx + size // 2, wave_y + 4), (cx + size, wave_y), 2)
    
    # Eyes
    pygame.draw.circle(surface, color, (cx - 4, cy - 2), 2)
    pygame.draw.circle(surface, color, (cx + 4, cy - 2), 2)


def draw_icon_save(surface, rect, color):
    """Save - floppy disk icon."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Outer square
    pygame.draw.rect(surface, color, (cx - size, cy - size, size * 2, size * 2), 2)
    
    # Top notch (save slot)
    pygame.draw.rect(surface, color, (cx - size // 2, cy - size, size, size // 2 + 2), 0)
    
    # Label area at bottom
    pygame.draw.rect(surface, color, (cx - size + 3, cy + 2, size * 2 - 6, size - 4), 1)


def draw_icon_exit(surface, rect, color):
    """Exit - door with arrow."""
    cx, cy = rect.center
    size = min(rect.width, rect.height) // 2 - 3
    
    # Door frame
    pygame.draw.rect(surface, color, (cx - size, cy - size, size + 2, size * 2), 2)
    
    # Arrow pointing out
    pygame.draw.line(surface, color, (cx - 2, cy), (cx + size, cy), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy - 4), 2)
    pygame.draw.line(surface, color, (cx + size, cy), (cx + size - 5, cy + 4), 2)


# =============================================================================
# Icon Registry - Maps names to drawing functions
# =============================================================================

# PNG icon name mappings (png filename without extension -> internal name)
PNG_MAPPINGS = {
    # Tools
    'brush': 'brush',
    'select': 'select',
    'line': 'line',
    'rectangle': 'rect',  # PNG is 'rectangle', internal name is 'rect'
    'circle': 'circle',
    'point': 'point',
    
    # Constraints
    'coincident': 'coincident',
    'midpoint': 'midpoint',
    'equal': 'equal',
    'fix': 'length',  # PNG is 'fix', internal name is 'length'
    'parallel': 'parallel',
    'perpendicular': 'perpendicular',
    'extend': 'extend',
    'anchor': 'anchor',  # NEW
    
    # Actions
    'play': 'play',
    'pause': 'pause',
    'circle_play': 'anim_play',  # NEW - for animation play
    'circle_pause': 'anim_pause',  # NEW - for animation pause
    'atom': 'atomize',  # PNG is 'atom', internal name is 'atomize'
    'hide': 'hide',  # NEW
    'unhide': 'unhide',  # NEW
}

# Procedural fallbacks for icons without PNGs
PROCEDURAL_ICONS = {
    'ref_line': draw_icon_ref_line,
    'collinear': draw_icon_collinear,
    'horizontal': draw_icon_horizontal,
    'vertical': draw_icon_vertical,
    'angle': draw_icon_angle,
    'undo': draw_icon_undo,
    'redo': draw_icon_redo,
    'delete': draw_icon_delete,
    'ghost': draw_icon_ghost,
    'save': draw_icon_save,
    'exit': draw_icon_exit,
}

# Build the final icon registry
_ICONS = None

def _build_registry():
    """Build the icon registry, preferring PNGs over procedural."""
    global _ICONS
    if _ICONS is not None:
        return _ICONS
    
    _ICONS = {}
    cache = _get_cache()
    
    # First, add all procedural fallbacks
    _ICONS.update(PROCEDURAL_ICONS)
    
    # Then, override with PNG icons where available
    for png_name, internal_name in PNG_MAPPINGS.items():
        if cache.has_icon(png_name):
            _ICONS[internal_name] = _make_png_icon_func(png_name)
        else:
            print(f"[Icons] PNG not found for '{png_name}', using fallback if available")
    
    return _ICONS


def get_icon(name):
    """
    Get an icon drawing function by name.
    
    Args:
        name: Icon name (e.g., 'select', 'brush', 'coincident')
    
    Returns:
        Callable that draws the icon: func(surface, rect, color)
        Returns None if icon not found.
    """
    registry = _build_registry()
    return registry.get(name, None)


def get_icon_fixed_size(name, size=32):
    """
    Get an icon drawing function that maintains a fixed size regardless of button dimensions.
    
    Use this for wide buttons where you want the icon centered at its native size
    rather than stretched to fill the button.
    
    Args:
        name: Icon name (e.g., 'atomize', 'extend')
        size: Fixed icon size in pixels (default 32)
    
    Returns:
        Callable that draws the icon: func(surface, rect, color)
        Returns None if icon not found.
    """
    base_icon = get_icon(name)
    if base_icon is None:
        return None
    
    def draw_fixed(surface, rect, color):
        # Create a fixed-size rect centered in the button's rect
        fixed_rect = pygame.Rect(0, 0, size, size)
        fixed_rect.center = rect.center
        base_icon(surface, fixed_rect, color)
    
    return draw_fixed


def list_icons():
    """Return a list of all available icon names."""
    registry = _build_registry()
    return list(registry.keys())


def reload_icons():
    """Reload all icons from disk (useful for hot-reloading during development)."""
    global _icon_cache, _ICONS
    _icon_cache = None
    _ICONS = None
    _build_registry()
    print("[Icons] Reloaded all icons")


# =============================================================================
# Legacy Compatibility - Direct access to icon functions
# =============================================================================

# These allow direct imports like: from ui.icons import ICON_SELECT
# They're resolved at first access
class _IconProxy:
    """Lazy proxy for backward compatibility with ICON_* constants."""
    def __init__(self, name):
        self._name = name
        self._func = None
    
    def __call__(self, surface, rect, color):
        if self._func is None:
            self._func = get_icon(self._name)
        if self._func:
            self._func(surface, rect, color)


# Legacy constants (if any code uses ICON_* directly)
ICON_BRUSH = _IconProxy('brush')
ICON_SELECT = _IconProxy('select')
ICON_LINE = _IconProxy('line')
ICON_RECT = _IconProxy('rect')
ICON_CIRCLE = _IconProxy('circle')
ICON_POINT = _IconProxy('point')