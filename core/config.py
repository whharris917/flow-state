"""
Configuration - Global Constants and Theme

All application-wide constants and visual theme definitions.
"""

import numpy as np

# =============================================================================
# Window Layout
# =============================================================================
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 930 
PANEL_LEFT_WIDTH = 350 
PANEL_RIGHT_WIDTH = 350
TOP_MENU_H = 40

# =============================================================================
# Application Modes
# =============================================================================
MODE_SIM = 0
MODE_EDITOR = 1

# =============================================================================
# Tool IDs
# =============================================================================
TOOL_BRUSH = 0
TOOL_LINE = 1
TOOL_REF = 2
TOOL_POINT = 3
TOOL_RECT = 4
TOOL_CIRCLE = 5
TOOL_SELECT = 6
TOOL_SOURCE = 7  # ProcessObject: Particle emitter

# =============================================================================
# World & Physics Defaults
# =============================================================================
DEFAULT_WORLD_SIZE = 50.0
DEFAULT_DT = 0.002
DEFAULT_GRAVITY = 0.0
DEFAULT_DAMPING = 1.0 
DEFAULT_DRAW_M = 10
DEFAULT_SKIN_DISTANCE = 0.5

# =============================================================================
# Visual Theme (Professional Dark)
# =============================================================================
PARTICLE_RADIUS_SCALE = 0.4
BACKGROUND_COLOR = (18, 20, 24)       # Deep Charcoal
GRID_COLOR = (30, 34, 40)             # Soft Slate
PANEL_BG_COLOR = (30, 34, 40)         # Panel Background
PANEL_BORDER_COLOR = (50, 55, 65)     # Borders

# Accents & Logic Colors
COLOR_ACCENT = (70, 160, 220)         # Industrial Blue
COLOR_SUCCESS = (110, 190, 130)       # Sage Green
COLOR_DANGER = (210, 100, 100)        # Soft Red
COLOR_WARNING = (230, 180, 80)        # Amber
COLOR_TEXT = (235, 235, 240)          # Soft White
COLOR_TEXT_DIM = (160, 165, 175)      # Muted Text

# Input Fields
COLOR_INPUT_BG = (45, 48, 55)
COLOR_INPUT_ACTIVE = (60, 65, 75)

# Particles
COLOR_DYNAMIC = (70, 160, 220)        # Match Accent
COLOR_STATIC = (206, 145, 120)        # Clay/Orange

# ProcessObjects
COLOR_SOURCE = (100, 200, 255)        # Light blue for Sources
COLOR_SOURCE_DISABLED = (80, 80, 90)  # Gray for disabled Sources

# Default Atom Properties
ATOM_SIGMA = 1.0
ATOM_EPSILON = 1.0
ATOM_MASS = 1.0

# =============================================================================
# Runtime Flags (set by command line)
# =============================================================================
DEBUG = False
SCREEN_WIDTH = WINDOW_WIDTH
SCREEN_HEIGHT = WINDOW_HEIGHT
FULLSCREEN = False
