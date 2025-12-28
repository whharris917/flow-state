"""
Configuration Settings
"""

import pygame

# === WINDOW SETTINGS ===
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
WINDOW_TITLE = "Flow State"
FPS = 60

# === UI DIMENSIONS & SCALING ===
UI_SCALE = 1.0

def scale(val):
    """Global scaling helper."""
    return int(val * UI_SCALE)

# Default base values (at 1.0 scale)
BASE_PANEL_WIDTH = 250
BASE_TOP_MENU_H = 30

# These will be updated by set_ui_scale()
PANEL_LEFT_WIDTH = BASE_PANEL_WIDTH
PANEL_RIGHT_WIDTH = BASE_PANEL_WIDTH
TOP_MENU_H = BASE_TOP_MENU_H

def set_ui_scale(new_scale):
    """Recalculate all dimensional constants based on new scale."""
    global UI_SCALE, PANEL_LEFT_WIDTH, PANEL_RIGHT_WIDTH, TOP_MENU_H
    
    # Clamp scale to reasonable limits (0.5x to 3.0x)
    UI_SCALE = max(0.5, min(3.0, new_scale))
    
    # Recalculate derived constants
    PANEL_LEFT_WIDTH = int(BASE_PANEL_WIDTH * UI_SCALE)
    PANEL_RIGHT_WIDTH = int(BASE_PANEL_WIDTH * UI_SCALE)
    TOP_MENU_H = int(BASE_TOP_MENU_H * UI_SCALE)

# Initialize once
set_ui_scale(1.0)


# === COLORS ===
BACKGROUND_COLOR = (20, 20, 20)      # Dark Gray/Black
GRID_COLOR = (40, 40, 40)            # Faint Grid lines
COLOR_STATIC = (100, 100, 120)       # Walls/Obstacles
COLOR_DYNAMIC = (50, 150, 255)       # Fluid Particles
COLOR_ACCENT = (70, 130, 180)        # UI Accent (Steel Blue)
COLOR_TEXT = (220, 220, 220)         # Off-white text
COLOR_TEXT_DIM = (150, 150, 150)
COLOR_DANGER = (200, 60, 60)         # Red
COLOR_SUCCESS = (60, 200, 100)       # Green
COLOR_WARNING = (255, 200, 50)       # Yellow/Orange

PANEL_BG_COLOR = (30, 30, 32)
PANEL_BORDER_COLOR = (60, 60, 65)

COLOR_INPUT_BG = (40, 40, 42)
COLOR_INPUT_ACTIVE = (50, 50, 60)

COLOR_SOURCE = (100, 255, 100)
COLOR_SOURCE_DISABLED = (60, 60, 60)

# === SIMULATION CONSTANTS ===
PARTICLE_RADIUS_SCALE = 0.5  # Visual scaling factor relative to sigma (Reduced from 5.0)
DEFAULT_WORLD_SIZE = 50.0    # Simulation world width in simulation units
DEFAULT_GRAVITY = 9.81
DEFAULT_DAMPING = 0.99
DEFAULT_DT = 0.002
DEFAULT_SKIN_DISTANCE = 0.3
DEFAULT_DRAW_M = 10.0        # Default physics steps per frame

# === LENNARD-JONES PARAMETERS ===
ATOM_SIGMA = 1.0     # Particle Size
ATOM_EPSILON = 1.0   # Interaction Strength
ATOM_MASS = 1.0

# === TOOL IDS ===
TOOL_SELECT = 0
TOOL_BRUSH = 1
TOOL_LINE = 2
TOOL_RECT = 3
TOOL_CIRCLE = 4
TOOL_POINT = 5
TOOL_REF = 6
TOOL_SOURCE = 7

# === MODES ===
MODE_SIM = 0
MODE_EDITOR = 1