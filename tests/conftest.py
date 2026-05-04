"""Shared fixtures for the Flow State test suite."""

# Force SDL to a headless driver before pygame initializes — must run before
# any pygame import (transitive imports happen as soon as ui.* is touched).
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest
import pygame

from model.sketch import Sketch
from engine.simulation import Simulation
from engine.compiler import Compiler
from core.scene import Scene
from core.session import Session


@pytest.fixture(scope="session", autouse=True)
def pygame_session():
    """Initialize pygame once per test session (headless)."""
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


@pytest.fixture
def sketch():
    """Fresh Sketch instance with default preset materials."""
    return Sketch()


@pytest.fixture
def simulation():
    """Fresh Simulation with Numba warmup skipped."""
    return Simulation(skip_warmup=True)


@pytest.fixture
def scene():
    """Fresh Scene with Numba warmup skipped."""
    return Scene(skip_warmup=True)


@pytest.fixture
def session():
    """Fresh Session (camera, selection, constraint_builder, status, etc.)."""
    return Session()


@pytest.fixture
def compiler(sketch, simulation):
    """Compiler bridging the test sketch and simulation directly (no MaterialManager)."""
    return Compiler(sketch, simulation)


# ---------------------------------------------------------------------------
# Layout helper for tools that need it
# ---------------------------------------------------------------------------

def make_layout(left=250, top=30, width=800, height=600, right_panel_width=250):
    """Construct a screen-layout dict matching what FlowStateApp builds at runtime.

    Tools dispatch on layout['LEFT_X'] / ['RIGHT_X'] / ['MID_X'] / ['MID_W'] / ['MID_H'].
    """
    return {
        "LEFT_X": left,
        "MID_X": left,
        "MID_W": width,
        "MID_H": height,
        "MID_Y": top,
        "RIGHT_X": left + width,
        "scene": (left, top, width, height),
    }


@pytest.fixture
def layout():
    return make_layout()


# ---------------------------------------------------------------------------
# Fake app for ToolContext
# ---------------------------------------------------------------------------

class _SilentSoundManager:
    def play_sound(self, sound_id):
        pass


class FakeApp:
    """Minimal app surface needed by ToolContext.

    A real ToolContext only reads from .scene, .session, .layout, .sound_manager
    (plus optionally .input_handler for clear_constraint_ui). This lets us build
    one with a real Scene + Session without dragging in pygame display init or
    the FlowStateApp god object.
    """

    def __init__(self, scene, session, layout):
        self.scene = scene
        self.session = session
        self.layout = layout
        self.sound_manager = _SilentSoundManager()
        self.input_handler = None


@pytest.fixture
def fake_app(scene, session, layout):
    return FakeApp(scene, session, layout)


@pytest.fixture
def tool_ctx(fake_app):
    """Real ToolContext wrapping a FakeApp — ready for tool tests."""
    from core.tool_context import ToolContext
    return ToolContext(fake_app)


# ---------------------------------------------------------------------------
# Event helper
# ---------------------------------------------------------------------------

def make_event(event_type, **kwargs):
    """Construct a pygame.event.Event with sensible defaults for mouse/key tests."""
    return pygame.event.Event(event_type, kwargs)
