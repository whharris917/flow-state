"""
SinkTool - Tool for Creating Particle Sink Absorbers (Drains)

Two-click placement, mirroring SourceTool:
1. First click: Set center position (with snap support)
2. Second click: Set radius

The Sink's center handle participates in constraints like any Point.
The visual treatment is dashed red (vs. Source's dashed blue) so the
two are immediately distinguishable in the viewport.
"""

import pygame
import math

import core.config as config
import core.utils as utils

from core.sink_commands import AddSinkCommand
from model.process_objects import Sink, SinkProperties
from ui.tools import Tool


# Sink visual palette — red, distinct from Source's blue (100, 180, 255)
SINK_COLOR = (220, 80, 80)
SINK_COLOR_DARK = (160, 50, 50)


class SinkTool(Tool):
    """
    Tool for creating Sink (particle absorber) ProcessObjects.

    Two-click workflow identical to SourceTool. During step 2 a preview
    dashed red circle follows the mouse.
    """

    def __init__(self, ctx):
        super().__init__(ctx, name="Sink")

        self.center = None
        self.preview_radius = None
        self.center_snap = None

    def activate(self):
        self._reset()

    def deactivate(self):
        self._reset()
        self.ctx.snap_target = None

    def _reset(self):
        self.center = None
        self.preview_radius = None
        self.center_snap = None

    def cancel(self):
        if self.center is not None:
            self._reset()
            self.ctx.set_status("Sink cancelled")
        self.ctx.snap_target = None

    def handle_event(self, event, layout):
        """Handle pygame events. Returns True if event was consumed."""

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Y-gate matches SourceTool/BrushTool — ignore clicks in the
            # top menu bar or below the viewport.
            if not (layout['LEFT_X'] < mx < layout['RIGHT_X']):
                return False
            if not (config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
                return False

            wx, wy, snap = self._get_snapped(mx, my, layout)

            if self.center is None:
                # First click: set center
                self.center = (wx, wy)
                self.center_snap = snap
                self.ctx.set_status("Click to set sink radius")
                return True
            else:
                # Second click: set radius and create Sink
                radius = math.hypot(wx - self.center[0], wy - self.center[1])
                radius = max(0.5, radius)

                self._create_sink(radius)
                self._reset()
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos

            if self.center is not None:
                wx, wy = self._get_world_pos(mx, my, layout)
                self.preview_radius = math.hypot(wx - self.center[0], wy - self.center[1])
                self.preview_radius = max(0.5, self.preview_radius)
            else:
                self._update_hover_snap(mx, my, layout)

            return False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
                return True

        return False

    def update(self, dt, layout):
        pass

    def draw_overlay(self, screen, renderer, layout):
        """Draw tool-specific overlay graphics."""
        mx, my = pygame.mouse.get_pos()

        if not (layout['MID_X'] < mx < layout['RIGHT_X']):
            return

        if self.center is not None:
            cx_screen, cy_screen = self._world_to_screen(
                self.center[0], self.center[1], layout
            )
            pygame.draw.circle(screen, SINK_COLOR, (cx_screen, cy_screen), 5)
            pygame.draw.circle(screen, (255, 255, 255), (cx_screen, cy_screen), 5, 1)

            if self.preview_radius is not None:
                screen_radius = self._world_to_screen_distance(self.preview_radius, layout)
                self._draw_dashed_circle(
                    screen,
                    (cx_screen, cy_screen),
                    int(screen_radius),
                    SINK_COLOR,
                )

                pygame.draw.line(
                    screen, SINK_COLOR,
                    (cx_screen, cy_screen), (mx, my), 1
                )
        else:
            # Crosshair at mouse position before first click
            pygame.draw.circle(screen, SINK_COLOR, (mx, my), 4)

    def _create_sink(self, radius):
        """Create the Sink and add to scene via AddSinkCommand."""
        cmd = AddSinkCommand(
            scene=self.ctx._get_scene(),
            center=self.center,
            radius=radius,
            properties=SinkProperties(),
        )
        self.ctx.execute(cmd)
        sink = cmd.sink

        # If we snapped to a point with Ctrl held, create a coincident constraint
        if self.center_snap and pygame.key.get_mods() & pygame.KMOD_CTRL:
            handle_indices = sink.get_handle_indices(self.ctx._get_sketch())
            if 'center' in handle_indices:
                center_idx = handle_indices['center']
                snap_entity, snap_pt = self.center_snap

                constraint_cmd = self.ctx.create_coincident_command(
                    center_idx, 0, snap_entity, snap_pt
                )
                if constraint_cmd is not None:
                    self.ctx.execute(constraint_cmd)

        self.ctx.set_status(f"Sink created (r={radius:.1f})")

    # =========================================================================
    # Coordinate / snap helpers (verbatim from SourceTool)
    # =========================================================================

    def _get_world_pos(self, mx, my, layout):
        pan_x, pan_y = self.ctx.pan
        return utils.screen_to_sim(
            mx, my,
            self.ctx.zoom,
            pan_x,
            pan_y,
            self.ctx.world_size,
            layout
        )

    def _get_snapped(self, mx, my, layout):
        mods = pygame.key.get_mods()
        snap_to_points = bool(mods & pygame.KMOD_CTRL)
        constrain_to_axis = bool(mods & pygame.KMOD_SHIFT)
        pan_x, pan_y = self.ctx.pan
        return utils.get_snapped_pos(
            mx, my,
            self.ctx._get_sketch().entities,
            self.ctx.zoom,
            pan_x,
            pan_y,
            self.ctx.world_size,
            layout,
            snap_to_points=snap_to_points,
            constrain_to_axis=constrain_to_axis
        )

    def _update_hover_snap(self, mx, my, layout):
        _, _, snap = self._get_snapped(mx, my, layout)
        self.ctx.snap_target = snap

    def _world_to_screen(self, wx, wy, layout):
        pan_x, pan_y = self.ctx.pan
        return utils.sim_to_screen(
            wx, wy,
            self.ctx.zoom,
            pan_x,
            pan_y,
            self.ctx.world_size,
            layout
        )

    def _world_to_screen_distance(self, world_dist, layout):
        zoom = self.ctx.zoom
        world_size = self.ctx.world_size
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * zoom
        return world_dist * final_scale

    def _draw_dashed_circle(self, screen, center, radius, color):
        """Draw a dashed circle preview (verbatim from SourceTool)."""
        if radius < 5:
            pygame.draw.circle(screen, color[:3], center, radius, 1)
            return

        circumference = 2 * math.pi * radius
        dash_length = 8
        gap_length = 4
        segment_length = dash_length + gap_length
        num_segments = max(8, int(circumference / segment_length))

        for i in range(num_segments):
            if i % 2 == 0:
                start_angle = (i / num_segments) * 2 * math.pi
                end_angle = ((i + 0.6) / num_segments) * 2 * math.pi

                steps = max(2, int((end_angle - start_angle) * radius / 5))
                points = []
                for j in range(steps + 1):
                    angle = start_angle + (end_angle - start_angle) * j / steps
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((int(x), int(y)))

                if len(points) >= 2:
                    pygame.draw.lines(screen, color[:3], False, points, 2)
