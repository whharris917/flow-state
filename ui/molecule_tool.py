"""
MoleculeTool — Click-to-place a molecule template.

The tool holds a reference to a selected MoleculeTemplate and instances it
at the click location via AddMoleculeCommand. A live preview follows the
mouse so the user can see what the placement will look like.

For R2 the selected template is set via `tool.set_template(template)`; the
right-panel palette in R4 dispatches to this method when a palette button
is clicked. If no template is selected, clicks fall through with a status
message rather than crashing.

Rotation control is deferred — placements use 0 rotation. R4 may add a
right-drag-to-rotate gesture if the Lead validates the feature.
"""

import pygame
import math

import core.config as config
import core.utils as utils

from core.molecule_commands import AddMoleculeCommand
from ui.tools import Tool


# Visual palette for the preview overlay. Distinct from Source (blue) and
# Sink (red). Cyan-green: clearly "constructive" without overlap with
# either existing ProcessObject colour.
MOLECULE_PREVIEW_COLOR = (120, 220, 200)
MOLECULE_PREVIEW_BOND_COLOR = (100, 180, 160)


class MoleculeTool(Tool):
    """Click on the canvas to place the currently selected molecule template."""

    def __init__(self, ctx):
        super().__init__(ctx, name="Molecule")
        self.template = None    # Current MoleculeTemplate; None = nothing to place
        self.rotation = 0.0     # Reserved for future rotation control

    # =========================================================================
    # Public selection API (called by right-panel palette in R4)
    # =========================================================================

    def set_template(self, template):
        """Select a MoleculeTemplate to place on next click. None unsets."""
        self.template = template
        if template is not None:
            self.ctx.set_status(
                f"Molecule '{template.name}' selected — click to place"
            )

    # =========================================================================
    # Tool lifecycle
    # =========================================================================

    def activate(self):
        if self.template is None:
            # Default to the first molecule in the sketch's palette so the
            # tool is usable even before R4 wires up the right-panel buttons.
            sketch = self.ctx._get_sketch()
            if sketch.molecules:
                first_name = next(iter(sketch.molecules))
                self.template = sketch.molecules[first_name]
                self.ctx.set_status(
                    f"Molecule '{first_name}' (default) — click to place"
                )
            else:
                self.ctx.set_status("No molecules in palette")

    def deactivate(self):
        pass

    def cancel(self):
        """ESC clears the selected template."""
        if self.template is not None:
            self.template = None
            self.ctx.set_status("Molecule placement cancelled")

    def update(self, dt, layout):
        pass

    # =========================================================================
    # Event handling
    # =========================================================================

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Y-gate identical to Source/Sink tools — ignore clicks in the
            # top menu bar or panels.
            if not (layout['LEFT_X'] < mx < layout['RIGHT_X']):
                return False
            if not (config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
                return False

            if self.template is None:
                self.ctx.set_status("No molecule selected")
                return True

            wx, wy = self._get_world_pos(mx, my, layout)
            self._place_molecule(wx, wy)
            return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
                return True

        return False

    # =========================================================================
    # Placement
    # =========================================================================

    def _place_molecule(self, wx, wy):
        if self.template is None:
            return
        cmd = AddMoleculeCommand(
            scene=self.ctx._get_scene(),
            template=self.template,
            world_pos=(wx, wy),
            rotation=self.rotation,
        )
        ok = self.ctx.execute(cmd)
        if ok:
            self.ctx.set_status(
                f"Placed '{self.template.name}' at ({wx:.1f}, {wy:.1f})"
            )

    # =========================================================================
    # Preview overlay
    # =========================================================================

    def draw_overlay(self, screen, renderer, layout):
        """Draw a preview of the molecule at the cursor before placement."""
        if self.template is None:
            return
        mx, my = pygame.mouse.get_pos()
        if not (layout['MID_X'] < mx < layout['RIGHT_X']):
            return
        if not (config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
            return

        wx, wy = self._get_world_pos(mx, my, layout)

        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)

        # Compute screen positions for each atom in the template
        atom_screen = []
        for ma in self.template.atoms:
            ax = wx + cos_r * ma.x - sin_r * ma.y
            ay = wy + sin_r * ma.x + cos_r * ma.y
            sx, sy = self._world_to_screen(ax, ay, layout)
            atom_screen.append((sx, sy))

        # Draw bonds first (so atoms cover them at endpoints)
        for mb in self.template.bonds:
            a = atom_screen[mb.atom_a]
            b = atom_screen[mb.atom_b]
            pygame.draw.line(
                screen, MOLECULE_PREVIEW_BOND_COLOR, a, b, 2
            )

        # Draw atom dots
        for sx, sy in atom_screen:
            pygame.draw.circle(screen, MOLECULE_PREVIEW_COLOR, (sx, sy), 5)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 5, 1)

    # =========================================================================
    # Coordinate helpers (mirror SinkTool / SourceTool)
    # =========================================================================

    def _get_world_pos(self, mx, my, layout):
        pan_x, pan_y = self.ctx.pan
        return utils.screen_to_sim(
            mx, my,
            self.ctx.zoom, pan_x, pan_y,
            self.ctx.world_size, layout,
        )

    def _world_to_screen(self, wx, wy, layout):
        pan_x, pan_y = self.ctx.pan
        return utils.sim_to_screen(
            wx, wy,
            self.ctx.zoom, pan_x, pan_y,
            self.ctx.world_size, layout,
        )
