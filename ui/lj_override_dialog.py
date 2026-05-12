"""LjOverrideDialog — view and edit per-pair LJ ε cross-overrides.

Modal dialog that lets the user inspect and tune the
Sketch.lj_cross_overrides table. Picking a material pair shows the current
effective ε (override if present, else Berthelot's geometric mean) and the
L-B baseline for comparison. The user can apply a new ε, remove an
existing override, or just browse the seeded R2 defaults.

Mutations land on sketch.lj_cross_overrides directly; the controller
re-builds the eps_ij_matrix and pushes it to the simulation when the
dialog closes (AppController.apply_lj_override_dialog).

Usage pattern (matches SourcePropertiesDialog):
- Controller pushes the dialog onto the modal stack
- Controller polls dialog.done each frame in actions.update()
- On close (Done button or click outside), controller calls
  apply_lj_override_dialog(dialog) which pushes the updated matrix to sim
"""

import math

import pygame

import core.config as config
from ui.ui_widgets import Dropdown, InputField, Button


class LjOverrideDialog:
    """Modal dialog for editing Sketch.lj_cross_overrides."""

    # Layout constants
    DIALOG_W = 480
    DIALOG_H = 460
    ROW_H = 30
    LIST_TOP = 220
    LIST_ROW_H = 22
    MAX_LIST_ROWS = 7

    def __init__(self, x, y, sketch):
        self.rect = pygame.Rect(x, y, self.DIALOG_W, self.DIALOG_H)
        self.sketch = sketch
        self.done = False
        self.apply = False  # True iff anything was modified
        self.visible = True

        material_names = list(sketch.materials.keys())
        if not material_names:
            material_names = ['Water']

        # Pick first two distinct materials for initial pair display.
        a_idx = 0
        b_idx = 1 if len(material_names) > 1 else 0

        # Dropdowns
        col_x = x + 130
        self.dropdown_a = Dropdown(
            col_x, y + 50, 180, 25, material_names, selected_index=a_idx,
        )
        self.dropdown_a.on_change = lambda *_: self._refresh_input_from_pair()

        self.dropdown_b = Dropdown(
            col_x, y + 90, 180, 25, material_names, selected_index=b_idx,
        )
        self.dropdown_b.on_change = lambda *_: self._refresh_input_from_pair()

        # ε input + Apply / Reset buttons
        self.in_eps = InputField(
            col_x, y + 130, 100, 25, "1.000",
        )

        self.btn_apply = Button(
            col_x, y + 165, 100, 28, "Set Override", toggle=False,
            color_inactive=config.COLOR_SUCCESS,
        )
        self.btn_reset = Button(
            col_x + 110, y + 165, 90, 28, "Reset (L-B)", toggle=False,
        )

        self.btn_done = Button(
            x + self.DIALOG_W - 110, y + self.DIALOG_H - 40, 90, 30,
            "Done", toggle=False, color_inactive=config.COLOR_ACCENT,
        )

        # Initial input value reflects the currently-selected pair.
        self._refresh_input_from_pair()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _current_pair(self):
        """Return (name_a, name_b) for the dropdowns' current selection."""
        return self.dropdown_a.get_selected(), self.dropdown_b.get_selected()

    def _lb_value(self, name_a, name_b):
        """Pure L-B for the pair, ignoring any registered override."""
        mat_a = self.sketch.get_material(name_a)
        mat_b = self.sketch.get_material(name_b)
        return math.sqrt(mat_a.epsilon * mat_b.epsilon)

    def _has_override(self, name_a, name_b):
        return frozenset((name_a, name_b)) in self.sketch.lj_cross_overrides

    def _effective_eps(self, name_a, name_b):
        return self.sketch.get_lj_epsilon(name_a, name_b)

    def _refresh_input_from_pair(self):
        """Populate the input field with the current effective ε for the
        selected pair, so the user sees the value before editing."""
        a, b = self._current_pair()
        self.in_eps.set_value(self._effective_eps(a, b))

    # =========================================================================
    # Apply / Reset (mutate sketch.lj_cross_overrides)
    # =========================================================================

    def _do_apply(self):
        """Set the override for the currently-selected pair to the input
        field's value. Same-material pairs are silently ignored (the
        diagonal of the matrix should always be the material's own ε)."""
        a, b = self._current_pair()
        if a == b:
            return
        new_eps = self.in_eps.get_value(self._effective_eps(a, b))
        if new_eps <= 0.0:
            # Reject non-positive ε — would make the well well-defined but
            # the user almost certainly meant to type a small-but-positive
            # value. Snap to a tiny floor.
            new_eps = 1e-3
            self.in_eps.set_value(new_eps)
        self.sketch.set_lj_cross_override(a, b, new_eps)
        self.apply = True  # mark dialog as having mutated state

    def _do_reset(self):
        """Remove any override on the currently-selected pair — pair will
        fall back to Berthelot's geometric-mean rule."""
        a, b = self._current_pair()
        if a == b:
            return
        if self._has_override(a, b):
            self.sketch.remove_lj_cross_override(a, b)
            self.apply = True
        # Refresh the input field to show the now-L-B value
        self._refresh_input_from_pair()

    # =========================================================================
    # Event handling
    # =========================================================================

    def handle_event(self, event):
        if not self.visible:
            return False

        # Dropdowns first so their overlays catch clicks before the body
        if self.dropdown_a.handle_event(event):
            return True
        if self.dropdown_b.handle_event(event):
            return True
        if self.in_eps.handle_event(event):
            return True

        if self.btn_apply.handle_event(event):
            self._do_apply()
            return True
        if self.btn_reset.handle_event(event):
            self._do_reset()
            return True
        if self.btn_done.handle_event(event):
            self.done = True
            return True

        # Absorb stray mouse events landing on the dialog body
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True
            # Also absorb dropdown-overlay clicks
            for dd in (self.dropdown_a, self.dropdown_b):
                if dd.expanded and dd.get_expanded_rect().collidepoint(event.pos):
                    return True
        if event.type == pygame.MOUSEMOTION:
            return True
        return False

    def update(self, dt):
        self.in_eps.update(dt)
        self.btn_apply.update(dt)
        self.btn_reset.update(dt)
        self.btn_done.update(dt)

    # =========================================================================
    # Draw
    # =========================================================================

    def draw(self, screen, font):
        if not self.visible:
            return

        # Drop shadow
        shadow = self.rect.copy()
        shadow.x += 5
        shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)

        # Body
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)

        # Title
        title = font.render("Cross-ε Overrides", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 12))

        # Subtitle
        subtitle = font.render(
            "Per-pair LJ ε. Breaks Berthelot's geometric-mean rule.",
            True, config.COLOR_TEXT_DIM,
        )
        screen.blit(subtitle, (self.rect.x + 15, self.rect.y + 32))

        x = self.rect.x
        y = self.rect.y

        # Labels for the input row
        screen.blit(font.render("Material A:", True, config.COLOR_TEXT),
                    (x + 20, y + 55))
        screen.blit(font.render("Material B:", True, config.COLOR_TEXT),
                    (x + 20, y + 95))
        screen.blit(font.render("ε for pair:", True, config.COLOR_TEXT),
                    (x + 20, y + 135))

        # Display the L-B reference and override-state hint
        a, b = self._current_pair()
        if a == b:
            hint = "(same material — no cross-pair)"
            hint_col = config.COLOR_TEXT_DIM
        else:
            lb = self._lb_value(a, b)
            if self._has_override(a, b):
                hint = f"override active   |   L-B baseline: {lb:.3f}"
                hint_col = (160, 220, 160)  # green-ish — override active
            else:
                hint = f"using L-B baseline: {lb:.3f}"
                hint_col = config.COLOR_TEXT_DIM
        screen.blit(font.render(hint, True, hint_col),
                    (x + 240, y + 135))

        # Section divider
        list_y = y + self.LIST_TOP
        pygame.draw.line(
            screen, config.PANEL_BORDER_COLOR,
            (x + 15, list_y - 8), (x + self.DIALOG_W - 15, list_y - 8),
        )
        screen.blit(font.render("Active overrides:", True, (220, 220, 220)),
                    (x + 20, list_y - 5))

        # List of currently-registered overrides
        overrides = self.sketch.lj_cross_overrides
        if not overrides:
            screen.blit(
                font.render("(none — all pairs use L-B)",
                            True, config.COLOR_TEXT_DIM),
                (x + 30, list_y + 20),
            )
        else:
            # Sort by material names for stable rendering
            rows = []
            for key, eps in overrides.items():
                names = sorted(key)
                if len(names) == 1:
                    names = [names[0], names[0]]
                lb = self._lb_value(names[0], names[1])
                rows.append((names[0], names[1], eps, lb))
            rows.sort()
            for i, (na, nb, eps, lb) in enumerate(rows[: self.MAX_LIST_ROWS]):
                ry = list_y + 20 + i * self.LIST_ROW_H
                # Format: "Polar  +  Nonpolar    ε=0.250   (L-B: 1.000)"
                pair_txt = f"{na}  +  {nb}"
                value_txt = f"ε={eps:.3f}"
                lb_txt = f"(L-B: {lb:.3f})"
                screen.blit(font.render(pair_txt, True, config.COLOR_TEXT),
                            (x + 30, ry))
                screen.blit(font.render(value_txt, True, (200, 220, 255)),
                            (x + 220, ry))
                screen.blit(font.render(lb_txt, True, config.COLOR_TEXT_DIM),
                            (x + 320, ry))
            if len(rows) > self.MAX_LIST_ROWS:
                extra = len(rows) - self.MAX_LIST_ROWS
                screen.blit(
                    font.render(f"... and {extra} more",
                                True, config.COLOR_TEXT_DIM),
                    (x + 30, list_y + 20 + self.MAX_LIST_ROWS * self.LIST_ROW_H),
                )

        # Widgets (drawn last so overlays render above body)
        self.in_eps.draw(screen, font)
        self.btn_apply.draw(screen, font)
        self.btn_reset.draw(screen, font)
        self.btn_done.draw(screen, font)
        self.dropdown_a.draw(screen, font)
        self.dropdown_b.draw(screen, font)
