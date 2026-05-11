"""
MoleculeBuilderDialog — author a MoleculeTemplate inside a modal dialog.

The dialog hosts a mini-canvas where the user places atoms (with a chosen
material) and draws bonds between them. The result is a MoleculeTemplate
saved into the Sketch's molecule palette on Save.

UX model:
- **Atom mode**: click in the canvas → adds a MoleculeAtom at that local
  coordinate, using the currently selected material from the dropdown.
- **Bond mode**: click an atom → highlights it. Click another atom →
  creates a MoleculeBond between them with (k, r_eq) defaulted from
  Sketch.get_bond_default for the pair's materials. The most recently
  created bond becomes the "selected bond" whose k / r_eq are editable
  via two InputFields at the bottom.
- **Clear** button wipes all atoms and bonds (resets the template).

Programmatic API for tests:
- ``add_atom_at_canvas_pos(canvas_x, canvas_y, material_name=None)``
- ``begin_bond(atom_idx)`` / ``complete_bond(atom_idx)``
- ``clear_template()``
- ``apply_to_sketch(sketch)``

Polled-flag pattern matches existing dialogs (SourcePropertiesDialog):
caller pushes onto modal stack, polls ``done`` each frame in
``actions.update()``, applies if ``apply`` is True.
"""

import math
import pygame

import core.config as config
from model.molecule import MoleculeAtom, MoleculeBond, MoleculeTemplate
from ui.ui_widgets import InputField, Button, Dropdown


# Canvas tuning — local coordinate range visible in the canvas rect.
# A view_range of 8 means the canvas shows local coords from -4 to +4
# along the shorter axis. The longer axis gets proportionally more.
CANVAS_VIEW_RANGE = 8.0
ATOM_RADIUS_PX = 8
HIGHLIGHT_RADIUS_PX = 11
PICK_THRESHOLD_PX = 12
BOND_COLOR = (200, 200, 200)
BOND_SELECTED_COLOR = (255, 220, 80)
PENDING_BOND_COLOR = (255, 220, 80)
CANVAS_BG_COLOR = (24, 24, 26)
CANVAS_BORDER_COLOR = (70, 70, 75)


class MoleculeBuilderDialog:
    """Modal dialog for authoring a MoleculeTemplate."""

    def __init__(self, x, y, sketch, template=None):
        # Dialog frame
        self.rect = pygame.Rect(x, y, 520, 540)
        self.sketch = sketch
        self.done = False
        self.apply = False
        self.cancelled = False
        self.visible = True

        # Working template — copied from input or freshly created.
        if template is not None:
            self.template = MoleculeTemplate(
                name=template.name,
                atoms=[MoleculeAtom(a.x, a.y, a.material_name) for a in template.atoms],
                bonds=[MoleculeBond(b.atom_a, b.atom_b, b.k, b.r_eq)
                       for b in template.bonds],
            )
        else:
            self.template = MoleculeTemplate(name="NewMolecule", atoms=[], bonds=[])

        # Mode: 'atom' or 'bond'
        self.mode = 'atom'
        # When in bond mode, the first-clicked atom index waiting for a partner.
        self.pending_bond_first = None
        # Index of the bond whose k/r_eq is currently editable.
        self.selected_bond = None

        # --- Widget layout (positions relative to self.rect) ---
        # Header row: name + material dropdown
        self.in_name = InputField(x + 90, y + 40, 240, 25, self.template.name)

        material_names = list(sketch.materials.keys())
        try:
            sel = material_names.index(material_names[0])
        except (ValueError, IndexError):
            sel = 0
        self.dropdown = Dropdown(
            x + 90, y + 75, 240, 25,
            material_names if material_names else ['Water'],
            selected_index=sel,
        )

        # Mode toggle buttons (radio-style)
        self.btn_atom = Button(x + 20, y + 110, 90, 28, "Atom", toggle=True)
        self.btn_atom.is_active = True
        self.btn_bond = Button(x + 120, y + 110, 90, 28, "Bond", toggle=True)
        self.btn_clear = Button(x + 400, y + 110, 90, 28, "Clear",
                                toggle=False, color_inactive=config.COLOR_DANGER)

        # Canvas rect — inside the dialog
        self.canvas_rect = pygame.Rect(x + 20, y + 150, 480, 280)

        # Selected-bond inputs
        self.in_k = InputField(x + 90, y + 445, 100, 25, "")
        self.in_r_eq = InputField(x + 280, y + 445, 100, 25, "")

        # Footer buttons
        self.btn_cancel = Button(x + 20, y + 490, 100, 30, "Cancel", toggle=False)
        self.btn_save = Button(x + 400, y + 490, 100, 30, "Save", toggle=False,
                               color_inactive=config.COLOR_SUCCESS)

    # =========================================================================
    # Coordinate helpers
    # =========================================================================

    def _scale(self):
        """Pixels per local-coord unit."""
        # Use the shorter dimension so the canvas always shows at least
        # CANVAS_VIEW_RANGE units along that axis.
        return min(self.canvas_rect.width, self.canvas_rect.height) / CANVAS_VIEW_RANGE

    def _canvas_center_px(self):
        return (self.canvas_rect.centerx, self.canvas_rect.centery)

    def local_to_canvas(self, local_x, local_y):
        """Local molecule coords → pixel coords inside the canvas."""
        cx, cy = self._canvas_center_px()
        s = self._scale()
        # Invert y so positive y is up (matches molecule convention)
        return (int(cx + local_x * s), int(cy - local_y * s))

    def canvas_to_local(self, px, py):
        """Pixel coords inside the canvas → local molecule coords."""
        cx, cy = self._canvas_center_px()
        s = self._scale()
        return ((px - cx) / s, -(py - cy) / s)

    def _pick_atom_at_pixel(self, px, py):
        """Return atom index nearest (px, py) within PICK_THRESHOLD_PX, or None."""
        best_idx = None
        best_d2 = (PICK_THRESHOLD_PX + 1) ** 2
        for i, a in enumerate(self.template.atoms):
            ax, ay = self.local_to_canvas(a.x, a.y)
            d2 = (ax - px) ** 2 + (ay - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

    # =========================================================================
    # Programmatic API (used by event handler AND by tests)
    # =========================================================================

    def add_atom_local(self, local_x, local_y, material_name=None):
        """Append an atom at the given local coordinate. Skips if too close
        to an existing atom (using PICK_THRESHOLD_PX in pixel space)."""
        if material_name is None:
            material_name = self.dropdown.get_selected()
        # Check overlap in pixel space
        px, py = self.local_to_canvas(local_x, local_y)
        if self._pick_atom_at_pixel(px, py) is not None:
            return None
        self.template.atoms.append(MoleculeAtom(
            x=float(local_x), y=float(local_y), material_name=material_name,
        ))
        return len(self.template.atoms) - 1

    def begin_bond(self, atom_idx):
        """Start a bond from atom_idx. Subsequent complete_bond completes it."""
        if 0 <= atom_idx < len(self.template.atoms):
            self.pending_bond_first = atom_idx

    def complete_bond(self, atom_idx):
        """Complete a pending bond, ending at atom_idx. Returns the new bond
        index, or None if no bond was created (self-bond or duplicate)."""
        if self.pending_bond_first is None:
            return None
        first = self.pending_bond_first
        self.pending_bond_first = None
        if atom_idx == first:
            return None
        if not (0 <= atom_idx < len(self.template.atoms)):
            return None
        # Reject duplicate bonds (same unordered pair)
        for b in self.template.bonds:
            if {b.atom_a, b.atom_b} == {first, atom_idx}:
                return None
        # Default (k, r_eq) from sketch.bond_defaults for the material pair.
        mat_a = self.template.atoms[first].material_name
        mat_b = self.template.atoms[atom_idx].material_name
        k, r_eq = self.sketch.get_bond_default(mat_a, mat_b)
        bond = MoleculeBond(atom_a=first, atom_b=atom_idx, k=k, r_eq=r_eq)
        self.template.bonds.append(bond)
        b_idx = len(self.template.bonds) - 1
        self.selected_bond = b_idx
        self._refresh_bond_inputs()
        return b_idx

    def cancel_pending_bond(self):
        """Drop any half-started bond. ESC and mode-switch call this."""
        self.pending_bond_first = None

    def clear_template(self):
        """Reset the working template to empty atoms+bonds. Name is kept."""
        self.template.atoms = []
        self.template.bonds = []
        self.pending_bond_first = None
        self.selected_bond = None
        self._refresh_bond_inputs()

    def select_bond(self, bond_idx):
        """Pick a bond for k/r_eq editing."""
        if 0 <= bond_idx < len(self.template.bonds):
            self.selected_bond = bond_idx
            self._refresh_bond_inputs()

    def _refresh_bond_inputs(self):
        if self.selected_bond is None or self.selected_bond >= len(self.template.bonds):
            self.in_k.set_value("")
            self.in_r_eq.set_value("")
            return
        b = self.template.bonds[self.selected_bond]
        self.in_k.set_value(f"{b.k:.2f}")
        self.in_r_eq.set_value(f"{b.r_eq:.3f}")

    def apply_bond_input_edits(self):
        """Push the InputField contents into the selected bond's k/r_eq."""
        if self.selected_bond is None:
            return
        b = self.template.bonds[self.selected_bond]
        try:
            new_k = float(self.in_k.text)
            if new_k > 0:
                b.k = new_k
        except (ValueError, AttributeError):
            pass
        try:
            new_r_eq = float(self.in_r_eq.text)
            if new_r_eq > 0:
                b.r_eq = new_r_eq
        except (ValueError, AttributeError):
            pass

    def apply_to_sketch(self, sketch):
        """Commit the working template to the sketch's molecule palette.

        Uses the current value of the name InputField; if empty the
        template's existing name is kept.
        """
        try:
            new_name = self.in_name.text.strip()
        except AttributeError:
            new_name = ""
        if new_name:
            self.template.name = new_name
        # Pull any pending bond-input edits before saving.
        self.apply_bond_input_edits()
        sketch.add_molecule(self.template)

    # =========================================================================
    # Mode toggle
    # =========================================================================

    def set_mode(self, mode):
        if mode not in ('atom', 'bond'):
            return
        self.mode = mode
        self.btn_atom.is_active = (mode == 'atom')
        self.btn_bond.is_active = (mode == 'bond')
        self.cancel_pending_bond()

    # =========================================================================
    # Event handling (pygame)
    # =========================================================================

    def handle_event(self, event):
        if not self.visible:
            return False

        # Header widgets first
        if self.dropdown.handle_event(event):
            return True
        if self.in_name.handle_event(event):
            return True

        # Bond k/r_eq edits write back on every event so the model stays in sync.
        if self.in_k.handle_event(event):
            self.apply_bond_input_edits()
            return True
        if self.in_r_eq.handle_event(event):
            self.apply_bond_input_edits()
            return True

        # Mode toggle buttons
        if self.btn_atom.handle_event(event):
            self.set_mode('atom')
            return True
        if self.btn_bond.handle_event(event):
            self.set_mode('bond')
            return True

        if self.btn_clear.handle_event(event):
            self.clear_template()
            return True

        # Footer
        if self.btn_cancel.handle_event(event):
            self.cancelled = True
            self.done = True
            return True
        if self.btn_save.handle_event(event):
            self.apply = True
            self.done = True
            return True

        # Canvas clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.canvas_rect.collidepoint(event.pos):
                self._handle_canvas_click(event.pos)
                return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # ESC: if a bond is pending, drop it. Otherwise close dialog.
                if self.pending_bond_first is not None:
                    self.cancel_pending_bond()
                    return True
                self.cancelled = True
                self.done = True
                return True

        # Absorb stray mouse events inside the dialog so the world doesn't catch them
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True
            if self.dropdown.expanded:
                if self.dropdown.get_expanded_rect().collidepoint(event.pos):
                    return True
        if event.type == pygame.MOUSEMOTION:
            return True
        return False

    def _handle_canvas_click(self, mouse_pos):
        px, py = mouse_pos
        if self.mode == 'atom':
            # Click an existing atom → select it (no-op for atom mode, but
            # could be used in the future). Otherwise place a new atom.
            existing = self._pick_atom_at_pixel(px, py)
            if existing is not None:
                return
            local_x, local_y = self.canvas_to_local(px, py)
            self.add_atom_local(local_x, local_y)
        elif self.mode == 'bond':
            picked = self._pick_atom_at_pixel(px, py)
            if picked is None:
                # Click in empty space cancels a pending bond
                self.cancel_pending_bond()
                return
            if self.pending_bond_first is None:
                self.begin_bond(picked)
            else:
                self.complete_bond(picked)

    def update(self, dt):
        self.in_name.update(dt)
        self.in_k.update(dt)
        self.in_r_eq.update(dt)
        self.btn_atom.update(dt)
        self.btn_bond.update(dt)
        self.btn_clear.update(dt)
        self.btn_cancel.update(dt)
        self.btn_save.update(dt)

    # =========================================================================
    # Rendering
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

        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)

        # Title
        screen.blit(font.render("Molecule Builder", True, (255, 255, 255)),
                    (self.rect.x + 15, self.rect.y + 12))

        # Field labels
        x = self.rect.x
        y = self.rect.y
        screen.blit(font.render("Name:", True, config.COLOR_TEXT), (x + 20, y + 45))
        screen.blit(font.render("Material:", True, config.COLOR_TEXT), (x + 20, y + 80))

        # Canvas backdrop
        pygame.draw.rect(screen, CANVAS_BG_COLOR, self.canvas_rect)
        pygame.draw.rect(screen, CANVAS_BORDER_COLOR, self.canvas_rect, 1)

        # Crosshair at canvas origin (local 0,0)
        ox, oy = self.local_to_canvas(0.0, 0.0)
        pygame.draw.line(screen, (50, 50, 55), (ox - 6, oy), (ox + 6, oy), 1)
        pygame.draw.line(screen, (50, 50, 55), (ox, oy - 6), (ox, oy + 6), 1)

        # Bonds first (so atom dots cover the line endpoints)
        for b_idx, b in enumerate(self.template.bonds):
            color = BOND_SELECTED_COLOR if b_idx == self.selected_bond else BOND_COLOR
            a = self.template.atoms[b.atom_a]
            c = self.template.atoms[b.atom_b]
            ax, ay = self.local_to_canvas(a.x, a.y)
            cx, cy = self.local_to_canvas(c.x, c.y)
            pygame.draw.line(screen, color, (ax, ay), (cx, cy), 2)

        # Pending bond preview (from pending_bond_first to mouse)
        if self.pending_bond_first is not None and self.mode == 'bond':
            mx, my = pygame.mouse.get_pos()
            if self.canvas_rect.collidepoint(mx, my):
                a = self.template.atoms[self.pending_bond_first]
                ax, ay = self.local_to_canvas(a.x, a.y)
                pygame.draw.line(screen, PENDING_BOND_COLOR, (ax, ay), (mx, my), 1)

        # Atoms
        for i, a in enumerate(self.template.atoms):
            ax, ay = self.local_to_canvas(a.x, a.y)
            mat = self.sketch.get_material(a.material_name)
            color = getattr(mat, 'color', (50, 150, 255))
            # Highlight ring when this atom is the pending bond's first endpoint
            if i == self.pending_bond_first:
                pygame.draw.circle(screen, PENDING_BOND_COLOR, (ax, ay),
                                   HIGHLIGHT_RADIUS_PX, 2)
            pygame.draw.circle(screen, color, (ax, ay), ATOM_RADIUS_PX)
            pygame.draw.circle(screen, (255, 255, 255), (ax, ay),
                               ATOM_RADIUS_PX, 1)

        # Selected bond editor labels
        screen.blit(font.render("Selected bond:", True, config.COLOR_TEXT),
                    (x + 20, y + 425))
        screen.blit(font.render("k:", True, config.COLOR_TEXT), (x + 60, y + 450))
        screen.blit(font.render("r_eq:", True, config.COLOR_TEXT), (x + 230, y + 450))

        # Status footer (mode + atom/bond count)
        n_atoms = len(self.template.atoms)
        n_bonds = len(self.template.bonds)
        mode_label = "Atom mode" if self.mode == 'atom' else "Bond mode"
        status = f"{mode_label} | atoms: {n_atoms}  bonds: {n_bonds}"
        screen.blit(font.render(status, True, config.COLOR_TEXT_DIM),
                    (x + 20, y + 145 - 22))

        # Widgets
        self.dropdown.draw(screen, font)
        self.in_name.draw(screen, font)
        self.btn_atom.draw(screen, font)
        self.btn_bond.draw(screen, font)
        self.btn_clear.draw(screen, font)
        self.in_k.draw(screen, font)
        self.in_r_eq.draw(screen, font)
        self.btn_cancel.draw(screen, font)
        self.btn_save.draw(screen, font)
