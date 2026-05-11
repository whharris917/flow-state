"""
MoleculePaletteWidget — right-panel surface for the molecule palette.

Two controls in a stacked layout:
- A dropdown of molecule names from sketch.molecules. Selecting a name
  sets the MoleculeTool's active template and activates the tool so the
  user can click in the canvas to place molecules of that kind.
- A "Builder..." button that opens MoleculeBuilderDialog so the user can
  author or edit a molecule template.

The dropdown is dynamic: when the palette changes (a new molecule is
saved from the builder), the controller calls ``refresh()`` to rebuild
the option list.
"""

import pygame
import shared.config as config

from ui.ui_widgets import UIContainer, Dropdown, Button


class MoleculePaletteWidget(UIContainer):
    """Dropdown + Builder button for the molecule palette."""

    def __init__(self, x, y, w, controller):
        s = config.scale
        self._row_h = s(28)
        self._btn_h = s(35)
        self._spacing = s(4)
        total_h = self._row_h + self._spacing + self._btn_h + s(4)

        super().__init__(x, y, w, total_h, layout_type='vertical',
                         padding=0, spacing=self._spacing)
        self.controller = controller
        self.sketch = controller.sketch
        self._width = w

        # Dropdown — names come from sketch.molecules at construct time;
        # refresh() rebuilds when the palette changes.
        names = self._molecule_names()
        self.dropdown = Dropdown(0, 0, w, self._row_h, names, selected_index=0)
        self.dropdown.on_change = self._on_select_molecule
        self.add_child(self.dropdown)

        self.btn_builder = Button(0, 0, w, self._btn_h, "Molecule Builder...",
                                  toggle=False)
        self.add_child(self.btn_builder)

    # =========================================================================
    # State refresh
    # =========================================================================

    def _molecule_names(self):
        """Return the list of dropdown options, or a placeholder when empty."""
        names = list(self.sketch.molecules.keys())
        return names if names else ["(no molecules)"]

    def refresh(self):
        """Rebuild the dropdown options from the current sketch palette.

        Called by the controller after the Molecule Builder dialog saves
        a new template so the dropdown reflects the updated palette.
        """
        names = self._molecule_names()
        # Preserve current selection if the same name is still present;
        # otherwise fall back to index 0.
        prev = None
        try:
            prev = self.dropdown.get_selected()
        except (IndexError, AttributeError):
            prev = None
        self.dropdown.options = names
        if prev is not None and prev in names:
            self.dropdown.selected_index = names.index(prev)
        else:
            self.dropdown.selected_index = 0

    # =========================================================================
    # Event handling
    # =========================================================================

    def handle_event(self, event):
        # Dropdown first so its expanded overlay clicks land before the button
        if self.dropdown.handle_event(event):
            return True
        if self.btn_builder.handle_event(event):
            self.controller.actions.open_molecule_builder_dialog()
            return True
        return False

    def update(self, dt):
        self.dropdown.update(dt) if hasattr(self.dropdown, 'update') else None
        self.btn_builder.update(dt)

    # =========================================================================
    # Selection callback
    # =========================================================================

    def _on_select_molecule(self, index_or_name, name=None):
        """Dropdown.on_change callback. The Dropdown invokes this with
        (index, name) at line 834 of ui_widgets.py; tests invoke it with
        (name) directly. Handle both signatures by sniffing the call shape.
        """
        if name is None:
            # Single-arg call shape (tests): the arg is the name directly
            actual_name = index_or_name
        else:
            actual_name = name
        if actual_name is None or actual_name == "(no molecules)":
            return
        tpl = self.sketch.get_molecule(actual_name)
        if tpl is None:
            return
        # Look up the MoleculeTool in the session's tool registry
        session = self.controller.session
        mol_tool = session.tools.get(config.TOOL_MOLECULE)
        if mol_tool is None:
            return
        mol_tool.set_template(tpl)
        self.controller.change_tool(config.TOOL_MOLECULE)
