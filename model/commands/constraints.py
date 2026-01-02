"""
Constraint Commands - Adding, removing, and toggling constraints

These commands handle constraint management including anchors and infinite lines.
All commands implement the Command ABC from core.commands per C7 compliance.
"""

from core.command_base import Command


class AddConstraintCommand(Command):
    """
    Add a constraint to the sketch.

    Captures geometry state before solver runs so undo restores both the
    constraint AND the original entity positions.
    """

    def __init__(self, sketch, constraint, historize=True, supersede=False, solve=True):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.constraint = constraint
        self.solve = solve  # Whether to run solver after adding
        self.description = f"Add {constraint.type}"
        self.previous_geometry = {}  # {entity_idx: geometry_data}

    def _get_entity_indices(self):
        """Extract all unique entity indices from the constraint."""
        indices = set()
        for item in self.constraint.indices:
            if isinstance(item, tuple):
                # Point reference: (entity_idx, point_idx)
                indices.add(item[0])
            else:
                # Direct entity index
                indices.add(item)
        return indices

    def _capture_geometry(self, entity_idx):
        """Capture the full geometry state of an entity."""
        if entity_idx >= len(self.sketch.entities):
            return None
        entity = self.sketch.entities[entity_idx]
        # Use entity's to_dict for full state capture
        return entity.to_dict()

    def _restore_geometry(self, entity_idx, geometry_data):
        """Restore an entity's geometry from captured state."""
        if entity_idx >= len(self.sketch.entities) or not geometry_data:
            return
        entity = self.sketch.entities[entity_idx]
        e_type = geometry_data.get('type')

        if e_type == 'line':
            entity.start[:] = geometry_data['start']
            entity.end[:] = geometry_data['end']
        elif e_type == 'circle':
            entity.center[:] = geometry_data['center']
            entity.radius = geometry_data['radius']
        elif e_type == 'point':
            entity.pos[:] = [geometry_data['x'], geometry_data['y']]

    def execute(self) -> bool:
        # Capture geometry of all involved entities BEFORE adding constraint
        self.previous_geometry.clear()
        for entity_idx in self._get_entity_indices():
            self.previous_geometry[entity_idx] = self._capture_geometry(entity_idx)

        # Add constraint and run solver (existing behavior)
        self.sketch.add_constraint_object(self.constraint, solve=self.solve)
        return True

    def undo(self):
        # Remove the constraint
        if self.constraint in self.sketch.constraints:
            self.sketch.constraints.remove(self.constraint)

        # Restore original geometry positions
        for entity_idx, geometry_data in self.previous_geometry.items():
            self._restore_geometry(entity_idx, geometry_data)


class RemoveConstraintCommand(Command):
    """Remove a constraint from the sketch."""

    def __init__(self, sketch, constraint_index, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.constraint_index = constraint_index
        self.constraint_data = None
        self.description = "Delete Constraint"

    def execute(self) -> bool:
        if 0 <= self.constraint_index < len(self.sketch.constraints):
            constraint = self.sketch.constraints[self.constraint_index]
            self.constraint_data = constraint.to_dict()
            self.sketch.constraints.pop(self.constraint_index)
            return True
        return False

    def undo(self):
        if self.constraint_data:
            from model.constraints import create_constraint
            constraint = create_constraint(self.constraint_data)
            if constraint:
                self.sketch.constraints.insert(self.constraint_index, constraint)


class ToggleAnchorCommand(Command):
    """Toggle anchor state of a point."""

    def __init__(self, sketch, entity_index, point_index,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.point_index = point_index
        self.description = "Toggle Anchor"

    def execute(self) -> bool:
        self.sketch.toggle_anchor(self.entity_index, self.point_index)
        return True

    def undo(self):
        # Toggle again to reverse
        self.sketch.toggle_anchor(self.entity_index, self.point_index)


class ToggleInfiniteCommand(Command):
    """Toggle infinite state of a reference line."""

    def __init__(self, sketch, entity_index, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.description = "Toggle Infinite"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            if hasattr(entity, 'infinite'):
                entity.infinite = not entity.infinite
                return True
        return False

    def undo(self):
        # Toggle again to reverse
        self.execute()
