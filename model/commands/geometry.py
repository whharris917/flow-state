"""
Geometry Commands - Entity creation, deletion, and movement

These commands handle CRUD operations on geometric entities (Line, Circle, Point).
All commands implement the Command ABC from core.commands per C7 compliance.
"""

import numpy as np
from core.command_base import Command


class AddLineCommand(Command):
    """Add a line to the sketch."""

    changes_topology = True  # Adding entity requires rebuild

    def __init__(self, sketch, start, end, is_ref=False, material_id="Wall",
                 historize=True, supersede=False, physical=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.start = tuple(start)
        self.end = tuple(end)
        self.is_ref = is_ref
        self.material_id = material_id
        self.physical = physical
        self.created_index = None
        self.description = "Add Line"

    def execute(self) -> bool:
        self.created_index = self.sketch.add_line(
            self.start, self.end,
            is_ref=self.is_ref,
            material_id=self.material_id
        )
        # Only atomize non-reference lines (ref lines are never physical)
        if self.created_index is not None and self.physical and not self.is_ref:
            self.sketch.entities[self.created_index].physical = True
        return self.created_index is not None

    def undo(self):
        if self.created_index is not None:
            self.sketch.remove_entity(self.created_index)
            self.created_index = None


class AddCircleCommand(Command):
    """Add a circle to the sketch."""

    changes_topology = True  # Adding entity requires rebuild

    def __init__(self, sketch, center, radius, material_id="Wall",
                 historize=True, supersede=False, physical=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.center = tuple(center)
        self.radius = radius
        self.material_id = material_id
        self.physical = physical
        self.created_index = None
        self.description = "Add Circle"

    def execute(self) -> bool:
        self.created_index = self.sketch.add_circle(
            self.center, self.radius,
            material_id=self.material_id
        )
        if self.created_index is not None and self.physical:
            self.sketch.entities[self.created_index].physical = True
        return self.created_index is not None

    def undo(self):
        if self.created_index is not None:
            self.sketch.remove_entity(self.created_index)
            self.created_index = None


class RemoveEntityCommand(Command):
    """Remove an entity from the sketch."""

    changes_topology = True  # Removing entity requires rebuild

    def __init__(self, sketch, entity_index, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.entity_data = None  # Stored on execute for undo
        self.description = "Delete"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            # Store entity data for undo
            entity = self.sketch.entities[self.entity_index]
            self.entity_data = entity.to_dict()
            self.sketch.remove_entity(self.entity_index)
            return True
        return False

    def undo(self):
        if self.entity_data:
            # Recreate the entity
            from model.geometry import Line, Circle, Point

            e_type = self.entity_data['type']
            if e_type == 'line':
                entity = Line.from_dict(self.entity_data)
            elif e_type == 'circle':
                entity = Circle.from_dict(self.entity_data)
            elif e_type == 'point':
                entity = Point.from_dict(self.entity_data)
            else:
                return

            # Insert at original position
            self.sketch.entities.insert(self.entity_index, entity)


class MoveEntityCommand(Command):
    """Move an entity by a delta."""

    def __init__(self, sketch, entity_index, dx, dy, point_indices=None,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.dx = dx
        self.dy = dy
        self.point_indices = point_indices  # Which points to move (None = all)
        self.description = "Move"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            entity.move(self.dx, self.dy, self.point_indices)
            return True
        return False

    def undo(self):
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            entity.move(-self.dx, -self.dy, self.point_indices)

    def merge(self, other: Command) -> bool:
        """Merge consecutive moves of the same entity."""
        if isinstance(other, MoveEntityCommand):
            if (other.entity_index == self.entity_index and
                other.point_indices == self.point_indices and
                other.historize == self.historize):
                self.dx += other.dx
                self.dy += other.dy
                return True
        return False


class MoveMultipleCommand(Command):
    """Move multiple entities by a delta."""

    def __init__(self, sketch, entity_indices, dx, dy,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_indices = list(entity_indices)
        self.dx = dx
        self.dy = dy
        self.description = "Move Selection"

    def execute(self) -> bool:
        for idx in self.entity_indices:
            if 0 <= idx < len(self.sketch.entities):
                self.sketch.entities[idx].move(self.dx, self.dy)
        return len(self.entity_indices) > 0

    def undo(self):
        for idx in self.entity_indices:
            if 0 <= idx < len(self.sketch.entities):
                self.sketch.entities[idx].move(-self.dx, -self.dy)

    def merge(self, other: Command) -> bool:
        """Merge consecutive moves of the same selection."""
        if isinstance(other, MoveMultipleCommand):
            if (set(other.entity_indices) == set(self.entity_indices) and
                other.historize == self.historize):
                self.dx += other.dx
                self.dy += other.dy
                return True
        return False


class SetEntityGeometryCommand(Command):
    """
    Set all points of an entity to absolute positions.

    Used for operations that change both position AND rotation (e.g., parametric
    drag with User Servo). Unlike MoveEntityCommand which stores delta (dx, dy),
    this stores the complete before/after geometry state for exact undo/redo.

    Args:
        sketch: The Sketch instance
        entity_index: Index of the entity
        old_positions: List of (x, y) tuples for original point positions
        new_positions: List of (x, y) tuples for new point positions
        historize: Whether to add to undo stack
    """

    def __init__(self, sketch, entity_index, old_positions, new_positions,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.old_positions = [tuple(p) for p in old_positions]
        self.new_positions = [tuple(p) for p in new_positions]
        self.description = "Transform"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            for i, pos in enumerate(self.new_positions):
                entity.set_point(i, np.array(pos, dtype=np.float64))
            return True
        return False

    def undo(self):
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            for i, pos in enumerate(self.old_positions):
                entity.set_point(i, np.array(pos, dtype=np.float64))
