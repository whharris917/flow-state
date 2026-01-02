"""
Point Commands - Point editing and circle radius commands

These commands handle individual point manipulation and circle resizing.
All commands implement the Command ABC from core.commands per C7 compliance.
"""

import numpy as np
from core.command_base import Command


class SetPointCommand(Command):
    """
    Set a specific point on an entity to an absolute position.

    This is the fundamental command for point editing. During drags:
    - Execute with historize=False for intermediate positions
    - Execute with historize=True for final commit, with old_position
      explicitly set to the position at drag start

    Args:
        sketch: The Sketch instance
        entity_index: Index of the entity in sketch.entities
        point_index: Which point on the entity (0, 1, etc.)
        new_position: Target position (x, y)
        old_position: Optional original position for undo. If None, captured on execute.
        historize: Whether to add to undo stack
        supersede: Whether to undo previous command first
    """

    def __init__(self, sketch, entity_index, point_index, new_position,
                 old_position=None, historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.point_index = point_index
        self.new_position = tuple(new_position)
        # old_position can be pre-set (for commit commands) or captured on first execute
        self.old_position = tuple(old_position) if old_position is not None else None
        self.description = "Edit Point"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]

            # Capture old position only if not pre-set
            if self.old_position is None:
                old_pt = entity.get_point(self.point_index)
                self.old_position = (float(old_pt[0]), float(old_pt[1]))

            entity.set_point(self.point_index, np.array(self.new_position))
            return True
        return False

    def undo(self):
        if self.old_position is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                entity = self.sketch.entities[self.entity_index]
                entity.set_point(self.point_index, np.array(self.old_position))

    def merge(self, other: Command) -> bool:
        """Merge consecutive point edits on the same point."""
        if isinstance(other, SetPointCommand):
            if (other.entity_index == self.entity_index and
                other.point_index == self.point_index and
                other.historize == self.historize):
                # Keep our old_position, take their new_position
                self.new_position = other.new_position
                return True
        return False


class SetCircleRadiusCommand(Command):
    """
    Set the radius of a circle entity.

    During resize drags:
    - Execute with historize=False for intermediate sizes
    - Execute with historize=True for final commit, with old_radius
      explicitly set to the radius at drag start

    Args:
        sketch: The Sketch instance
        entity_index: Index of the circle entity
        new_radius: Target radius
        old_radius: Optional original radius for undo. If None, captured on execute.
        historize: Whether to add to undo stack
        supersede: Whether to undo previous command first
    """

    def __init__(self, sketch, entity_index, new_radius, old_radius=None,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.new_radius = float(new_radius)
        # old_radius can be pre-set (for commit commands) or captured on first execute
        self.old_radius = float(old_radius) if old_radius is not None else None
        self.description = "Resize Circle"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            if hasattr(entity, 'radius'):
                # Capture old radius only if not pre-set
                if self.old_radius is None:
                    self.old_radius = float(entity.radius)
                entity.radius = self.new_radius
                return True
        return False

    def undo(self):
        if self.old_radius is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                entity = self.sketch.entities[self.entity_index]
                if hasattr(entity, 'radius'):
                    entity.radius = self.old_radius

    def merge(self, other: Command) -> bool:
        """Merge consecutive radius changes on the same circle."""
        if isinstance(other, SetCircleRadiusCommand):
            if (other.entity_index == self.entity_index and
                other.historize == self.historize):
                # Keep our old_radius, take their new_radius
                self.new_radius = other.new_radius
                return True
        return False
