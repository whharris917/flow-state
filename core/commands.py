"""
Command Pattern for Undo/Redo

Commands encapsulate actions that can be executed and undone.
This replaces the expensive full-state snapshot approach.

Usage:
    queue = CommandQueue(scene)
    queue.execute(AddLineCommand(sketch, start, end))
    queue.undo()  # Removes the line
    queue.redo()  # Adds it back
"""

from abc import ABC, abstractmethod
import copy
import numpy as np


class Command(ABC):
    """
    Base class for all undoable commands.
    
    Each command must implement:
    - execute(): Perform the action, return True if successful
    - undo(): Reverse the action
    
    Optionally:
    - redo(): Re-perform (defaults to calling execute())
    - merge(other): Combine with another command (for drag operations)
    """
    
    def __init__(self):
        self.description = "Command"
    
    @abstractmethod
    def execute(self) -> bool:
        """Execute the command. Return True if successful."""
        pass
    
    @abstractmethod
    def undo(self):
        """Undo the command."""
        pass
    
    def redo(self):
        """Redo the command. Default implementation calls execute()."""
        self.execute()
    
    def merge(self, other: 'Command') -> bool:
        """
        Try to merge another command into this one.
        Used for combining sequential drag operations.
        Return True if merged, False otherwise.
        """
        return False


class CommandQueue:
    """
    Manages command history for undo/redo.
    
    Features:
    - Configurable history limit
    - Command merging for drag operations
    - Clear separation from domain logic
    """
    
    def __init__(self, max_history=50):
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = max_history
    
    def execute(self, command: Command) -> bool:
        """
        Execute a command and add to history.
        
        Returns True if command executed successfully.
        """
        if command.execute():
            # Try to merge with previous command
            if self.undo_stack and self.undo_stack[-1].merge(command):
                pass  # Merged into existing command
            else:
                self.undo_stack.append(command)
                if len(self.undo_stack) > self.max_history:
                    self.undo_stack.pop(0)
            
            # Clear redo stack on new action
            self.redo_stack.clear()
            return True
        return False
    
    def undo(self) -> bool:
        """Undo the last command. Returns True if successful."""
        if not self.undo_stack:
            return False
        
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return True
    
    def redo(self) -> bool:
        """Redo the last undone command. Returns True if successful."""
        if not self.redo_stack:
            return False
        
        command = self.redo_stack.pop()
        command.redo()
        self.undo_stack.append(command)
        return True
    
    def clear(self):
        """Clear all history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
    
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0


# =============================================================================
# Geometry Commands
# =============================================================================

class AddLineCommand(Command):
    """Add a line to the sketch."""
    
    def __init__(self, sketch, start, end, is_ref=False, material_id="Default"):
        super().__init__()
        self.sketch = sketch
        self.start = tuple(start)
        self.end = tuple(end)
        self.is_ref = is_ref
        self.material_id = material_id
        self.created_index = None
        self.description = "Add Line"
    
    def execute(self) -> bool:
        self.created_index = self.sketch.add_line(
            self.start, self.end, 
            is_ref=self.is_ref, 
            material_id=self.material_id
        )
        return self.created_index is not None
    
    def undo(self):
        if self.created_index is not None:
            self.sketch.remove_entity(self.created_index)
            self.created_index = None


class AddCircleCommand(Command):
    """Add a circle to the sketch."""
    
    def __init__(self, sketch, center, radius, material_id="Default"):
        super().__init__()
        self.sketch = sketch
        self.center = tuple(center)
        self.radius = radius
        self.material_id = material_id
        self.created_index = None
        self.description = "Add Circle"
    
    def execute(self) -> bool:
        self.created_index = self.sketch.add_circle(
            self.center, self.radius,
            material_id=self.material_id
        )
        return self.created_index is not None
    
    def undo(self):
        if self.created_index is not None:
            self.sketch.remove_entity(self.created_index)
            self.created_index = None


class RemoveEntityCommand(Command):
    """Remove an entity from the sketch."""
    
    def __init__(self, sketch, entity_index):
        super().__init__()
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
    
    def __init__(self, sketch, entity_index, dx, dy, point_indices=None):
        super().__init__()
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
                other.point_indices == self.point_indices):
                self.dx += other.dx
                self.dy += other.dy
                return True
        return False


class MoveMultipleCommand(Command):
    """Move multiple entities by a delta."""
    
    def __init__(self, sketch, entity_indices, dx, dy):
        super().__init__()
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
            if set(other.entity_indices) == set(self.entity_indices):
                self.dx += other.dx
                self.dy += other.dy
                return True
        return False


# =============================================================================
# Constraint Commands
# =============================================================================

class AddConstraintCommand(Command):
    """Add a constraint to the sketch."""
    
    def __init__(self, sketch, constraint):
        super().__init__()
        self.sketch = sketch
        self.constraint = constraint
        self.description = f"Add {constraint.type}"
    
    def execute(self) -> bool:
        self.sketch.add_constraint_object(self.constraint)
        return True
    
    def undo(self):
        # Find and remove the constraint
        if self.constraint in self.sketch.constraints:
            self.sketch.constraints.remove(self.constraint)


class RemoveConstraintCommand(Command):
    """Remove a constraint from the sketch."""
    
    def __init__(self, sketch, constraint_index):
        super().__init__()
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
    
    def __init__(self, sketch, entity_index, point_index):
        super().__init__()
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


# =============================================================================
# Composite Commands
# =============================================================================

class CompositeCommand(Command):
    """
    A command that groups multiple commands together.
    All sub-commands are executed/undone as a unit.
    """
    
    def __init__(self, commands, description="Multiple Actions"):
        super().__init__()
        self.commands = list(commands)
        self.description = description
    
    def execute(self) -> bool:
        for cmd in self.commands:
            cmd.execute()
        return len(self.commands) > 0
    
    def undo(self):
        # Undo in reverse order
        for cmd in reversed(self.commands):
            cmd.undo()


class AddRectangleCommand(CompositeCommand):
    """Add a rectangle (4 lines + constraints)."""
    
    def __init__(self, sketch, x1, y1, x2, y2, material_id="Default"):
        from model.constraints import Coincident, Angle
        
        # Create 4 lines
        line_cmds = [
            AddLineCommand(sketch, (x1, y1), (x2, y1), material_id=material_id),
            AddLineCommand(sketch, (x2, y1), (x2, y2), material_id=material_id),
            AddLineCommand(sketch, (x2, y2), (x1, y2), material_id=material_id),
            AddLineCommand(sketch, (x1, y2), (x1, y1), material_id=material_id),
        ]
        
        super().__init__(line_cmds, "Add Rectangle")
        self.sketch = sketch
        
        # We'll add constraints after lines are created
        self._constraint_cmds = []
    
    def execute(self) -> bool:
        # First create the lines
        result = super().execute()
        
        if result:
            # Now add corner constraints
            base = self.commands[0].created_index
            if base is not None:
                from model.constraints import Coincident, Angle
                
                constraints = [
                    Coincident(base, 1, base+1, 0),
                    Coincident(base+1, 1, base+2, 0),
                    Coincident(base+2, 1, base+3, 0),
                    Coincident(base+3, 1, base, 0),
                    Angle('HORIZONTAL', base),
                    Angle('VERTICAL', base+1),
                    Angle('HORIZONTAL', base+2),
                    Angle('VERTICAL', base+3),
                ]
                
                for c in constraints:
                    cmd = AddConstraintCommand(self.sketch, c)
                    cmd.execute()
                    self._constraint_cmds.append(cmd)
        
        return result
    
    def undo(self):
        # Undo constraints first
        for cmd in reversed(self._constraint_cmds):
            cmd.undo()
        self._constraint_cmds.clear()
        
        # Then undo lines
        super().undo()