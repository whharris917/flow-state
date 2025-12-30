"""
Command Pattern for Undo/Redo

Commands encapsulate actions that can be executed and undone.
This replaces the expensive full-state snapshot approach.

Key Concepts:

1. historize flag:
   - historize=True (default): Command is added to undo stack after execution
   - historize=False: Command executes but is NOT added to history
   Used for intermediate states during drag operations (SelectTool).

2. supersede flag:
   - supersede=True: Undo the previous command before executing this one
   - Effectively "replaces" the previous command in the undo stack
   Used for live preview during geometry creation (LineTool, CircleTool, etc.)

Supersede Pattern (for geometry creation):
    MOUSEDOWN  → Execute command (historize=True, supersede=False)
    MOUSEMOVE  → Execute command (historize=True, supersede=True) - replaces previous
    MOUSEUP    → Execute command (historize=True, supersede=True) - final version stays
    CANCEL     → scene.undo() - removes preview, stack is clean

Historize Pattern (for geometry editing):
    MOUSEDOWN  → Capture original state
    MOUSEMOVE  → Execute command (historize=False) - responsive feedback
    MOUSEUP    → Execute command (historize=True) - single undo entry
    CANCEL     → Restore from captured state

Usage:
    queue = CommandQueue()
    
    # Normal undoable command
    queue.execute(AddLineCommand(sketch, start, end))
    
    # Superseding command (replaces previous)
    queue.execute(AddLineCommand(sketch, start, new_end, supersede=True))
    
    # Non-historized command (for SelectTool drags)
    queue.execute(MoveEntityCommand(sketch, idx, dx, dy, historize=False))
"""

from abc import ABC, abstractmethod
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

    Class Attributes:
        changes_topology: If True, this command changes entity topology
                          (add/delete entities, material changes).
                          Scene uses this to determine if full rebuild is needed
                          vs just syncing entity positions.

    Args:
        historize: If True (default), command is added to undo stack.
                   If False, command executes but is not recorded in history.
        supersede: If True, undo the previous command before executing.
                   Used for "live preview" during drag creation operations.
    """

    # Override in subclasses that change topology
    changes_topology = False

    def __init__(self, historize=True, supersede=False):
        self.description = "Command"
        self.historize = historize
        self.supersede = supersede
    
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
    - historize flag: commands with historize=False execute but aren't recorded
    - supersede flag: commands with supersede=True replace the previous command
    """
    
    def __init__(self, max_history=50):
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = max_history
    
    def execute(self, command: Command) -> bool:
        """
        Execute a command and optionally add to history.
        
        If command.supersede is True:
            - First undo the most recent command (if any)
            - Then execute this command
            - The new command replaces the old one in the stack
        
        If command.historize is True:
            - Command is added to undo stack after execution
            - Redo stack is cleared
        
        If command.historize is False:
            - Command executes but is NOT added to history
            - Redo stack is NOT cleared
        
        Returns True if command executed successfully.
        """
        # Supersede: undo the previous command first (replacement pattern)
        if command.supersede and self.undo_stack:
            prev = self.undo_stack.pop()
            prev.undo()
            # Note: don't add to redo_stack - this is replacement, not user undo
        
        if command.execute():
            # Only add to history if historize=True
            if command.historize:
                # Superseding commands don't merge - they already replaced
                if not command.supersede and self.undo_stack:
                    if self.undo_stack[-1].merge(command):
                        return True  # Merged into existing command
                
                self.undo_stack.append(command)
                if len(self.undo_stack) > self.max_history:
                    self.undo_stack.pop(0)
                
                # Clear redo stack on new historized action
                self.redo_stack.clear()
            
            return True
        return False
    
    def undo(self) -> bool:
        """Undo the last historized command. Returns True if successful."""
        if not self.undo_stack:
            return False
        
        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        return True
    
    def discard(self) -> bool:
        """
        Discard the last command. Like undo but CANNOT be redone.
        
        Use this for canceling preview operations where the user
        explicitly abandoned the operation.
        """
        if not self.undo_stack:
            return False
        
        command = self.undo_stack.pop()
        command.undo()
        # Intentionally NOT added to redo_stack - this is a discard, not undo
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

    changes_topology = True  # Adding entity requires rebuild

    def __init__(self, sketch, start, end, is_ref=False, material_id="Default",
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
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

    changes_topology = True  # Adding entity requires rebuild

    def __init__(self, sketch, center, radius, material_id="Default",
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
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


# =============================================================================
# Point Editing Commands
# =============================================================================

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


# =============================================================================
# Circle Commands
# =============================================================================

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


# =============================================================================
# Constraint Commands
# =============================================================================

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


class SetPhysicalCommand(Command):
    """Set the physical (atomize) state of an entity."""

    changes_topology = True  # Physical flag affects atomization

    def __init__(self, sketch, entity_index, is_physical,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.is_physical = is_physical
        self.old_physical = None
        self.description = "Set Physical"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_physical = getattr(entity, 'physical', False)
            self.sketch.update_entity(self.entity_index, physical=self.is_physical)
            return True
        return False

    def undo(self):
        if self.old_physical is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                self.sketch.update_entity(self.entity_index, physical=self.old_physical)


class SetEntityDynamicCommand(Command):
    """
    Set the dynamic (two-way coupling) state of an entity.

    When dynamic=True:
    - Entity responds to physics forces from tethered atoms
    - Atoms are tethered (is_static=3) instead of static (is_static=1)
    - Entity has mass, velocity, and can rotate

    When dynamic=False:
    - Entity is immovable (infinite mass)
    - Atoms are static walls
    """

    changes_topology = True  # Dynamic flag changes atom type (static vs tethered)

    def __init__(self, sketch, entity_index, is_dynamic,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.is_dynamic = is_dynamic
        self.old_dynamic = None
        self.description = "Set Dynamic"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_dynamic = getattr(entity, 'dynamic', False)
            entity.dynamic = self.is_dynamic
            return True
        return False

    def undo(self):
        if self.old_dynamic is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                entity = self.sketch.entities[self.entity_index]
                entity.dynamic = self.old_dynamic


class SetMaterialCommand(Command):
    """Set the material of an entity."""

    changes_topology = True  # Material affects atomization (spacing, properties)

    def __init__(self, sketch, entity_index, material_id,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.entity_index = entity_index
        self.material_id = material_id
        self.old_material_id = None
        self.description = "Set Material"

    def execute(self) -> bool:
        if 0 <= self.entity_index < len(self.sketch.entities):
            entity = self.sketch.entities[self.entity_index]
            self.old_material_id = getattr(entity, 'material_id', 'Default')
            self.sketch.update_entity(self.entity_index, material_id=self.material_id)
            return True
        return False

    def undo(self):
        if self.old_material_id is not None:
            if 0 <= self.entity_index < len(self.sketch.entities):
                self.sketch.update_entity(self.entity_index, material_id=self.old_material_id)


class SetDriverCommand(Command):
    """Set animation driver parameters on a constraint."""

    def __init__(self, sketch, constraint_index, driver_data,
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
        self.sketch = sketch
        self.constraint_index = constraint_index
        self.driver_data = driver_data
        self.old_driver_data = None
        self.description = "Set Driver"

    def execute(self) -> bool:
        if 0 <= self.constraint_index < len(self.sketch.constraints):
            constraint = self.sketch.constraints[self.constraint_index]
            self.old_driver_data = getattr(constraint, 'driver', None)
            self.sketch.set_driver(self.constraint_index, self.driver_data)
            return True
        return False

    def undo(self):
        if 0 <= self.constraint_index < len(self.sketch.constraints):
            self.sketch.set_driver(self.constraint_index, self.old_driver_data)


# =============================================================================
# Composite Commands
# =============================================================================

class CompositeCommand(Command):
    """
    A command that groups multiple commands together.
    All sub-commands are executed/undone as a unit.
    """
    
    def __init__(self, commands, description="Multiple Actions", 
                 historize=True, supersede=False):
        super().__init__(historize, supersede)
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
    """
    Add a rectangle (4 lines + corner constraints).

    Uses supersede pattern for live preview during drag.

    Performance optimization: Constraints are added with solve=False,
    then solver runs once at the end.
    """

    changes_topology = True  # Adding entities requires rebuild

    def __init__(self, sketch, x1, y1, x2, y2, material_id="Default",
                 historize=True, supersede=False):
        # Create 4 line commands (not historized individually - we're the unit)
        line_cmds = [
            AddLineCommand(sketch, (x1, y1), (x2, y1), material_id=material_id, historize=False),
            AddLineCommand(sketch, (x2, y1), (x2, y2), material_id=material_id, historize=False),
            AddLineCommand(sketch, (x2, y2), (x1, y2), material_id=material_id, historize=False),
            AddLineCommand(sketch, (x1, y2), (x1, y1), material_id=material_id, historize=False),
        ]
        
        super().__init__(line_cmds, "Add Rectangle", historize, supersede)
        self.sketch = sketch
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        
        # Constraint commands added after lines are created
        self._constraint_cmds = []
    
    @property
    def created_indices(self):
        """Return indices of created lines."""
        return [cmd.created_index for cmd in self.commands if cmd.created_index is not None]
    
    def execute(self) -> bool:
        # First create the lines
        result = super().execute()
        
        if result:
            # Now add corner constraints
            base = self.commands[0].created_index
            if base is not None:
                from model.constraints import Coincident, Angle
                
                constraints = [
                    # Corner coincidents
                    Coincident(base, 1, base+1, 0),
                    Coincident(base+1, 1, base+2, 0),
                    Coincident(base+2, 1, base+3, 0),
                    Coincident(base+3, 1, base, 0),
                    # Angle constraints (horizontal/vertical)
                    Angle('HORIZONTAL', base),
                    Angle('VERTICAL', base+1),
                    Angle('HORIZONTAL', base+2),
                    Angle('VERTICAL', base+3),
                ]
                
                # Add constraints WITHOUT solving (solve=False)
                for c in constraints:
                    cmd = AddConstraintCommand(self.sketch, c, historize=False, solve=False)
                    cmd.execute()
                    self._constraint_cmds.append(cmd)
                
                # Solve ONCE after all constraints are added
                self.sketch.solve(iterations=500)
        
        return result
    
    def undo(self):
        # Undo constraints first
        for cmd in reversed(self._constraint_cmds):
            cmd.undo()
        self._constraint_cmds.clear()
        
        # Then undo lines
        super().undo()