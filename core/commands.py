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

Architecture Note (CR-2026-004):
    The Command ABC and CommandQueue remain in core/commands.py per TU-SCENE C1
    to prevent circular dependencies. Concrete commands have been moved to
    model/commands/ and are re-exported here for backward compatibility (C6).
"""

# Command ABC is in command_base.py to prevent circular imports
from core.command_base import Command


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
# Re-exports for backward compatibility (CR-2026-004 C6)
# =============================================================================
# All concrete commands have been moved to model/commands/ but are re-exported
# here so existing imports continue to work.

from model.commands import (
    # Geometry
    AddLineCommand,
    AddCircleCommand,
    RemoveEntityCommand,
    MoveEntityCommand,
    MoveMultipleCommand,
    SetEntityGeometryCommand,
    # Points
    SetPointCommand,
    SetCircleRadiusCommand,
    # Constraints
    AddConstraintCommand,
    RemoveConstraintCommand,
    ToggleAnchorCommand,
    ToggleInfiniteCommand,
    # Properties
    SetPhysicalCommand,
    SetEntityDynamicCommand,
    SetMaterialCommand,
    SetDriverCommand,
    # Composite
    CompositeCommand,
    AddRectangleCommand,
)

__all__ = [
    # Core infrastructure (stays here)
    'Command',
    'CommandQueue',
    # Re-exported commands (from model.commands)
    'AddLineCommand',
    'AddCircleCommand',
    'RemoveEntityCommand',
    'MoveEntityCommand',
    'MoveMultipleCommand',
    'SetEntityGeometryCommand',
    'SetPointCommand',
    'SetCircleRadiusCommand',
    'AddConstraintCommand',
    'RemoveConstraintCommand',
    'ToggleAnchorCommand',
    'ToggleInfiniteCommand',
    'SetPhysicalCommand',
    'SetEntityDynamicCommand',
    'SetMaterialCommand',
    'SetDriverCommand',
    'CompositeCommand',
    'AddRectangleCommand',
]
