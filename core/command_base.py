"""
Command Base Class

The Command ABC is placed in this separate module to prevent circular imports
between core/commands.py (which re-exports concrete commands) and model/commands/
(which defines concrete commands that inherit from Command).

Per CR-2026-004 C1: The Command ABC remains in the core/ package.

Usage:
    # In model/commands/*.py:
    from core.command_base import Command

    # In application code (preferred):
    from core.commands import Command  # Re-exported from here
"""

from abc import ABC, abstractmethod


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
