"""
Composite Commands - Multi-step command grouping

These commands group multiple commands together as atomic units.
All commands implement the Command ABC from core.commands per C7 compliance.
"""

from core.command_base import Command
from model.commands.geometry import AddLineCommand
from model.commands.constraints import AddConstraintCommand


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

    def __init__(self, sketch, x1, y1, x2, y2, material_id="Wall",
                 historize=True, supersede=False, physical=False):
        # Create 4 line commands (not historized individually - we're the unit)
        line_cmds = [
            AddLineCommand(sketch, (x1, y1), (x2, y1), material_id=material_id, historize=False, physical=physical),
            AddLineCommand(sketch, (x2, y1), (x2, y2), material_id=material_id, historize=False, physical=physical),
            AddLineCommand(sketch, (x2, y2), (x1, y2), material_id=material_id, historize=False, physical=physical),
            AddLineCommand(sketch, (x1, y2), (x1, y1), material_id=material_id, historize=False, physical=physical),
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
