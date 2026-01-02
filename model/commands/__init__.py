"""
Model Commands Package

This package contains all commands that operate on the model domain.
Commands are split by category for maintainability while maintaining
a flat import structure.

Per CR-2026-004 C7: All commands import Command ABC from core.commands.
Per CR-2026-004 C6: All commands are re-exported from core.commands for backward compatibility.

Usage:
    from model.commands import AddLineCommand, AddCircleCommand
    # or
    from core.commands import AddLineCommand  # backward compatible
"""

# Geometry commands
from model.commands.geometry import (
    AddLineCommand,
    AddCircleCommand,
    RemoveEntityCommand,
    MoveEntityCommand,
    MoveMultipleCommand,
    SetEntityGeometryCommand,
)

# Point commands
from model.commands.points import (
    SetPointCommand,
    SetCircleRadiusCommand,
)

# Constraint commands
from model.commands.constraints import (
    AddConstraintCommand,
    RemoveConstraintCommand,
    ToggleAnchorCommand,
    ToggleInfiniteCommand,
)

# Property commands
from model.commands.properties import (
    SetPhysicalCommand,
    SetEntityDynamicCommand,
    SetMaterialCommand,
    SetDriverCommand,
)

# Composite commands
from model.commands.composite import (
    CompositeCommand,
    AddRectangleCommand,
)

__all__ = [
    # Geometry
    'AddLineCommand',
    'AddCircleCommand',
    'RemoveEntityCommand',
    'MoveEntityCommand',
    'MoveMultipleCommand',
    'SetEntityGeometryCommand',
    # Points
    'SetPointCommand',
    'SetCircleRadiusCommand',
    # Constraints
    'AddConstraintCommand',
    'RemoveConstraintCommand',
    'ToggleAnchorCommand',
    'ToggleInfiniteCommand',
    # Properties
    'SetPhysicalCommand',
    'SetEntityDynamicCommand',
    'SetMaterialCommand',
    'SetDriverCommand',
    # Composite
    'CompositeCommand',
    'AddRectangleCommand',
]
