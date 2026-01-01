"""
Entity Protocols - Interfaces for cross-domain communication.

These protocols define what UI and other consumers can expect from
entities WITHOUT needing to import concrete types. This eliminates
isinstance() checks and enables true polymorphism.

Created as part of CR-2026-001 to decouple UI from model concrete types.

Usage:
    # Instead of:
    from model.geometry import Line, Circle
    if isinstance(entity, Line): ...

    # Use:
    from model.protocols import EntityType
    if entity.entity_type == EntityType.LINE: ...
    data = entity.get_render_data()

Protocols defined:
    - EntityType: Enum for entity type tags
    - Renderable: Interface for entities that can be rendered
    - Draggable: Interface for entities that can be manipulated
    - Serializable: Interface for entities that can be saved/loaded
"""

from typing import Protocol, List, Tuple, Dict, Any, Optional, runtime_checkable
from enum import Enum


class EntityType(Enum):
    """
    Entity type tags - used instead of isinstance() checks.

    Consumers switch on this enum rather than importing concrete classes.
    This allows the renderer and tools to handle different entity types
    without creating import dependencies on model.geometry.

    Usage:
        if entity.entity_type == EntityType.LINE:
            # Handle line-specific rendering
        elif entity.entity_type == EntityType.CIRCLE:
            # Handle circle-specific rendering
    """
    LINE = "line"
    CIRCLE = "circle"
    POINT = "point"


@runtime_checkable
class Renderable(Protocol):
    """
    Protocol for entities that can be rendered.

    Implementers: Line, Circle, Point (in model/geometry.py)
    Consumers: Renderer (reads entity_type and get_render_data)

    The renderer uses this protocol to draw entities without knowing
    their concrete types. Each entity provides its rendering data
    through get_render_data(), which returns a dictionary with
    type-specific information.
    """

    @property
    def entity_type(self) -> EntityType:
        """Return the entity type tag for switch-based dispatch."""
        ...

    @property
    def is_reference(self) -> bool:
        """True if this is a reference/construction entity (not physical)."""
        ...

    @property
    def material_id(self) -> str:
        """Material identifier for color lookup."""
        ...

    def get_render_data(self) -> Dict[str, Any]:
        """
        Return type-specific rendering data.

        Returns a dictionary with rendering information. The contents
        depend on entity_type:

        LINE returns:
            {
                "start": (x, y),      # Start point in world coords
                "end": (x, y),        # End point in world coords
                "infinite": bool,     # Whether line extends infinitely
                "anchored": [bool, bool],  # Anchor state per endpoint
                "physical": bool,     # Whether entity is atomized
                "dynamic": bool,      # Whether entity has two-way coupling
            }

        CIRCLE returns:
            {
                "center": (x, y),     # Center in world coords
                "radius": float,      # Radius in world units
                "anchored": [bool],   # Anchor state of center
                "physical": bool,
                "dynamic": bool,
            }

        POINT returns:
            {
                "pos": (x, y),        # Position in world coords
                "is_anchor": bool,    # Whether point is anchored
                "is_handle": bool,    # Whether point is a ProcessObject handle
                "anchored": [bool],
            }
        """
        ...


@runtime_checkable
class Draggable(Protocol):
    """
    Protocol for entities that can be manipulated by tools.

    Implementers: Line, Circle, Point (in model/geometry.py)
    Consumers: SelectTool, ToolContext queries

    This protocol provides a uniform interface for tools to interact
    with entities during drag operations, without knowing concrete types.
    """

    def point_count(self) -> int:
        """
        Number of control points.

        Line: 2 (start, end)
        Circle: 1 (center)
        Point: 1 (position)
        """
        ...

    def get_point(self, index: int) -> Tuple[float, float]:
        """
        Get control point position by index.

        Args:
            index: Point index (0 to point_count()-1)

        Returns:
            (x, y) tuple in world coordinates
        """
        ...

    def get_anchored(self, index: int) -> bool:
        """
        Check if control point at index is anchored.

        Anchored points are fixed during constraint solving.

        Args:
            index: Point index

        Returns:
            True if the point is anchored
        """
        ...

    def distance_to(self, x: float, y: float) -> float:
        """
        Distance from world point (x, y) to this entity.

        Used for hit-testing during entity selection.

        Args:
            x, y: World coordinates

        Returns:
            Distance in world units
        """
        ...

    def parameter_at(self, x: float, y: float) -> Optional[float]:
        """
        Return parametric position (0.0-1.0) of closest point on entity.

        Used for drag handle calculation to determine where on the
        entity the user grabbed.

        Line: t parameter along segment (0 = start, 1 = end)
        Circle: angle / (2*pi)
        Point: always 0.0

        Args:
            x, y: World coordinates

        Returns:
            Parameter value, or None if not applicable
        """
        ...


@runtime_checkable
class Serializable(Protocol):
    """
    Protocol for entities that can be saved/loaded.

    Implementers: Line, Circle, Point, all Constraints
    Consumers: file_io.py, scene.py (save/load methods)

    This protocol ensures consistent serialization across all
    entity types.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON storage.

        The dictionary must include a 'type' key identifying the
        entity type for deserialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary previously created by to_dict()

        Returns:
            Reconstructed entity instance
        """
        ...
