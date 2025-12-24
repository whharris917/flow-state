"""
SelectionManager - Entity and Point Selection State

Owns:
- Set of selected entity indices
- Set of selected point references (entity_idx, point_idx)
- Selection manipulation methods

API:
- Use .walls for direct mutable access to entity selection set
- Use .select_entity(), .deselect_entity(), .clear() for method-based access
- Use .entities for a read-only copy (safe for iteration)
"""

from typing import Set, Tuple, Optional, Iterable


class SelectionManager:
    """
    Manages the current selection of entities and points.
    
    Entities are selected by their index in the Sketch.entities list.
    Points are selected by (entity_index, point_index) tuples.
    """
    
    def __init__(self):
        self._entities: Set[int] = set()
        self._points: Set[Tuple[int, int]] = set()
    
    # =========================================================================
    # Entity Selection
    # =========================================================================
    
    @property
    def entities(self) -> Set[int]:
        """Get a read-only copy of selected entity indices (safe for iteration)."""
        return self._entities.copy()
    
    @property
    def walls(self) -> Set[int]:
        """Get direct mutable access to the entity selection set."""
        return self._entities
    
    @walls.setter
    def walls(self, value: Set[int]):
        """Replace the entity selection set."""
        self._entities = set(value) if value else set()
    
    def select_entity(self, index: int):
        """Add an entity to the selection."""
        self._entities.add(index)
    
    def deselect_entity(self, index: int):
        """Remove an entity from the selection."""
        self._entities.discard(index)
    
    def toggle_entity(self, index: int):
        """Toggle an entity's selection state."""
        if index in self._entities:
            self._entities.discard(index)
        else:
            self._entities.add(index)
    
    def is_entity_selected(self, index: int) -> bool:
        """Check if an entity is selected."""
        return index in self._entities
    
    def set_entity_selection(self, indices: Iterable[int]):
        """Replace the entity selection with a new set."""
        self._entities = set(indices)
    
    # =========================================================================
    # Point Selection
    # =========================================================================
    
    @property
    def points(self) -> Set[Tuple[int, int]]:
        """Get direct mutable access to the point selection set."""
        return self._points
    
    @points.setter
    def points(self, value: Set[Tuple[int, int]]):
        """Replace the point selection set."""
        self._points = set(value) if value else set()
    
    def select_point(self, entity_idx: int, point_idx: int):
        """Add a point to the selection."""
        self._points.add((entity_idx, point_idx))
    
    def deselect_point(self, entity_idx: int, point_idx: int):
        """Remove a point from the selection."""
        self._points.discard((entity_idx, point_idx))
    
    def toggle_point(self, entity_idx: int, point_idx: int):
        """Toggle a point's selection state."""
        key = (entity_idx, point_idx)
        if key in self._points:
            self._points.discard(key)
        else:
            self._points.add(key)
    
    def is_point_selected(self, entity_idx: int, point_idx: int) -> bool:
        """Check if a point is selected."""
        return (entity_idx, point_idx) in self._points
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def clear(self):
        """Clear all selections."""
        self._entities.clear()
        self._points.clear()
    
    def clear_entities(self):
        """Clear only entity selection."""
        self._entities.clear()
    
    def clear_points(self):
        """Clear only point selection."""
        self._points.clear()
    
    @property
    def has_selection(self) -> bool:
        """Check if anything is selected."""
        return len(self._entities) > 0 or len(self._points) > 0
    
    @property
    def has_entities(self) -> bool:
        """Check if any entities are selected."""
        return len(self._entities) > 0
    
    @property
    def has_points(self) -> bool:
        """Check if any points are selected."""
        return len(self._points) > 0
    
    @property
    def entity_count(self) -> int:
        """Get number of selected entities."""
        return len(self._entities)
    
    @property
    def point_count(self) -> int:
        """Get number of selected points."""
        return len(self._points)
    
    # =========================================================================
    # Index Remapping (for entity deletion)
    # =========================================================================
    
    def remap_after_deletion(self, deleted_index: int):
        """
        Update selection indices after an entity is deleted.
        
        Args:
            deleted_index: The index of the entity that was deleted
        """
        # Remove the deleted entity from selection
        self._entities.discard(deleted_index)
        
        # Decrement indices greater than deleted
        self._entities = {
            idx - 1 if idx > deleted_index else idx
            for idx in self._entities
        }
        
        # Update point references
        new_points = set()
        for entity_idx, point_idx in self._points:
            if entity_idx == deleted_index:
                continue  # Remove points on deleted entity
            elif entity_idx > deleted_index:
                new_points.add((entity_idx - 1, point_idx))
            else:
                new_points.add((entity_idx, point_idx))
        self._points = new_points
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize selection state."""
        return {
            'entities': list(self._entities),
            'points': list(self._points),
        }
    
    def from_dict(self, data: dict):
        """Deserialize selection state."""
        self._entities = set(data.get('entities', []))
        self._points = set(tuple(p) for p in data.get('points', []))
