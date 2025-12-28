"""
ConstraintBuilder - Constraint Construction State Machine

Owns:
- Pending constraint type
- Accumulated target walls and points
- Current snap target
- Logic for determining when enough targets are collected

API:
- pending_type: Current constraint type being built (or None)
- target_walls: List of entity indices accumulated
- target_points: List of (entity_idx, point_idx) tuples accumulated
- snap_target: Current snap target for visual feedback
- start(), add_wall(), add_point(), reset(): State machine operations
"""

from typing import Optional, List, Tuple
from core.definitions import CONSTRAINT_DEFS


class ConstraintBuilder:
    """
    Manages the state machine for interactive constraint construction.
    
    Workflow:
    1. User triggers constraint type (e.g., clicks "Parallel" button)
    2. Builder enters pending state, collecting targets
    3. User clicks on walls/points to add targets
    4. When enough targets collected, constraint is created
    5. Builder resets to idle
    
    The builder does NOT create constraints directly - it signals
    when ready and provides the accumulated targets to the caller.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to idle state."""
        self._pending_type: Optional[str] = None
        self._target_walls: List[int] = []
        self._target_points: List[Tuple[int, int]] = []
        self._snap_target: Optional[Tuple[int, int]] = None
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_pending(self) -> bool:
        """Check if we're in the middle of constructing a constraint."""
        return self._pending_type is not None
    
    @property
    def pending_type(self) -> Optional[str]:
        """Get the type of constraint being constructed."""
        return self._pending_type
    
    @pending_type.setter
    def pending_type(self, value: Optional[str]):
        """Set the pending constraint type."""
        self._pending_type = value
    
    @property
    def target_walls(self) -> List[int]:
        """Get list of accumulated wall/entity targets."""
        return self._target_walls
    
    @target_walls.setter
    def target_walls(self, value: List[int]):
        """Set the target walls list."""
        self._target_walls = list(value) if value else []
    
    @property
    def target_points(self) -> List[Tuple[int, int]]:
        """Get list of accumulated point targets."""
        return self._target_points
    
    @target_points.setter
    def target_points(self, value: List[Tuple[int, int]]):
        """Set the target points list."""
        self._target_points = list(value) if value else []
    
    @property
    def snap_target(self) -> Optional[Tuple[int, int]]:
        """Get current snap target (wall_idx, point_idx) or None."""
        return self._snap_target
    
    @snap_target.setter
    def snap_target(self, value: Optional[Tuple[int, int]]):
        """Set current snap target."""
        self._snap_target = value
    
    # =========================================================================
    # State Machine Operations
    # =========================================================================
    
    def start(self, constraint_type: str, initial_walls: List[int] = None, 
              initial_points: List[Tuple[int, int]] = None):
        """
        Start constructing a new constraint.
        
        Args:
            constraint_type: Type of constraint (e.g., 'PARALLEL', 'COINCIDENT')
            initial_walls: Optional pre-selected walls to start with
            initial_points: Optional pre-selected points to start with
        """
        self._pending_type = constraint_type
        self._target_walls = list(initial_walls) if initial_walls else []
        self._target_points = list(initial_points) if initial_points else []
        self._snap_target = None
    
    def add_wall(self, wall_idx: int) -> bool:
        """
        Add a wall/entity target.
        
        Args:
            wall_idx: Index of the wall to add
            
        Returns:
            True if added (not duplicate), False if already present
        """
        if wall_idx not in self._target_walls:
            self._target_walls.append(wall_idx)
            return True
        return False
    
    def add_point(self, entity_idx: int, point_idx: int) -> bool:
        """
        Add a point target.
        
        Args:
            entity_idx: Index of the entity containing the point
            point_idx: Index of the point on the entity
            
        Returns:
            True if added (not duplicate), False if already present
        """
        pt = (entity_idx, point_idx)
        if pt not in self._target_points:
            self._target_points.append(pt)
            return True
        return False
    
    def get_requirement(self) -> Optional[dict]:
        """
        Get the requirement for the current pending constraint type.
        
        Returns:
            Dict with 'w' (wall count), 'p' (point count), 'msg' (help message)
            or None if no pending constraint
        """
        if not self._pending_type:
            return None
        
        if self._pending_type in CONSTRAINT_DEFS:
            # Return first matching rule (simplest case)
            return CONSTRAINT_DEFS[self._pending_type][0]
        return None
    
    def check_ready(self) -> bool:
        """
        Check if we have enough targets to create the constraint.
        
        Returns:
            True if constraint can be created with current targets
        """
        if not self._pending_type:
            return False
        
        if self._pending_type not in CONSTRAINT_DEFS:
            return False
        
        rules = CONSTRAINT_DEFS[self._pending_type]
        for rule in rules:
            if len(self._target_walls) >= rule['w'] and len(self._target_points) >= rule['p']:
                return True
        return False
    
    def get_status_message(self) -> str:
        """
        Get a status message describing the current state.
        
        Returns:
            Human-readable status message
        """
        if not self._pending_type:
            return ""
        
        req = self.get_requirement()
        if req:
            nw = len(self._target_walls)
            np = len(self._target_points)
            return f"{self._pending_type} ({nw}W, {np}P): {req['msg']}"
        return f"{self._pending_type}: Select targets..."
    
    def is_wall_targeted(self, wall_idx: int) -> bool:
        """Check if a wall is already a target."""
        return wall_idx in self._target_walls
    
    def is_point_targeted(self, entity_idx: int, point_idx: int) -> bool:
        """Check if a point is already a target."""
        return (entity_idx, point_idx) in self._target_points

    # =========================================================================
    # Command Factory Methods
    # =========================================================================

    def try_build_command(self, sketch):
        """
        Attempt to build an AddConstraintCommand from current state.

        Tries exact match first, then auto-trims if over-selected.
        Does NOT execute the command or reset state - caller handles that.

        Args:
            sketch: The Sketch instance to create constraint for

        Returns:
            AddConstraintCommand if successful, None if not ready/invalid
        """
        from core.commands import AddConstraintCommand

        if not self._pending_type:
            return None

        # Try exact match
        constraint = sketch.try_create_constraint(
            self._pending_type,
            self._target_walls,
            self._target_points
        )
        if constraint:
            return AddConstraintCommand(sketch, constraint)

        # Try auto-trimming (user selected more than needed)
        rules = CONSTRAINT_DEFS.get(self._pending_type, [])
        for rule in rules:
            if len(self._target_walls) >= rule['w'] and len(self._target_points) >= rule['p']:
                constraint = sketch.try_create_constraint(
                    self._pending_type,
                    self._target_walls[:rule['w']],
                    self._target_points[:rule['p']]
                )
                if constraint:
                    return AddConstraintCommand(sketch, constraint)

        return None

    def build_multi_command(self, sketch, entity_indices):
        """
        Build a CompositeCommand for multi-apply constraints (H/V).

        Used for constraints that apply individually to each selected entity.

        Args:
            sketch: The Sketch instance
            entity_indices: List of entity indices to apply constraint to

        Returns:
            CompositeCommand wrapping individual AddConstraintCommands, or None
        """
        from core.commands import AddConstraintCommand, CompositeCommand

        if not self._pending_type:
            return None

        commands = []
        for idx in entity_indices:
            constraint = sketch.try_create_constraint(self._pending_type, [idx], [])
            if constraint:
                commands.append(AddConstraintCommand(sketch, constraint, historize=False))

        if commands:
            return CompositeCommand(
                commands,
                f"Apply {self._pending_type} to {len(commands)} items"
            )
        return None

    def is_multi_apply(self) -> bool:
        """Check if current pending type is a multi-apply constraint (H/V)."""
        if not self._pending_type:
            return False
        rules = CONSTRAINT_DEFS.get(self._pending_type, [])
        return bool(rules and rules[0].get('multi'))
