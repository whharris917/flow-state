"""
ToolContext - Narrow Interface for Tools

This facade provides tools with a controlled interface to the application,
replacing the "God Object" pattern where tools received the full app reference.

Created as part of CR-2026-001 to eliminate the God Object anti-pattern.

Design Principles (per TU-SCENE Condition C3):
1. Tools should NOT have direct access to model objects (Sketch, entities)
2. Tools should NOT have direct access to engine objects (Simulation)
3. All mutations must go through Commands
4. Read access is provided through query methods

What Tools CAN do:
- Execute commands (mutation path)
- Read view state (camera, selection)
- Write interaction state (selection, status, snap)
- Query geometry through read-only methods
- Paint/erase particles (BrushTool only)

What Tools CANNOT do:
- Import or access concrete entity types
- Directly modify Sketch or Simulation
- Access other UI components
- Bypass the command system
"""

from typing import Optional, Dict, Any, Tuple
from model.protocols import EntityType


class ToolContext:
    """
    Controlled interface for tools.

    This class acts as a facade over the application, exposing only
    the operations that tools are authorized to perform. This enforces
    the Air Gap principle by design.

    Usage:
        # In tool initialization (done by FlowStateApp):
        ctx = ToolContext(app)
        tool = LineTool(ctx)

        # In tool methods:
        def handle_event(self, event, layout):
            world_pos = self.ctx.screen_to_world(event.pos[0], event.pos[1])
            entity_idx = self.ctx.find_entity_at(world_pos[0], world_pos[1], 5.0)
            # ...
            self.ctx.execute(AddLineCommand(...))
    """

    def __init__(self, app):
        """
        Initialize with app reference (internal only).

        Args:
            app: FlowStateApp instance (hidden from tools)

        Note: Tools receive this context, not the app itself.
        The app reference is private to prevent bypass.
        """
        self._app = app

    # =========================================================================
    # Command Execution (Write Path) - TU-SCENE C3
    # =========================================================================

    def execute(self, command) -> bool:
        """
        Execute a command through the scene.

        This is the ONLY way tools should mutate application state.
        All geometry changes, constraint additions, property modifications
        must go through commands.

        Args:
            command: A Command instance from model.commands

        Returns:
            True if command executed successfully
        """
        return self._app.scene.execute(command)

    def discard(self) -> bool:
        """
        Discard the last command (for cancel operations).

        Unlike undo, discarded commands cannot be redone.
        Use this when the user cancels an in-progress operation.

        Returns:
            True if a command was discarded
        """
        return self._app.scene.discard()

    # =========================================================================
    # View State (Read-Only)
    # =========================================================================

    @property
    def zoom(self) -> float:
        """Current camera zoom level."""
        return self._app.session.camera.zoom

    @property
    def pan(self) -> Tuple[float, float]:
        """Current camera pan offset (pan_x, pan_y)."""
        return (self._app.session.camera.pan_x, self._app.session.camera.pan_y)

    @property
    def world_size(self) -> float:
        """World size in world units."""
        return self._app.scene.simulation.world_size

    @property
    def layout(self) -> dict:
        """Screen layout dictionary."""
        return self._app.layout

    @property
    def mode(self) -> int:
        """Current application mode (MODE_SIM or MODE_EDITOR)."""
        return self._app.session.mode

    @property
    def auto_atomize(self) -> bool:
        """Whether new geometry should be automatically atomized."""
        return self._app.session.auto_atomize

    # =========================================================================
    # Active Material (Read-Only Copy)
    # =========================================================================

    def get_active_material(self) -> Dict[str, Any]:
        """
        Get a copy of the currently selected material properties.

        Returns a copy to prevent direct mutation.

        Returns:
            Dictionary with sigma, epsilon, color, name
        """
        mat = self._app.session.active_material
        return {
            'name': mat.name,
            'sigma': mat.sigma,
            'epsilon': mat.epsilon,
            'color': mat.color,
        }

    # =========================================================================
    # Interaction State (Read/Write) - Transient State per State Classification
    # =========================================================================

    @property
    def interaction_state(self):
        """Current interaction state enum."""
        return self._app.session.state

    @interaction_state.setter
    def interaction_state(self, value):
        """Set interaction state."""
        self._app.session.state = value

    @property
    def selection(self):
        """
        Selection manager for reading/modifying selection.

        Provides:
            .walls: Set of selected entity indices
            .points: Set of selected (entity_idx, point_idx) tuples
            .clear(): Clear all selections
        """
        return self._app.session.selection

    @property
    def snap_target(self):
        """Current snap target (entity_idx, point_idx) or None."""
        return self._app.session.constraint_builder.snap_target

    @snap_target.setter
    def snap_target(self, value):
        """Set snap target for visual feedback."""
        self._app.session.constraint_builder.snap_target = value

    def set_status(self, message: str):
        """Set status bar message."""
        self._app.session.status.set(message)

    def play_sound(self, sound_id: str):
        """Play a UI sound effect."""
        self._app.sound_manager.play_sound(sound_id)

    # =========================================================================
    # Coordinate Transforms
    # =========================================================================

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.

        Args:
            screen_x, screen_y: Screen pixel coordinates

        Returns:
            (world_x, world_y) tuple
        """
        scene_rect = self.layout.get('scene', (0, 0, 0, 0))
        rel_x = screen_x - scene_rect[0]
        rel_y = screen_y - scene_rect[1]
        pan_x, pan_y = self.pan
        world_x = (rel_x - pan_x) / self.zoom
        world_y = (rel_y - pan_y) / self.zoom
        return (world_x, world_y)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates to screen coordinates.

        Args:
            world_x, world_y: World coordinates

        Returns:
            (screen_x, screen_y) tuple
        """
        scene_rect = self.layout.get('scene', (0, 0, 0, 0))
        pan_x, pan_y = self.pan
        screen_x = world_x * self.zoom + pan_x + scene_rect[0]
        screen_y = world_y * self.zoom + pan_y + scene_rect[1]
        return (screen_x, screen_y)

    # =========================================================================
    # Geometry Queries (Read-Only) - TU-SCENE C3
    # =========================================================================

    def get_entity_count(self) -> int:
        """Number of entities in sketch."""
        return len(self._app.scene.sketch.entities)

    def find_entity_at(self, world_x: float, world_y: float, radius: float) -> int:
        """
        Find entity at world coordinates.

        Args:
            world_x, world_y: World coordinates to search
            radius: Search radius in world units

        Returns:
            Entity index, or -1 if none found
        """
        return self._app.scene.sketch.find_entity_at(world_x, world_y, radius)

    def get_entity_type(self, index: int) -> Optional[EntityType]:
        """
        Get the type of an entity.

        Args:
            index: Entity index

        Returns:
            EntityType enum value, or None if index invalid
        """
        if 0 <= index < len(self._app.scene.sketch.entities):
            return self._app.scene.sketch.entities[index].entity_type
        return None

    def get_entity_render_data(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get rendering data for an entity.

        Args:
            index: Entity index

        Returns:
            Dictionary with entity_type, material_id, is_reference, and data,
            or None if invalid index
        """
        if 0 <= index < len(self._app.scene.sketch.entities):
            entity = self._app.scene.sketch.entities[index]
            return {
                'entity_type': entity.entity_type,
                'material_id': entity.material_id,
                'is_reference': entity.is_reference,
                'data': entity.get_render_data()
            }
        return None

    def get_entity_point(self, entity_idx: int, point_idx: int) -> Optional[Tuple[float, float]]:
        """
        Get a control point position from an entity.

        Args:
            entity_idx: Entity index
            point_idx: Point index on entity

        Returns:
            (x, y) tuple, or None if invalid
        """
        if 0 <= entity_idx < len(self._app.scene.sketch.entities):
            entity = self._app.scene.sketch.entities[entity_idx]
            if 0 <= point_idx < entity.point_count():
                return entity.get_point(point_idx)
        return None

    def get_entity_point_count(self, entity_idx: int) -> int:
        """
        Get number of control points for an entity.

        Args:
            entity_idx: Entity index

        Returns:
            Number of points, or 0 if invalid
        """
        if 0 <= entity_idx < len(self._app.scene.sketch.entities):
            return self._app.scene.sketch.entities[entity_idx].point_count()
        return 0

    def get_entity_parameter_at(self, entity_idx: int, world_x: float, world_y: float) -> Optional[float]:
        """
        Get parametric position on entity closest to point.

        Args:
            entity_idx: Entity index
            world_x, world_y: World coordinates

        Returns:
            Parameter (0-1), or None if invalid
        """
        if 0 <= entity_idx < len(self._app.scene.sketch.entities):
            return self._app.scene.sketch.entities[entity_idx].parameter_at(world_x, world_y)
        return None

    def is_entity_anchored(self, entity_idx: int, point_idx: int) -> bool:
        """
        Check if a point on an entity is anchored.

        Args:
            entity_idx: Entity index
            point_idx: Point index

        Returns:
            True if anchored, False otherwise
        """
        if 0 <= entity_idx < len(self._app.scene.sketch.entities):
            return self._app.scene.sketch.entities[entity_idx].get_anchored(point_idx)
        return False

    # =========================================================================
    # Snap Detection
    # =========================================================================

    def find_snap(self, screen_x: float, screen_y: float,
                  anchor_only: bool = False,
                  exclude_idx: int = -1) -> Optional[Tuple[int, int]]:
        """
        Find a snap target near screen coordinates.

        Args:
            screen_x, screen_y: Screen coordinates
            anchor_only: If True, only snap to anchored points
            exclude_idx: Entity index to exclude from search

        Returns:
            (entity_idx, point_idx) tuple, or None if no snap found
        """
        import core.utils as utils
        return utils.find_snap(
            self._app.scene.sketch,
            self.zoom, self.pan[0], self.pan[1],
            self.world_size, self.layout,
            anchor_only, exclude_idx,
            (screen_x, screen_y)
        )

    # =========================================================================
    # Brush Operations (BrushTool Only) - Routes through Scene per TU-SCENE C2
    # =========================================================================

    def paint_particles(self, world_x: float, world_y: float, radius: float,
                        material: Optional[Dict] = None) -> int:
        """
        Paint particles at world coordinates.

        Delegates to Scene which owns the ParticleBrush.

        Args:
            world_x, world_y: Center of brush in world coordinates
            radius: Brush radius in world units
            material: Optional material dict (uses active material if None)

        Returns:
            Number of particles added
        """
        return self._app.scene.paint_particles(world_x, world_y, radius, material)

    def erase_particles(self, world_x: float, world_y: float, radius: float) -> int:
        """
        Erase particles at world coordinates.

        Delegates to Scene which owns the ParticleBrush.

        Args:
            world_x, world_y: Center of brush in world coordinates
            radius: Brush radius in world units

        Returns:
            Number of particles removed
        """
        return self._app.scene.erase_particles(world_x, world_y, radius)

    def snapshot_particles(self):
        """
        Take a snapshot for particle undo.

        Call this before a brush operation to enable undo.
        """
        self._app.scene.simulation.snapshot()

    # =========================================================================
    # Interaction Data (for Solver Integration) - CR-2026-004 C-INPUT-01
    # =========================================================================

    def set_interaction_data(self, target_pos: Tuple[float, float],
                             entity_idx: int, handle_t: float):
        """
        Set interaction data for the solver's User Servo.

        The solver uses this to move geometry interactively while respecting
        constraints. The mouse position becomes a constraint target.

        Args:
            target_pos: World coordinates of mouse/drag target
            entity_idx: Index of entity being dragged
            handle_t: Parametric position on line (0.0=start, 1.0=end)
        """
        self._app.scene.sketch.interaction_data = {
            'target': target_pos,
            'entity_idx': entity_idx,
            'handle_t': handle_t
        }

    def update_interaction_target(self, target_pos: Tuple[float, float]):
        """
        Update the target position during an active drag.

        Args:
            target_pos: New world coordinates of drag target
        """
        if self._app.scene.sketch.interaction_data is not None:
            self._app.scene.sketch.interaction_data['target'] = target_pos

    def clear_interaction_data(self):
        """Clear interaction data when drag ends."""
        self._app.scene.sketch.interaction_data = None

    # =========================================================================
    # Constraint Builder Access - CR-2026-004 C3-UI
    # =========================================================================

    @property
    def constraint_builder(self):
        """
        Access to the constraint builder for binary constraint operations.

        Returns:
            ConstraintBuilder instance
        """
        return self._app.session.constraint_builder

    def clear_constraint_ui(self):
        """
        Clear constraint button selections in the UI.

        Per CR-2026-004 C-INPUT-02: Routes through app to handle UI state.
        """
        builder = self._app.session.constraint_builder
        builder.pending_type = None
        builder.snap_target = None
        # Clear button states if input_handler is available
        if hasattr(self._app, 'input_handler') and self._app.input_handler:
            for btn in self._app.input_handler.constraint_btn_map.keys():
                btn.is_active = False

    # =========================================================================
    # Coincident Constraint Factory - CR-2026-004 C1-UI
    # =========================================================================

    def create_coincident_command(self, e1_idx: int, p1_idx: int,
                                   e2_idx: int, p2_idx: int):
        """
        Create a Coincident constraint command via factory (preserves Air Gap).

        This method replaces direct Coincident imports in tools.

        Args:
            e1_idx: First entity index
            p1_idx: First point index
            e2_idx: Second entity index
            p2_idx: Second point index

        Returns:
            AddConstraintCommand or None if constraint cannot be created
        """
        constraint = self._app.scene.sketch.try_create_constraint(
            'COINCIDENT', [], [(e1_idx, p1_idx), (e2_idx, p2_idx)]
        )
        if constraint:
            from core.commands import AddConstraintCommand
            return AddConstraintCommand(self._app.scene.sketch, constraint)
        return None

    # =========================================================================
    # Entity Iteration (Read-Only) - CR-2026-004 C2-UI
    # =========================================================================

    def iter_entities(self):
        """
        Iterate over entities in read-only fashion.

        Yields (index, entity_data) tuples where entity_data is a read-only
        view containing entity_type and render_data.

        Yields:
            Tuple of (index, dict with 'entity_type' and 'render_data')
        """
        for i, entity in enumerate(self._app.scene.sketch.entities):
            yield i, {
                'entity_type': entity.entity_type,
                'render_data': entity.get_render_data(),
                'is_reference': entity.is_reference,
            }

    def get_entity_direct(self, index: int):
        """
        Get direct entity reference for command creation.

        WARNING: This provides direct model access. Use only for command
        instantiation or read-only queries. Do NOT mutate entities directly.

        Args:
            index: Entity index

        Returns:
            Entity instance or None if invalid
        """
        entities = self._app.scene.sketch.entities
        if 0 <= index < len(entities):
            return entities[index]
        return None

    # =========================================================================
    # Sketch Access (for Commands that need it) - Internal Use
    # =========================================================================

    def _get_sketch(self):
        """
        Get sketch reference for command creation.

        NOTE: This is intended for command instantiation only.
        Tools should not use this for direct sketch access.

        Returns:
            Sketch instance
        """
        return self._app.scene.sketch

    def _get_scene(self):
        """
        Get scene reference for advanced operations.

        NOTE: This is intended for operations that need full scene access,
        such as solver coordination. Use sparingly.

        Returns:
            Scene instance
        """
        return self._app.scene
