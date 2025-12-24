"""
Session - Application Interaction State (Refactored)

The Session is now a thin coordinator that holds references to focused managers.
Each manager owns a specific concern:

- CameraController: View transforms, zoom, pan
- SelectionManager: Entity and point selection
- ConstraintBuilder: Pending constraint construction
- StatusBar: Transient message display

The Session itself owns only:
- Mode (SIM vs EDITOR)
- Interaction state (IDLE, PANNING, PAINTING, etc.)
- Tool management
- View options (show_wall_atoms, show_constraints)
- Playback state (editor_paused, geo_time)

File I/O state (filepaths, placing_geo_data) remains here for now
but could be extracted to a FileManager in the future.

BACKWARD COMPATIBILITY:
This module provides property aliases so existing code continues to work.
E.g., session.zoom delegates to session.camera.zoom
"""

import time
from enum import Enum, auto
import core.config as config

from core.camera import CameraController
from core.selection import SelectionManager
from core.constraint_builder import ConstraintBuilder
from core.status_bar import StatusBar


class InteractionState(Enum):
    """Enumeration of possible interaction states."""
    IDLE = auto()
    PANNING = auto()
    PAINTING = auto()
    DRAGGING_GEOMETRY = auto()
    EDITING_CONSTRAINT = auto()


class Session:
    """
    Application interaction state coordinator.
    
    Delegates to focused managers for specific concerns.
    Provides backward-compatible property aliases.
    """
    
    def __init__(self):
        # =====================================================================
        # Core State (owned directly)
        # =====================================================================
        self.mode = config.MODE_SIM
        self.state = InteractionState.IDLE
        
        # =====================================================================
        # Focused Managers (composition over monolith)
        # =====================================================================
        self.camera = CameraController()
        self.selection = SelectionManager()
        self.constraint_builder = ConstraintBuilder()
        self.status = StatusBar()
        
        # =====================================================================
        # Tool Management
        # =====================================================================
        self.sim_tool = config.TOOL_BRUSH
        self.editor_tool = config.TOOL_SELECT
        self.current_tool = None
        self.tools = {}
        
        # =====================================================================
        # View Options
        # =====================================================================
        self.show_wall_atoms = True
        self.show_constraints = True
        
        # =====================================================================
        # Playback State
        # =====================================================================
        self.editor_paused = False
        self.geo_time = 0.0
        
        # =====================================================================
        # File I/O State (could be extracted to FileManager)
        # =====================================================================
        self.placing_geo_data = None
        self.current_sim_filepath = None
        self.current_geom_filepath = None
        
        # =====================================================================
        # State Storage (for mode switching)
        # =====================================================================
        self.sim_backup_state = None
        self.editor_storage = {'walls': [], 'constraints': []}
        self.was_sim_running = False
    
    # =========================================================================
    # Camera Property Aliases (Backward Compatibility)
    # =========================================================================
    
    @property
    def zoom(self):
        return self.camera.zoom
    
    @zoom.setter
    def zoom(self, value):
        self.camera.zoom = value
    
    @property
    def pan_x(self):
        return self.camera.pan_x
    
    @pan_x.setter
    def pan_x(self, value):
        self.camera.pan_x = value
    
    @property
    def pan_y(self):
        return self.camera.pan_y
    
    @pan_y.setter
    def pan_y(self, value):
        self.camera.pan_y = value
    
    @property
    def last_mouse_pos(self):
        return self.camera.last_mouse_pos
    
    @last_mouse_pos.setter
    def last_mouse_pos(self, value):
        self.camera.last_mouse_pos = value
    
    @property
    def sim_view(self):
        return self.camera._stored_views[config.MODE_SIM]
    
    @sim_view.setter
    def sim_view(self, value):
        self.camera._stored_views[config.MODE_SIM] = value
    
    @property
    def editor_view(self):
        return self.camera._stored_views[config.MODE_EDITOR]
    
    @editor_view.setter
    def editor_view(self, value):
        self.camera._stored_views[config.MODE_EDITOR] = value
    
    # =========================================================================
    # Selection Property Aliases (Backward Compatibility)
    # =========================================================================
    
    @property
    def selected_walls(self):
        return self.selection.walls
    
    @selected_walls.setter
    def selected_walls(self, value):
        self.selection.walls = value
    
    @property
    def selected_points(self):
        return self.selection.points
    
    @selected_points.setter
    def selected_points(self, value):
        self.selection.points = value
    
    # =========================================================================
    # Constraint Builder Property Aliases (Backward Compatibility)
    # =========================================================================
    
    @property
    def pending_constraint(self):
        return self.constraint_builder.pending_constraint
    
    @pending_constraint.setter
    def pending_constraint(self, value):
        self.constraint_builder.pending_constraint = value
    
    @property
    def pending_targets_walls(self):
        return self.constraint_builder.pending_targets_walls
    
    @pending_targets_walls.setter
    def pending_targets_walls(self, value):
        self.constraint_builder.pending_targets_walls = value
    
    @property
    def pending_targets_points(self):
        return self.constraint_builder.pending_targets_points
    
    @pending_targets_points.setter
    def pending_targets_points(self, value):
        self.constraint_builder.pending_targets_points = value
    
    @property
    def current_snap_target(self):
        return self.constraint_builder.current_snap_target
    
    @current_snap_target.setter
    def current_snap_target(self, value):
        self.constraint_builder.current_snap_target = value
    
    # =========================================================================
    # Status Property Aliases (Backward Compatibility)
    # =========================================================================
    
    @property
    def status_msg(self):
        return self.status.status_msg
    
    @status_msg.setter
    def status_msg(self, value):
        self.status.status_msg = value
    
    @property
    def status_time(self):
        return self.status.status_time
    
    @status_time.setter
    def status_time(self, value):
        self.status.status_time = value
    
    def set_status(self, msg):
        """Set a status message (delegated to StatusBar)."""
        self.status.set(msg)
    
    # =========================================================================
    # Tool Management
    # =========================================================================
    
    def change_tool(self, tool_id):
        """
        Switch to a different tool.
        
        Args:
            tool_id: The tool constant (e.g., config.TOOL_BRUSH)
        """
        # Deactivate current tool
        if self.current_tool:
            self.current_tool.deactivate()
        
        # Activate new tool
        if tool_id in self.tools:
            self.current_tool = self.tools[tool_id]
            self.current_tool.activate()
            
            # Store the tool selection based on current mode
            if self.mode == config.MODE_SIM:
                self.sim_tool = tool_id
            else:
                self.editor_tool = tool_id
    
    # =========================================================================
    # Mode Switching Helpers
    # =========================================================================
    
    def store_camera_for_mode(self, mode):
        """Store current camera state for a mode."""
        self.camera.store_view(mode)
    
    def restore_camera_for_mode(self, mode):
        """Restore camera state for a mode."""
        self.camera.restore_view(mode)
    
    def clear_interaction_state(self):
        """Clear all transient interaction state."""
        self.state = InteractionState.IDLE
        self.selection.clear()
        self.constraint_builder.reset()
        self.placing_geo_data = None