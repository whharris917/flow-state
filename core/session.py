"""
Session - Application Interaction State (Cleaned)

The Session is a thin coordinator that holds references to focused managers.
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

NOTE: All backward compatibility aliases have been REMOVED as of Session 2.
Access managers directly: session.camera, session.selection, 
session.constraint_builder, session.status
"""

from enum import Enum, auto
import core.config as config

from core.camera import CameraController
from core.selection import SelectionManager
from core.constraint_builder import ConstraintBuilder
from core.status_bar import StatusBar
from model.properties import Material, PRESET_MATERIALS


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
    Access managers directly - no aliases provided.
    
    Managers:
        session.camera - CameraController (zoom, pan, transforms)
        session.selection - SelectionManager (entity/point selection)
        session.constraint_builder - ConstraintBuilder (constraint creation)
        session.status - StatusBar (transient messages)
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
        # Active Material (for brush tool and new geometry atomization)
        # =====================================================================
        self.active_material = PRESET_MATERIALS['Water'].copy()
        self.material_library = dict(PRESET_MATERIALS)  # User can add to this
        
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
        
        # Input field reference (set by FlowStateApp)
        self.input_world = None
    
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
