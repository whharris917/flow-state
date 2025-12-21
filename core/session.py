import time
from enum import Enum, auto
import core.config as config


class InteractionState(Enum):
    IDLE = auto()
    PANNING = auto()
    PAINTING = auto()
    DRAGGING_GEOMETRY = auto()
    EDITING_CONSTRAINT = auto()

class Session:
    def __init__(self):
        # Core State
        self.mode = config.MODE_SIM
        self.state = InteractionState.IDLE
        
        # Camera - Active
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_mouse_pos = (0, 0)

        # Camera - Stored States (Persistent per mode)
        self.sim_view = {'zoom': 1.0, 'pan_x': 0.0, 'pan_y': 0.0}
        self.editor_view = {'zoom': 1.5, 'pan_x': 0.0, 'pan_y': 0.0}
        
        # Tool States (Persistent per mode)
        self.sim_tool = config.TOOL_BRUSH
        self.editor_tool = config.TOOL_SELECT
        self.current_tool = None # Current active tool instance
        self.tools = {}          # Dictionary to hold tool instances
        
        # Selection
        self.selected_walls = set()
        self.selected_points = set()
        
        # Constraint Construction State
        self.pending_constraint = None
        self.pending_targets_walls = []
        self.pending_targets_points = []
        self.current_snap_target = None
        
        # IO / Clipboard
        self.placing_geo_data = None
        
        # --- STATE STORAGE ---
        # Sim State is backed up when leaving Sim Mode
        self.sim_backup_state = None 
        # Editor State (Walls/Constraints) is stored here when in Sim Mode
        self.editor_storage = {'walls': [], 'constraints': []}
        
        # Simulation Running State Storage
        self.was_sim_running = False
        
        # File Paths
        self.current_sim_filepath = None
        self.current_geom_filepath = None
        
        # Status Bar
        self.status_msg = ""
        self.status_time = 0
        
        # UI References
        self.input_world = None
        self.temp_constraint_active = False
        
        # Playback State
        self.editor_paused = False
        
        # View Options
        self.show_wall_atoms = True 
        self.show_constraints = True 

    def set_status(self, msg):
        self.status_msg = msg
        self.status_time = time.time()

    def change_tool(self, tool_id):
        if hasattr(self, 'current_tool') and self.current_tool:
            self.current_tool.deactivate()
        
        if hasattr(self, 'tools') and tool_id in self.tools:
            self.current_tool = self.tools[tool_id]
            self.current_tool.activate()
            # Store the tool selection based on current mode
            if self.mode == config.MODE_SIM: self.sim_tool = tool_id
            else: self.editor_tool = tool_id