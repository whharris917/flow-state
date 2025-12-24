"""
Scene - The Document Container & Orchestrator

The Scene is the central "document" in Flow State. It contains and OWNS:
- Sketch: CAD geometry, constraints, materials, solver
- Simulation: Particle physics (atoms only, no geometry knowledge)
- Compiler: The one-way bridge from Sketch → Simulation
- GeometryManager: Import/export operations for CAD data
- CommandQueue: Undo/redo for CAD operations

The Scene also serves as the ORCHESTRATOR, managing the correct
order of operations during each frame update.

File formats:
- .mdl (Model): Sketch only - reusable CAD components
- .scn (Scene): Full state - Sketch + Simulation + View

Architecture:
- Scene OWNS: Sketch, Simulation, Compiler, GeometryManager, CommandQueue
- Sketch OWNS: entities, constraints, materials
- Simulation OWNS: particle arrays (pure physics)
- Compiler READS Sketch, WRITES to Simulation (one-way bridge)
"""

import json
import os

from model.sketch import Sketch
from model.simulation_geometry import GeometryManager
from engine.simulation import Simulation
from engine.compiler import Compiler
from core.commands import CommandQueue


class Scene:
    """
    The document container and orchestrator.
    
    Owns:
        - Sketch (CAD domain)
        - Simulation (Physics domain)
        - Compiler (bridge between them)
        - GeometryManager (CAD import/export)
        - CommandQueue (undo/redo for CAD)
    
    Orchestrates:
        - Frame updates (drivers → constraints → compile → physics)
        - Geometry compilation (sketch → atoms)
        - Undo/redo via commands
    """
    
    def __init__(self, skip_warmup=False):
        """
        Creates a new empty Scene.
        
        Args:
            skip_warmup: If True, skip Numba JIT warmup (for Editor-only mode)
        """
        # CAD Domain
        self.sketch = Sketch()
        
        # CAD Import/Export Helper
        self.geo = GeometryManager(self.sketch)
        
        # Physics Domain (pure - no CAD knowledge)
        self.simulation = Simulation(skip_warmup=skip_warmup)
        
        # Bridge (knows both domains)
        self.compiler = Compiler(self.sketch, self.simulation)
        
        # Command Queue for CAD undo/redo
        self.commands = CommandQueue(max_history=50)
        
        # Track geometry changes for rebuild optimization
        self._geometry_dirty = False
    
    # =========================================================================
    # Convenience Aliases (for common access patterns)
    # =========================================================================
    
    @property
    def entities(self):
        """Alias for sketch.entities."""
        return self.sketch.entities
    
    @property
    def constraints(self):
        """Alias for sketch.constraints."""
        return self.sketch.constraints
    
    @property
    def materials(self):
        """Alias for sketch.materials."""
        return self.sketch.materials
    
    # =========================================================================
    # Command Interface (CAD Undo/Redo)
    # =========================================================================
    
    def execute(self, command):
        """
        Execute a command and add to history.
        After execution, marks geometry dirty for rebuild.
        
        Args:
            command: A Command instance
            
        Returns:
            True if command executed successfully
        """
        result = self.commands.execute(command)
        if result:
            self._geometry_dirty = True
        return result
    
    def undo(self):
        """
        Undo the last CAD command.
        
        Returns:
            True if undo was successful
        """
        result = self.commands.undo()
        if result:
            self._geometry_dirty = True
            self.rebuild()
        return result
    
    def redo(self):
        """
        Redo the last undone CAD command.
        
        Returns:
            True if redo was successful
        """
        result = self.commands.redo()
        if result:
            self._geometry_dirty = True
            self.rebuild()
        return result
    
    def discard(self):
        """
        Discard the last CAD command. Like undo but CANNOT be redone.
        
        Use this for canceling preview operations where the user
        explicitly abandoned the operation (e.g., pressing Escape
        while drawing a line).
        
        Returns:
            True if discard was successful
        """
        result = self.commands.discard()
        if result:
            self._geometry_dirty = True
            self.rebuild()
        return result

    def can_undo(self):
        """Check if CAD undo is available."""
        return self.commands.can_undo()
    
    def can_redo(self):
        """Check if CAD redo is available."""
        return self.commands.can_redo()
    
    # =========================================================================
    # Orchestrator: Frame Update
    # =========================================================================
    
    def update(self, dt, geo_time, run_physics=True, physics_steps=1):
        """
        Main frame update - orchestrates the correct order of operations.
        
        This is the SINGLE ENTRY POINT for frame updates. It ensures:
        1. Constraint drivers are updated (animations)
        2. Constraints are solved (geometry moves)
        3. Static atoms are rebuilt if geometry changed
        4. Physics step runs (if enabled)
        
        Args:
            dt: Delta time since last frame (seconds)
            geo_time: Current geometry/animation time (seconds)
            run_physics: If True, run physics simulation step
            physics_steps: Number of physics sub-steps to run
        
        Returns:
            True if geometry was modified (rebuild occurred)
        """
        geometry_changed = False
        
        # 1. Update constraint drivers (animation)
        if self.sketch.constraints:
            old_values = self._snapshot_constraint_values()
            self.sketch.update_drivers(geo_time)
            new_values = self._snapshot_constraint_values()
            
            if old_values != new_values:
                geometry_changed = True
            
            # 2. Solve constraints
            self.sketch.solve()
            geometry_changed = True  # Conservative: assume solve moved something
        
        # 3. Rebuild static atoms if geometry changed
        if geometry_changed or self._geometry_dirty:
            self.rebuild()
            self._geometry_dirty = False
        
        # 4. Run physics step (if enabled)
        if run_physics and not self.simulation.paused:
            self.simulation.step(physics_steps)
        
        return geometry_changed
    
    def _snapshot_constraint_values(self):
        """Capture current constraint values for change detection."""
        return tuple(
            getattr(c, 'value', None) 
            for c in self.sketch.constraints
        )
    
    # =========================================================================
    # Compiler Interface
    # =========================================================================
    
    def rebuild(self):
        """
        Recompiles Sketch geometry into Simulation atoms.
        Call this after geometry changes that should affect physics.
        """
        self.compiler.rebuild()
    
    def mark_dirty(self):
        """Mark geometry as needing rebuild on next update."""
        self._geometry_dirty = True
    
    # =========================================================================
    # Model I/O (.mdl) - Sketch Only
    # =========================================================================
    
    def save_model(self, path):
        """
        Saves the Sketch as a .mdl file (CAD geometry only).
        
        Args:
            path: File path (should end in .mdl)
            
        Returns:
            Status message string
        """
        if not path:
            return "Cancelled"
        
        if not self.sketch.entities:
            return "No geometry to save"
        
        data = {
            'version': '1.0',
            'type': 'MODEL',
            'sketch': self.sketch.to_dict()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return f"Model saved: {os.path.basename(path)}"
        except Exception as e:
            return f"Save error: {e}"
    
    def import_model(self, path, offset_x=None, offset_y=None):
        """
        Imports a .mdl file, merging geometry into the current Sketch.
        
        Args:
            path: File path to .mdl file
            offset_x: X position for placement (None = original position)
            offset_y: Y position for placement (None = original position)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not path:
            return False, "Cancelled"
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if data.get('type') != 'MODEL':
                return False, "Invalid model file format"
            
            sketch_data = data.get('sketch', {})
            
            if offset_x is not None and offset_y is not None:
                self._import_sketch_at_offset(sketch_data, offset_x, offset_y)
            else:
                self._import_sketch_direct(sketch_data)
            
            self._geometry_dirty = True
            return True, f"Model imported: {os.path.basename(path)}"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Import error: {e}"
    
    def _import_sketch_at_offset(self, sketch_data, offset_x, offset_y):
        """Import sketch data at a specific world position."""
        from model.geometry import Line, Circle, Point
        from model.constraints import create_constraint
        from model.properties import Material
        
        entities_data = sketch_data.get('entities', [])
        if not entities_data:
            return
        
        # Calculate bounding box to find centroid
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for e in entities_data:
            if e['type'] == 'line':
                min_x = min(min_x, e['start'][0], e['end'][0])
                max_x = max(max_x, e['start'][0], e['end'][0])
                min_y = min(min_y, e['start'][1], e['end'][1])
                max_y = max(max_y, e['start'][1], e['end'][1])
            elif e['type'] == 'circle':
                min_x = min(min_x, e['center'][0] - e['radius'])
                max_x = max(max_x, e['center'][0] + e['radius'])
                min_y = min(min_y, e['center'][1] - e['radius'])
                max_y = max(max_y, e['center'][1] + e['radius'])
            elif e['type'] == 'point':
                min_x = min(min_x, e['x'])
                max_x = max(max_x, e['x'])
                min_y = min(min_y, e['y'])
                max_y = max(max_y, e['y'])
        
        if min_x == float('inf'):
            return
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        dx = offset_x - center_x
        dy = offset_y - center_y
        
        # Import materials first
        for mat_name, mat_data in sketch_data.get('materials', {}).items():
            if mat_name not in self.sketch.materials:
                self.sketch.materials[mat_name] = Material.from_dict(mat_data)
        
        base_idx = len(self.sketch.entities)
        
        # Import entities with offset
        for e in entities_data:
            if e['type'] == 'line':
                start = (e['start'][0] + dx, e['start'][1] + dy)
                end = (e['end'][0] + dx, e['end'][1] + dy)
                line = Line(start, end, e.get('ref', False), e.get('material_id', 'Default'))
                line.anchored = e.get('anchored', [False, False])
                if 'anim' in e:
                    line.anim = e['anim'].copy()
                    line.anim['ref_start'] = line.start.copy()
                    line.anim['ref_end'] = line.end.copy()
                self.sketch.entities.append(line)
                
            elif e['type'] == 'circle':
                center = (e['center'][0] + dx, e['center'][1] + dy)
                circle = Circle(center, e['radius'], e.get('material_id', 'Default'))
                circle.anchored = e.get('anchored', [False])
                if 'anim' in e:
                    circle.anim = e['anim'].copy()
                self.sketch.entities.append(circle)
                
            elif e['type'] == 'point':
                point = Point(e['x'] + dx, e['y'] + dy, e.get('anchored', False), e.get('material_id', 'Default'))
                self.sketch.entities.append(point)
        
        # Import constraints with remapped indices
        for c_data in sketch_data.get('constraints', []):
            remapped = self._remap_constraint_indices(c_data, base_idx)
            c = create_constraint(remapped)
            if c:
                self.sketch.constraints.append(c)
    
    def _import_sketch_direct(self, sketch_data):
        """Import sketch data at its original coordinates."""
        from model.geometry import Line, Circle, Point
        from model.constraints import create_constraint
        from model.properties import Material
        
        for mat_name, mat_data in sketch_data.get('materials', {}).items():
            if mat_name not in self.sketch.materials:
                self.sketch.materials[mat_name] = Material.from_dict(mat_data)
        
        base_idx = len(self.sketch.entities)
        
        for e in sketch_data.get('entities', []):
            if e['type'] == 'line':
                self.sketch.entities.append(Line.from_dict(e))
            elif e['type'] == 'circle':
                self.sketch.entities.append(Circle.from_dict(e))
            elif e['type'] == 'point':
                self.sketch.entities.append(Point.from_dict(e))
        
        for c_data in sketch_data.get('constraints', []):
            remapped = self._remap_constraint_indices(c_data, base_idx)
            c = create_constraint(remapped)
            if c:
                self.sketch.constraints.append(c)
    
    def _remap_constraint_indices(self, c_data, base_idx):
        """Remap constraint indices by adding base_idx offset."""
        c_copy = c_data.copy()
        new_indices = []
        
        for item in c_data.get('indices', []):
            if isinstance(item, (list, tuple)):
                new_indices.append([item[0] + base_idx, item[1]])
            else:
                new_indices.append(item + base_idx)
        
        c_copy['indices'] = new_indices
        return c_copy
    
    # =========================================================================
    # Scene I/O (.scn) - Full State
    # =========================================================================
    
    def save_scene(self, path, view_state=None):
        """
        Saves the full Scene (Sketch + Simulation + View) as a .scn file.
        
        Args:
            path: File path (should end in .scn)
            view_state: Dict with zoom, pan_x, pan_y (from Session)
            
        Returns:
            Status message string
        """
        if not path:
            return "Cancelled"
        
        data = {
            'version': '1.0',
            'type': 'SCENE',
            'sketch': self.sketch.to_dict(),
            'simulation': self.simulation.to_dict(),
        }
        
        if view_state:
            data['view'] = view_state
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return f"Scene saved: {os.path.basename(path)}"
        except Exception as e:
            return f"Save error: {e}"
    
    @classmethod
    def load_scene(cls, path, skip_warmup=False):
        """
        Loads a .scn file and returns a new Scene instance.
        
        Args:
            path: File path to .scn file
            skip_warmup: If True, skip Numba JIT warmup
            
        Returns:
            Tuple of (Scene or None, view_state or None, message: str)
        """
        if not path:
            return None, None, "Cancelled"
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if data.get('type') != 'SCENE':
                return None, None, "Invalid scene file format"
            
            scene = cls(skip_warmup=skip_warmup)
            
            if 'sketch' in data:
                scene.sketch.restore(data['sketch'])
            
            if 'simulation' in data:
                scene.simulation.restore(data['simulation'])
            
            view_state = data.get('view', None)
            
            return scene, view_state, f"Scene loaded: {os.path.basename(path)}"
            
        except json.JSONDecodeError as e:
            return None, None, f"Invalid JSON: {e}"
        except Exception as e:
            return None, None, f"Load error: {e}"
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def clear(self):
        """Clears all content from the Scene."""
        self.sketch.clear()
        self.simulation.clear(snapshot=False)
        self.commands.clear()
        self._geometry_dirty = False
    
    def new(self):
        """Resets the Scene to a fresh state."""
        self.sketch.clear()
        self.simulation.reset()
        self.commands.clear()
        self._geometry_dirty = False