"""
Scene - The Document Container & Orchestrator

The Scene is the central "document" in Flow State. It contains and OWNS:
- Sketch: CAD geometry, constraints, materials, solver
- Simulation: Particle physics (atoms only, no geometry knowledge)
- Compiler: The one-way bridge from Sketch → Simulation
- GeometryManager: Import/export operations for CAD data
- CommandQueue: Undo/redo for CAD operations
- ProcessObjects: Simulation boundary conditions (Sources, Sinks, etc.)

The Scene also serves as the ORCHESTRATOR, managing the correct
order of operations during each frame update.

File formats:
- .mdl (Model): Sketch only - reusable CAD components
- .scn (Scene): Full state - Sketch + Simulation + View + ProcessObjects

Architecture:
- Scene OWNS: Sketch, Simulation, Compiler, GeometryManager, CommandQueue, ProcessObjects
- Sketch OWNS: entities, constraints, materials
- Simulation OWNS: particle arrays (pure physics)
- Compiler READS Sketch, WRITES to Simulation (one-way bridge)
- ProcessObjects sit ALONGSIDE Sketch, exposing handles to it
"""

import json
import os

from model.sketch import Sketch
from model.simulation_geometry import GeometryManager
from model.properties import MaterialManager
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
        - ProcessObjects (boundary conditions: Sources, Sinks, etc.)
    
    Orchestrates:
        - Frame updates (drivers → constraints → compile → process objects → physics)
        - Geometry compilation (sketch → atoms)
        - ProcessObject execution (particle spawning, etc.)
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

        # Material Manager (single source of truth for materials)
        # Note: on_rebuild callback set after self.rebuild is defined
        self.material_manager = MaterialManager(self.sketch)

        # CAD Import/Export Helper
        self.geo = GeometryManager(self.sketch)
        
        # Physics Domain (pure - no CAD knowledge)
        self.simulation = Simulation(skip_warmup=skip_warmup)
        
        # Bridge (knows both domains)
        self.compiler = Compiler(self.sketch, self.simulation)
        
        # Command Queue for CAD undo/redo
        self.commands = CommandQueue(max_history=50)

        # ProcessObjects (Sources, Sinks, Heaters, Sensors)
        self.process_objects = []

        # --- Dirty Flags for Rebuild Optimization ---
        # topology_dirty: Entity count/material changed → requires full rebuild()
        # geometry_moved: Positions changed → only requires sync_entity_arrays()
        self._topology_dirty = False
        self._geometry_dirty = False  # Legacy flag, kept for compatibility

        # Connect material manager rebuild callback now that self.rebuild exists
        self.material_manager.on_rebuild = self.rebuild
    
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
    # ProcessObject Management
    # =========================================================================
    
    def add_process_object(self, obj):
        """
        Add a ProcessObject to the scene.
        
        This registers the object's handles with the Sketch so they can
        participate in constraints.
        
        Args:
            obj: ProcessObject instance (Source, Sink, etc.)
        """
        self.process_objects.append(obj)
        obj._owner_scene = self
        obj.register_handles(self.sketch)
    
    def remove_process_object(self, obj):
        """
        Remove a ProcessObject from the scene.
        
        This unregisters the object's handles from the Sketch.
        
        Args:
            obj: ProcessObject instance to remove
        """
        if obj in self.process_objects:
            obj.unregister_handles(self.sketch)
            obj._owner_scene = None
            self.process_objects.remove(obj)
    
    def get_process_object_for_handle(self, point):
        """
        Find the ProcessObject that owns a given handle Point.
        
        Args:
            point: A Point entity that might be a handle
            
        Returns:
            ProcessObject or None
        """
        if not getattr(point, 'is_handle', False):
            return None
        return getattr(point, '_owner_process_object', None)
    
    def find_process_object_at(self, x, y, tolerance=5.0):
        """
        Find a ProcessObject at the given world coordinates.
        
        Args:
            x, y: World coordinates
            tolerance: Hit tolerance in world units
            
        Returns:
            ProcessObject or None
        """
        for obj in self.process_objects:
            if hasattr(obj, 'hit_test') and obj.hit_test(x, y, tolerance):
                return obj
        return None
    
    # =========================================================================
    # Command Interface (CAD Undo/Redo)
    # =========================================================================
    
    def execute(self, command):
        """
        Execute a command and add to history.

        Sets appropriate dirty flags based on command type:
        - changes_topology=True → _topology_dirty (requires full rebuild)
        - changes_topology=False → _geometry_dirty (requires sync only)

        Args:
            command: A Command instance

        Returns:
            True if command executed successfully
        """
        result = self.commands.execute(command)
        if result:
            if getattr(command, 'changes_topology', False):
                self._topology_dirty = True
            else:
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
            # Conservative: undo always triggers full rebuild
            # (we don't track what the undone command was)
            self._topology_dirty = True
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
            # Conservative: redo always triggers full rebuild
            self._topology_dirty = True
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
            # Conservative: discard always triggers full rebuild
            self._topology_dirty = True
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

        This is the SINGLE ENTRY POINT for frame updates. Implements the
        new orchestration for two-way coupling:

        1. TOPOLOGY CHECK: Full rebuild if entities added/removed
        2. DRIVER UPDATE: Animate constraint values
        3. SYNC ENTITY ARRAYS: Copy positions to physics arrays (fast path)
        4. PHYSICS SUB-LOOP: (Future) Tether forces + entity integration
        5. CONSTRAINT SOLVE: PBD solver corrects geometry
        6. POST-CONSTRAINT SYNC: Update anchor positions after solve
        7. PROCESS OBJECTS: Sources emit, drains absorb
        8. PHYSICS STEP: Particle simulation

        Args:
            dt: Delta time since last frame (seconds)
            geo_time: Current geometry/animation time (seconds)
            run_physics: If True, run physics simulation step
            physics_steps: Number of physics sub-steps to run

        Returns:
            True if geometry was modified
        """
        geometry_moved = False

        # 1. TOPOLOGY CHECK - Full rebuild if entity count/material changed
        if self._topology_dirty:
            self.rebuild()
            self._topology_dirty = False
            self._geometry_dirty = False  # Rebuild covers sync

        # 2. DRIVER UPDATE - Animate constraint values
        if self.sketch.constraints:
            old_values = self._snapshot_constraint_values()
            self.sketch.update_drivers(geo_time)
            new_values = self._snapshot_constraint_values()

            if old_values != new_values:
                geometry_moved = True

        # 3. SYNC ENTITY ARRAYS (Before Physics) - Fast path for position updates
        # This copies current entity positions into flat arrays for physics kernel
        if self._geometry_dirty or geometry_moved:
            self.simulation.sync_entity_arrays(self.sketch.entities)
            # Also sync static atoms immediately so they don't lag during drags
            self.simulation.sync_static_atoms_to_geometry()
            self._geometry_dirty = False
            geometry_moved = True  # Mark that geometry changed

        # 4. PHYSICS SUB-LOOP - Two-way coupling between entities and particles
        # This is where tethered atoms and dynamic entities interact
        if run_physics and not self.simulation.paused:
            self._run_physics_coupling(dt)
            geometry_moved = True  # Entities may have moved

        # 5. CONSTRAINT SOLVE - PBD iterations
        # Run if constraints exist OR if user is interacting (User Servo)
        if self.sketch.constraints or self.sketch.interaction_data:
            self.sketch.solve()
            geometry_moved = True  # Conservative: assume solve moved something

        # 6. POST-CONSTRAINT SYNC - Update anchor positions after solver
        # This ensures physics kernel sees the constrained positions
        if geometry_moved:
            self.simulation.sync_entity_arrays(self.sketch.entities)
            # Teleport static atoms to follow their parent entities
            self.simulation.sync_static_atoms_to_geometry()

        # 7. PROCESS OBJECTS - Execute sources, drains, etc.
        if run_physics and not self.simulation.paused:
            for obj in self.process_objects:
                if obj.enabled:
                    obj.execute(self.simulation, dt)

        # 8. PHYSICS STEP - Particle simulation
        if run_physics and not self.simulation.paused:
            self.simulation.step(physics_steps)

        return geometry_moved
    
    def _snapshot_constraint_values(self):
        """Capture current constraint values for change detection."""
        return tuple(
            getattr(c, 'value', None)
            for c in self.sketch.constraints
        )

    def _run_physics_coupling(self, dt):
        """
        Execute the two-way physics coupling between entities and tethered atoms.

        This implements the core of dynamic rigid body interaction:
        1. Clear entity force accumulators
        2. Apply tether forces (atoms pull on entities, entities pull on atoms)
        3. Retrieve accumulated forces from simulation
        4. Apply forces to entity Python objects
        5. Integrate entity velocities and positions
        6. Sync updated entity positions back to simulation arrays

        Args:
            dt: Delta time for integration
        """
        entities = self.sketch.entities

        # Skip if no entities
        if not entities:
            return

        # Step A: Sync entity arrays (Sketch -> Sim)
        # This was already done in step 3 of update(), but we may need it
        # for sub-stepping in the future

        # Step B: Clear entity force accumulators
        self.simulation.clear_entity_forces()

        # Also clear Python-side accumulators for dynamic entities
        for entity in entities:
            if entity.dynamic:
                entity.clear_accumulators()

        # Step C: Apply tether forces (Numba kernel)
        # This computes spring forces between tethered atoms and their anchors
        # and accumulates reaction forces on entities
        self.simulation.apply_tether_forces()

        # Step D: Retrieve forces from simulation and apply to entities
        entity_forces = self.simulation.get_entity_forces()

        for i, entity in enumerate(entities):
            if not entity.dynamic:
                continue

            # Get accumulated force and torque from tethered atoms
            fx = entity_forces[i, 0]
            fy = entity_forces[i, 1]
            torque = entity_forces[i, 2]

            # Apply to entity
            entity.apply_force(fx, fy)
            entity.apply_torque(torque)

        # Step E: Integrate entity dynamics
        for entity in entities:
            if entity.dynamic:
                entity.integrate(dt)

        # Step F: Sync updated entity positions back to simulation
        # This updates anchor positions for the next physics step
        self.simulation.sync_entity_arrays(entities)
    
    # =========================================================================
    # Compiler Interface
    # =========================================================================
    
    def rebuild(self):
        """
        Recompiles Sketch geometry into Simulation atoms.
        Call this after geometry changes that should affect physics.
        """
        self.compiler.rebuild()
        # Snap tethered atoms to exact anchor positions for cold start
        # This prevents oscillation from floating-point precision differences
        self.simulation.snap_tethered_atoms_to_anchors()
    
    def mark_dirty(self, topology=False):
        """
        Mark geometry as needing update on next frame.

        Args:
            topology: If True, marks for full rebuild (entity changes).
                      If False (default), marks for sync only (position changes).
        """
        if topology:
            self._topology_dirty = True
        else:
            self._geometry_dirty = True

    def mark_topology_dirty(self):
        """Mark for full rebuild (entity add/remove, material change)."""
        self._topology_dirty = True
    
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

            self._topology_dirty = True  # Import adds entities
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
        
        for mat_name, mat_data in sketch_data.get('materials', {}).items():
            if mat_name not in self.sketch.materials:
                self.sketch.materials[mat_name] = Material.from_dict(mat_data)
        
        base_idx = len(self.sketch.entities)
        dx, dy = offset_x, offset_y
        
        for e in sketch_data.get('entities', []):
            if e['type'] == 'line':
                start = (e['start'][0] + dx, e['start'][1] + dy)
                end = (e['end'][0] + dx, e['end'][1] + dy)
                line = Line(start, end, e.get('ref', False), e.get('material_id', 'Wall'))
                line.anchored = e.get('anchored', [False, False])
                if 'anim' in e:
                    line.anim = e['anim'].copy()
                    line.anim['ref_start'] = line.start.copy()
                    line.anim['ref_end'] = line.end.copy()
                self.sketch.entities.append(line)
                
            elif e['type'] == 'circle':
                center = (e['center'][0] + dx, e['center'][1] + dy)
                circle = Circle(center, e['radius'], e.get('material_id', 'Wall'))
                circle.anchored = e.get('anchored', [False])
                if 'anim' in e:
                    circle.anim = e['anim'].copy()
                self.sketch.entities.append(circle)
                
            elif e['type'] == 'point':
                point = Point(e['x'] + dx, e['y'] + dy, e.get('anchored', False), e.get('material_id', 'Wall'))
                self.sketch.entities.append(point)
        
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
    # Scene I/O (.scn) - Full State Including ProcessObjects
    # =========================================================================
    
    def save_scene(self, path, view_state=None):
        """
        Saves the full Scene (Sketch + Simulation + ProcessObjects + View) as a .scn file.
        
        Args:
            path: File path (should end in .scn)
            view_state: Dict with zoom, pan_x, pan_y (from Session)
            
        Returns:
            Status message string
        """
        if not path:
            return "Cancelled"
        
        data = {
            'version': '1.1',
            'type': 'SCENE',
            'sketch': self.sketch.to_dict(),
            'simulation': self.simulation.to_dict(),
            'process_objects': [obj.to_dict() for obj in self.process_objects],
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
            
            # Restore ProcessObjects
            if 'process_objects' in data:
                from model.process_objects import create_process_object
                for obj_data in data['process_objects']:
                    obj = create_process_object(obj_data)
                    if obj:
                        scene.add_process_object(obj)
            
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
        # Remove all ProcessObjects first (unregisters handles)
        for obj in list(self.process_objects):
            self.remove_process_object(obj)

        self.sketch.clear()
        self.simulation.clear(snapshot=False)
        self.commands.clear()
        self._topology_dirty = False
        self._geometry_dirty = False

    def new(self):
        """Resets the Scene to a fresh state."""
        # Remove all ProcessObjects first
        for obj in list(self.process_objects):
            self.remove_process_object(obj)

        self.sketch.clear()
        self.simulation.reset()
        self.commands.clear()
        self._topology_dirty = False
        self._geometry_dirty = False
