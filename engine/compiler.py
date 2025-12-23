import numpy as np
import math
from model.geometry import Line, Circle

class Compiler:
    """
    The Bridge between the Sketch (CAD) and the Simulation (Physics).
    Reads geometry and materials, writes to static particle arrays.
    """
    def __init__(self, sketch, simulation):
        """
        Initialize the Compiler with references to both domains.
        
        Args:
            sketch: The Sketch instance (CAD domain - geometry, materials)
            simulation: The Simulation instance (Physics domain - particle arrays)
        """
        self.sketch = sketch
        self.sim = simulation

    def rebuild(self, sketch=None):
        """
        Rebuilds the static atoms in the simulation based on the provided sketch.
        
        Args:
            sketch: Optional Sketch to compile. If None, uses self.sketch.
        """
        if sketch is None:
            sketch = self.sketch
            
        # 1. Compact Arrays: Keep dynamic particles, discard old static ones
        is_dynamic = self.sim.is_static[:self.sim.count] == 0
        dyn_indices = np.where(is_dynamic)[0]
        self.sim._compact_arrays(dyn_indices)
        
        # 2. Iterate Geometry
        for w in sketch.entities:
            # Look up material properties
            mat = sketch.materials.get(w.material_id, sketch.materials["Default"])
            
            # Check Physical Flag from Material (Ghost vs Solid)
            if not mat.physical: continue
            
            if isinstance(w, Line):
                self._compile_line(w, mat)
            elif isinstance(w, Circle):
                self._compile_circle(w, mat)
        
        self.sim.rebuild_next = True

    def _compile_line(self, w, mat):
        if w.ref: return
        
        p1 = w.start; p2 = w.end
        spacing = mat.spacing
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-4: return 
        
        # Determine number of atoms
        num_atoms = max(1, int(length / spacing) + 1)
        
        # Check for Kinematics (Drivers/Animation)
        anim = w.anim
        is_rotating = False
        pivot = np.zeros(2)
        omega = 0.0
        
        if anim and anim['type'] == 'rotate':
            is_rotating = True
            omega = math.radians(anim['speed'])
            if anim['pivot'] == 'start': pivot = w.start
            elif anim['pivot'] == 'end': pivot = w.end
            else: pivot = (w.start + w.end) * 0.5

        # Generate Atoms
        for k in range(num_atoms):
            if self.sim.count >= self.sim.capacity: self.sim._resize_arrays()
            
            # Interpolate position
            t = k / max(1, num_atoms - 1) if num_atoms > 1 else 0.5
            pos = p1 + vec * t
            
            self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon), is_rotating, pivot, omega)

    def _compile_circle(self, w, mat):
        circumference = 2 * math.pi * w.radius
        num_atoms = max(3, int(circumference / mat.spacing))
        
        for k in range(num_atoms):
            if self.sim.count >= self.sim.capacity: self.sim._resize_arrays()
            
            angle = (k / num_atoms) * 2 * math.pi
            pos = w.center + np.array([math.cos(angle) * w.radius, math.sin(angle) * w.radius])
            
            self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon), False, np.zeros(2), 0.0)

    def _add_static_atom(self, pos, sig, eps_sqrt, rotating, pivot, omega):
        idx = self.sim.count
        self.sim.pos_x[idx] = pos[0]
        self.sim.pos_y[idx] = pos[1]
        self.sim.vel_x[idx] = 0.0
        self.sim.vel_y[idx] = 0.0
        
        if rotating:
            self.sim.is_static[idx] = 2 # Kinematic
            self.sim.kinematic_props[idx, 0] = pivot[0]
            self.sim.kinematic_props[idx, 1] = pivot[1]
            self.sim.kinematic_props[idx, 2] = omega
        else:
            self.sim.is_static[idx] = 1 # Static
            self.sim.kinematic_props[idx, :] = 0.0
            
        self.sim.atom_sigma[idx] = sig
        self.sim.atom_eps_sqrt[idx] = eps_sqrt
        self.sim.count += 1