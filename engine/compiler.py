"""
Compiler - The Bridge between Sketch (CAD) and Simulation (Physics)

Reads geometry and materials from Sketch, writes static particle arrays
to Simulation. This is the ONLY component that knows about both domains.

Data flow is ONE-WAY: Sketch → Simulation (never reverse)

Handle System:
- Points with is_handle=True are ProcessObject handles
- These are SKIPPED during atomization (they don't become particles)
- They exist only for constraint participation
"""

import numpy as np
import math
from model.geometry import Line, Circle, Point


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
        Rebuilds the static atoms in the simulation based on the sketch.
        
        This is the core compilation step: CAD geometry → Physics particles.
        
        Handle Points (is_handle=True) are SKIPPED - they exist only for
        constraint participation, not physics.
        
        Args:
            sketch: Optional Sketch to compile. If None, uses self.sketch.
        """
        if sketch is None:
            sketch = self.sketch
            
        # 1. Compact Arrays: Keep dynamic particles, discard old static ones
        is_dynamic = self.sim.is_static[:self.sim.count] == 0
        dyn_indices = np.where(is_dynamic)[0]
        self.sim.compact_arrays(dyn_indices)
        
        # 2. Iterate Geometry
        for w in sketch.entities:
            # Skip ProcessObject handles - they don't become atoms
            if isinstance(w, Point) and getattr(w, 'is_handle', False):
                continue

            # Skip entities that are not marked as physical
            if not getattr(w, 'physical', False):
                continue

            # Look up material properties
            mat = sketch.materials.get(w.material_id, sketch.materials["Default"])

            # Check Physical Flag from Material (non-physical materials are not atomized)
            if not mat.physical:
                continue

            if isinstance(w, Line):
                self._compile_line(w, mat)
            elif isinstance(w, Circle):
                self._compile_circle(w, mat)
            # Note: Regular Point entities could be atomized here if needed
            # Currently they are not (same as before)
        
        self.sim.rebuild_next = True

    def _compile_line(self, w, mat):
        """Compile a Line entity into static atoms."""
        # Skip reference lines
        if w.ref:
            return
        
        p1 = w.start
        p2 = w.end
        spacing = mat.spacing
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-4:
            return 
        
        # Determine number of atoms
        num_atoms = max(1, int(length / spacing) + 1)
        
        # Check for Kinematics (Drivers/Animation)
        anim = w.anim
        is_rotating = False
        pivot = np.zeros(2)
        omega = 0.0
        
        if anim and anim.get('type') == 'rotate':
            is_rotating = True
            omega = math.radians(anim['speed'])
            if anim['pivot'] == 'start':
                pivot = w.start
            elif anim['pivot'] == 'end':
                pivot = w.end
            else:
                pivot = (w.start + w.end) * 0.5

        # Generate Atoms
        for k in range(num_atoms):
            if self.sim.count >= self.sim.capacity:
                self.sim._resize_arrays()
            
            # Interpolate position
            t = k / max(1, num_atoms - 1) if num_atoms > 1 else 0.5
            pos = p1 + vec * t
            
            self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon), is_rotating, pivot, omega)

    def _compile_circle(self, w, mat):
        """Compile a Circle entity into static atoms."""
        circumference = 2 * math.pi * w.radius
        num_atoms = max(3, int(circumference / mat.spacing))
        
        for k in range(num_atoms):
            if self.sim.count >= self.sim.capacity:
                self.sim._resize_arrays()
            
            angle = (k / num_atoms) * 2 * math.pi
            pos = w.center + np.array([math.cos(angle) * w.radius, math.sin(angle) * w.radius])
            
            self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon), False, np.zeros(2), 0.0)

    def _add_static_atom(self, pos, sig, eps_sqrt, rotating, pivot, omega):
        """Add a single static or kinematic atom to the simulation."""
        idx = self.sim.count
        self.sim.pos_x[idx] = pos[0]
        self.sim.pos_y[idx] = pos[1]
        self.sim.vel_x[idx] = 0.0
        self.sim.vel_y[idx] = 0.0
        
        if rotating:
            self.sim.is_static[idx] = 2  # Kinematic
            self.sim.kinematic_props[idx, 0] = pivot[0]
            self.sim.kinematic_props[idx, 1] = pivot[1]
            self.sim.kinematic_props[idx, 2] = omega
        else:
            self.sim.is_static[idx] = 1  # Static
            self.sim.kinematic_props[idx, :] = 0.0
            
        self.sim.atom_sigma[idx] = sig
        self.sim.atom_eps_sqrt[idx] = eps_sqrt
        self.sim.count += 1
