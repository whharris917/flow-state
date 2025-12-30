"""
Compiler - The Bridge between Sketch (CAD) and Simulation (Physics)

Reads geometry and materials from Sketch, writes static particle arrays
to Simulation. This is the ONLY component that knows about both domains.

Data flow is ONE-WAY: Sketch → Simulation (never reverse)

Particle Types (is_static):
- 0: Dynamic particle (free-moving fluid)
- 1: Static particle (immovable wall)
- 2: Kinematic particle (scripted motion)
- 3: Tethered particle (bound to geometry via spring)

Handle System:
- Points with is_handle=True are ProcessObject handles
- These are SKIPPED during atomization (they don't become particles)
- They exist only for constraint participation

Tether System (Two-Way Coupling):
- When entity.dynamic=True, atoms are tethered instead of static
- Tethered atoms have local coordinates on their parent entity
- Spring forces connect atoms to their anchor points on the entity
"""

import numpy as np
import math
from model.geometry import Line, Circle, Point
import core.config as config

# Import constants from config (no more magic numbers here)


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
        Rebuilds the static/tethered atoms in the simulation based on the sketch.

        This is the core compilation step: CAD geometry → Physics particles.

        Handle Points (is_handle=True) are SKIPPED - they exist only for
        constraint participation, not physics.

        Atom types created:
        - Static (is_static=1): For non-dynamic entities
        - Kinematic (is_static=2): For animated entities
        - Tethered (is_static=3): For dynamic entities (two-way coupling)

        Args:
            sketch: Optional Sketch to compile. If None, uses self.sketch.
        """
        if sketch is None:
            sketch = self.sketch

        # 1. Compact Arrays: Keep fluid particles (is_static=0), discard others
        is_fluid = self.sim.is_static[:self.sim.count] == 0
        fluid_indices = np.where(is_fluid)[0]
        self.sim.compact_arrays(fluid_indices)

        # Clear joint_ids for geometry atoms (will be reassigned below)
        # Fluid particles (is_static=0) already have joint_ids=0 from compaction
        self.sim.joint_ids[:self.sim.count] = 0

        # Vertex-to-atom mapping: {(entity_idx, pt_idx): atom_index}
        # Used to assign joint IDs for coincident constraints
        self._vertex_to_atom = {}

        # 2. Iterate Geometry with entity index tracking
        for entity_idx, w in enumerate(sketch.entities):
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
                self._compile_line(entity_idx, w, mat)
            elif isinstance(w, Circle):
                self._compile_circle(entity_idx, w, mat)
            # Note: Regular Point entities could be atomized here if needed

        # 3. Assign joint IDs for coincident constraints
        self._assign_joint_ids(sketch)

        # 4. Sync entity arrays for physics kernel
        self.sim.sync_entity_arrays(sketch.entities)

        self.sim.rebuild_next = True

    def _compile_line(self, entity_idx, w, mat):
        """Compile a Line entity into static or tethered atoms."""
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

        # Auto-calculate mass for dynamic entities
        # Entity must be heavier than its atoms to prevent "tail wagging the dog"
        if w.dynamic:
            total_atom_mass = num_atoms * config.ATOM_MASS
            min_entity_mass = total_atom_mass * config.ENTITY_MASS_MULTIPLIER
            w.mass = max(w.mass, min_entity_mass)

        # Get tether stiffness from material or use default
        tether_k = getattr(mat, 'tether_stiffness', config.DEFAULT_TETHER_STIFFNESS)

        # Check for Kinematics (Drivers/Animation) - only for non-dynamic
        anim = w.anim
        is_rotating = False
        pivot = np.zeros(2)
        omega = 0.0

        if not w.dynamic and anim and anim.get('type') == 'rotate':
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

            # Interpolate position - t is the local coordinate
            t = k / max(1, num_atoms - 1) if num_atoms > 1 else 0.5
            pos = p1 + vec * t

            # Record atom index before adding (for vertex-to-atom mapping)
            atom_idx = self.sim.count

            if w.dynamic:
                # Tethered atom for two-way coupling
                self._add_tethered_atom(pos, entity_idx, t, tether_k,
                                        mat.sigma, math.sqrt(mat.epsilon))
            else:
                # Static or kinematic atom (with tether data for position sync)
                self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon),
                                      is_rotating, pivot, omega,
                                      entity_idx=entity_idx, local_t=t)

            # Track start (pt_idx=0) and end (pt_idx=1) atoms for coincident joints
            if k == 0:
                self._vertex_to_atom[(entity_idx, 0)] = atom_idx
            if k == num_atoms - 1:
                self._vertex_to_atom[(entity_idx, 1)] = atom_idx

    def _compile_circle(self, entity_idx, w, mat):
        """Compile a Circle entity into static or tethered atoms."""
        circumference = 2 * math.pi * w.radius
        num_atoms = max(3, int(circumference / mat.spacing))

        # Auto-calculate mass for dynamic entities
        # Entity must be heavier than its atoms to prevent "tail wagging the dog"
        if w.dynamic:
            total_atom_mass = num_atoms * config.ATOM_MASS
            min_entity_mass = total_atom_mass * config.ENTITY_MASS_MULTIPLIER
            w.mass = max(w.mass, min_entity_mass)

        # Get tether stiffness from material or use default
        tether_k = getattr(mat, 'tether_stiffness', config.DEFAULT_TETHER_STIFFNESS)

        for k in range(num_atoms):
            if self.sim.count >= self.sim.capacity:
                self.sim._resize_arrays()

            # Angle is the local coordinate for circles
            angle = (k / num_atoms) * 2 * math.pi
            pos = w.center + np.array([math.cos(angle) * w.radius,
                                       math.sin(angle) * w.radius])

            if w.dynamic:
                # Tethered atom - local_pos stores angle
                self._add_tethered_atom(pos, entity_idx, angle, tether_k,
                                        mat.sigma, math.sqrt(mat.epsilon))
            else:
                # Static atom (with tether data for position sync)
                self._add_static_atom(pos, mat.sigma, math.sqrt(mat.epsilon),
                                      False, np.zeros(2), 0.0,
                                      entity_idx=entity_idx, local_t=angle)

    def _add_tethered_atom(self, pos, entity_idx, local_t, stiffness, sig, eps_sqrt):
        """
        Add a tethered atom for two-way coupling.

        Tethered atoms (is_static=3) are bound to geometry via spring forces.
        They can drift from their anchor point, generating restoring forces
        that affect both the atom and the parent entity.

        Args:
            pos: Initial world position [x, y]
            entity_idx: Index of parent entity in sketch.entities
            local_t: Local coordinate on entity (t for lines, angle for circles)
            stiffness: Spring constant k for tether
            sig: LJ sigma parameter
            eps_sqrt: Square root of LJ epsilon
        """
        idx = self.sim.count
        self.sim.pos_x[idx] = pos[0]
        self.sim.pos_y[idx] = pos[1]
        self.sim.vel_x[idx] = 0.0
        self.sim.vel_y[idx] = 0.0
        self.sim.is_static[idx] = 3  # Tethered type
        self.sim.atom_sigma[idx] = sig
        self.sim.atom_eps_sqrt[idx] = eps_sqrt

        # Tether data
        self.sim.tether_entity_idx[idx] = entity_idx
        self.sim.tether_local_pos[idx, 0] = local_t
        self.sim.tether_local_pos[idx, 1] = 0.0  # Reserved for future use
        self.sim.tether_stiffness[idx] = stiffness

        # Clear kinematic props (not used for tethered)
        self.sim.kinematic_props[idx, :] = 0.0

        self.sim.count += 1

    def _add_static_atom(self, pos, sig, eps_sqrt, rotating, pivot, omega,
                         entity_idx=-1, local_t=0.0):
        """
        Add a single static or kinematic atom to the simulation.

        Args:
            pos: World position [x, y]
            sig: LJ sigma parameter
            eps_sqrt: Square root of LJ epsilon
            rotating: If True, this is a kinematic (rotating) atom
            pivot: Rotation pivot point [x, y] (only used if rotating)
            omega: Angular velocity (only used if rotating)
            entity_idx: Index of parent entity (-1 if none, used for static sync)
            local_t: Local coordinate on entity (t for lines, angle for circles)
        """
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

        # Record tether data for static atom sync (used to teleport atoms when geometry moves)
        self.sim.tether_entity_idx[idx] = entity_idx
        self.sim.tether_local_pos[idx, 0] = local_t
        self.sim.tether_local_pos[idx, 1] = 0.0
        self.sim.tether_stiffness[idx] = 0.0  # Not used for static atoms

        self.sim.count += 1

    def _assign_joint_ids(self, sketch):
        """
        Assign shared joint IDs to atoms at coincident constraint vertices.

        Atoms with the same non-zero joint_id will skip LJ force calculations
        between each other, preventing physics explosions at joints where
        two entities share a vertex.

        Args:
            sketch: The Sketch containing constraints to process
        """
        next_joint_id = 1  # Start at 1, since 0 means "no joint"

        for constraint in sketch.constraints:
            if constraint.type != 'COINCIDENT':
                continue

            # Coincident constraint indices: [(entity1, pt1), (entity2, pt2)]
            if len(constraint.indices) != 2:
                continue

            idx1 = constraint.indices[0]  # (entity_idx, pt_idx)
            idx2 = constraint.indices[1]  # (entity_idx, pt_idx)

            # Look up atom indices for both vertices
            atom1 = self._vertex_to_atom.get(tuple(idx1))
            atom2 = self._vertex_to_atom.get(tuple(idx2))

            # Only assign if both atoms exist (both entities were atomized)
            if atom1 is not None and atom2 is not None:
                self.sim.joint_ids[atom1] = next_joint_id
                self.sim.joint_ids[atom2] = next_joint_id
                next_joint_id += 1
