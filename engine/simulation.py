"""
Simulation - Pure Physics Domain

The Simulation owns the particle arrays and physics calculations.
It has NO knowledge of CAD geometry - it only knows about atoms.
The Scene orchestrates all operations.

NOTE: Brush operations (paint/erase particles) are now handled by 
ParticleBrush in engine/particle_brush.py. The deprecated 
add_particles_brush() and delete_particles_brush() methods have been removed.

ProcessObject Sources use has_particle_near() for rejection sampling
during particle spawning.
"""

import numpy as np
import math
import time
import core.config as config

from engine.physics_core import (
    integrate_n_steps, build_neighbor_list, check_displacement,
    apply_thermostat, spatial_sort, apply_tether_forces_pbd
)

# Entity type constants (must match physics kernel expectations)
ENTITY_TYPE_LINE = 0
ENTITY_TYPE_CIRCLE = 1
ENTITY_TYPE_POINT = 2


class Simulation:
    def __init__(self, skip_warmup=False):
        """
        Initialize the physics simulation.
        
        Args:
            skip_warmup: If True, skip Numba JIT warmup (for Editor-only mode)
        """
        self.capacity = 5000
        self.count = 0
        
        # --- Physics Parameters ---
        self.world_size = config.DEFAULT_WORLD_SIZE
        self.use_boundaries = False
        self.sigma = config.ATOM_SIGMA
        self.epsilon = config.ATOM_EPSILON
        self.skin_distance = config.DEFAULT_SKIN_DISTANCE
        
        # --- Particle Arrays ---
        self.pos_x = np.zeros(self.capacity, dtype=np.float32)
        self.pos_y = np.zeros(self.capacity, dtype=np.float32)
        self.vel_x = np.zeros(self.capacity, dtype=np.float32)
        self.vel_y = np.zeros(self.capacity, dtype=np.float32)
        self.force_x = np.zeros(self.capacity, dtype=np.float32)
        self.force_y = np.zeros(self.capacity, dtype=np.float32)
        self.is_static = np.zeros(self.capacity, dtype=np.int32)
        self.kinematic_props = np.zeros((self.capacity, 3), dtype=np.float32)
        self.atom_sigma = np.zeros(self.capacity, dtype=np.float32)
        self.atom_eps_sqrt = np.zeros(self.capacity, dtype=np.float32)
        self.atom_color = np.zeros((self.capacity, 3), dtype=np.uint8)  # RGB per particle

        # --- Tether Arrays (for Dynamic Two-Way Coupling) ---
        # is_static=3 means tethered atom
        # tether_entity_idx: which entity this atom is tethered to (-1 = none)
        # tether_local_pos: [t, 0] for lines (t=0..1), [theta, 0] for circles
        # tether_stiffness: spring constant for tether
        self.tether_entity_idx = np.full(self.capacity, -1, dtype=np.int32)
        self.tether_local_pos = np.zeros((self.capacity, 2), dtype=np.float32)
        self.tether_stiffness = np.zeros(self.capacity, dtype=np.float32)

        # --- Entity State Arrays (for physics kernel to read/write) ---
        # These are synced from Sketch entities before physics runs
        # entity_positions: [N, 4] - Line: [start_x, start_y, end_x, end_y]
        #                           Circle: [center_x, center_y, radius, 0]
        # entity_forces: [N, 3] - [fx, fy, torque] accumulated from tethers
        # entity_com: [N, 2] - center of mass [x, y]
        # entity_types: [N] - 0=Line, 1=Circle, 2=Point
        self.max_entities = 256
        self.entity_positions = np.zeros((self.max_entities, 4), dtype=np.float32)
        self.entity_forces = np.zeros((self.max_entities, 3), dtype=np.float32)
        self.entity_com = np.zeros((self.max_entities, 2), dtype=np.float32)
        self.entity_types = np.zeros(self.max_entities, dtype=np.int32)
        self.entity_count = 0

        # --- Neighbor List & Optimization ---
        self.max_pairs = self.capacity * 100
        self.pair_i = np.zeros(self.max_pairs, dtype=np.int32)
        self.pair_j = np.zeros(self.max_pairs, dtype=np.int32)
        self.pair_count = 0
        self.last_x = np.zeros(self.capacity, dtype=np.float32)
        self.last_y = np.zeros(self.capacity, dtype=np.float32)
        self.rebuild_next = False
        
        # --- Simulation State ---
        self.paused = True
        self.dt = config.DEFAULT_DT
        self.gravity = config.DEFAULT_GRAVITY
        self.target_temp = 0.5
        self.use_thermostat = False
        self.damping = config.DEFAULT_DAMPING
        
        self.r_cut_base = 2.5
        self._update_derived_params()
        
        # --- Metrics ---
        self.total_steps = 0
        self.sps = 0.0
        self.steps_accumulator = 0
        self.last_sps_update = time.time()
        
        # --- Physics Undo Stack (particles only, not CAD) ---
        self.undo_stack = []
        self.redo_stack = []
        
        if not skip_warmup:
            self._warmup_compiler()
        else:
            print("Skipping Numba warmup (Model Builder Mode)")

    # =========================================================================
    # Physics Parameter Management
    # =========================================================================

    def _update_derived_params(self):
        """Update derived physics parameters from base values."""
        self.r_list = self.r_cut_base + self.skin_distance
        self.r_list2 = self.r_list**2
        self.r_skin_sq_limit = (0.5 * self.skin_distance)**2
        self.cell_size = self.r_list

    def _warmup_compiler(self):
        """Pre-compile Numba functions with dummy data."""
        print("Warming up Numba compiler...")
        self.pos_x[0] = 10.0
        self.pos_y[0] = 10.0
        self.pos_x[1] = 12.0
        self.pos_y[1] = 10.0
        self.count = 2
        self.atom_sigma[:2] = 1.0
        self.atom_eps_sqrt[:2] = 1.0
        
        build_neighbor_list(
            self.pos_x[:2], self.pos_y[:2], self.r_list2, 
            self.cell_size, self.world_size, self.pair_i, self.pair_j
        )
        
        f32_vals = [
            np.float32(x) for x in [
                config.ATOM_MASS, self.dt, self.gravity, 
                self.r_cut_base**2, self.r_skin_sq_limit, 
                self.world_size, self.damping
            ]
        ]
        
        integrate_n_steps(
            1, self.pos_x[:2], self.pos_y[:2],
            self.vel_x[:2], self.vel_y[:2],
            self.force_x[:2], self.force_y[:2],
            self.last_x[:2], self.last_y[:2],
            self.is_static[:2], self.kinematic_props[:2],
            self.atom_sigma[:2], self.atom_eps_sqrt[:2],
            f32_vals[0], self.pair_i, self.pair_j, self.pair_count,
            self.tether_entity_idx[:2],  # For intra-entity exclusion
            f32_vals[1], f32_vals[2], f32_vals[3], f32_vals[4],
            f32_vals[5], self.use_boundaries, f32_vals[6]
        )
        
        spatial_sort(
            self.pos_x[:2], self.pos_y[:2], self.vel_x[:2], self.vel_y[:2],
            self.force_x[:2], self.force_y[:2], self.is_static[:2],
            self.kinematic_props[:2], self.atom_sigma[:2], self.atom_eps_sqrt[:2],
            self.world_size, self.cell_size
        )
        
        self.clear()
        print("Warmup complete.")

    # =========================================================================
    # Physics Undo/Redo (Particle State Only)
    # 
    # Note: CAD operations use the Command Queue in Scene.
    # This undo system is for particle brush operations only.
    # =========================================================================

    def snapshot(self):
        """Save current particle state for undo (physics only, not CAD)."""
        state = {
            'count': self.count,
            'pos_x': np.copy(self.pos_x[:self.count]),
            'pos_y': np.copy(self.pos_y[:self.count]),
            'vel_x': np.copy(self.vel_x[:self.count]),
            'vel_y': np.copy(self.vel_y[:self.count]),
            'is_static': np.copy(self.is_static[:self.count]),
            'kinematic_props': np.copy(self.kinematic_props[:self.count]),
            'atom_sigma': np.copy(self.atom_sigma[:self.count]),
            'atom_eps_sqrt': np.copy(self.atom_eps_sqrt[:self.count]),
            'atom_color': np.copy(self.atom_color[:self.count]),
            'world_size': self.world_size
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore_physics_state(self, state):
        """Restore particle physics state from a saved state."""
        self.count = state['count']
        if self.count > self.capacity:
            while self.capacity < self.count:
                self.capacity *= 2
            self._resize_arrays()
        
        self.pos_x[:self.count] = state['pos_x']
        self.pos_y[:self.count] = state['pos_y']
        self.vel_x[:self.count] = state['vel_x']
        self.vel_y[:self.count] = state['vel_y']
        self.is_static[:self.count] = state['is_static']
        self.kinematic_props[:self.count] = state['kinematic_props']
        self.atom_sigma[:self.count] = state['atom_sigma']
        self.atom_eps_sqrt[:self.count] = state['atom_eps_sqrt']
        if 'atom_color' in state:
            self.atom_color[:self.count] = state['atom_color']
        self.world_size = state['world_size']
        self.rebuild_next = True
        self.pair_count = 0

    def undo(self):
        """Undo last particle operation."""
        if not self.undo_stack:
            return False
        self._push_to_stack(self.redo_stack)
        prev_state = self.undo_stack.pop()
        self._restore_physics_state(prev_state)
        return True

    def redo(self):
        """Redo last undone particle operation."""
        if not self.redo_stack:
            return False
        self._push_to_stack(self.undo_stack)
        next_state = self.redo_stack.pop()
        self._restore_physics_state(next_state)
        return True
        
    def _push_to_stack(self, stack):
        """Save current state to a stack."""
        state = {
            'count': self.count,
            'pos_x': np.copy(self.pos_x[:self.count]),
            'pos_y': np.copy(self.pos_y[:self.count]),
            'vel_x': np.copy(self.vel_x[:self.count]),
            'vel_y': np.copy(self.vel_y[:self.count]),
            'is_static': np.copy(self.is_static[:self.count]),
            'kinematic_props': np.copy(self.kinematic_props[:self.count]),
            'atom_sigma': np.copy(self.atom_sigma[:self.count]),
            'atom_eps_sqrt': np.copy(self.atom_eps_sqrt[:self.count]),
            'atom_color': np.copy(self.atom_color[:self.count]),
            'world_size': self.world_size
        }
        stack.append(state)

    # =========================================================================
    # World Management
    # =========================================================================

    def resize_world(self, new_size):
        """Resize the simulation world."""
        self.snapshot()
        if new_size < 10.0:
            new_size = 10.0
        self.world_size = new_size
        self.clear(snapshot=False)

    def clear(self, snapshot=True):
        """Remove all particles."""
        if snapshot:
            self.snapshot()
        self.count = 0
        self.pair_count = 0
        self.pos_x.fill(0)
        self.pos_y.fill(0)
        self.vel_x.fill(0)
        self.vel_y.fill(0)
        self.is_static.fill(0)
        self.kinematic_props.fill(0)
        self.rebuild_next = True

    def reset(self):
        """Reset physics to defaults."""
        self.snapshot()
        self.world_size = config.DEFAULT_WORLD_SIZE
        self.dt = config.DEFAULT_DT
        self.gravity = config.DEFAULT_GRAVITY
        self.target_temp = 0.5
        self.damping = config.DEFAULT_DAMPING
        self.use_boundaries = False
        self.sigma = config.ATOM_SIGMA
        self.epsilon = config.ATOM_EPSILON
        self.skin_distance = config.DEFAULT_SKIN_DISTANCE
        self._update_derived_params()
        self.clear(snapshot=False)

    # =========================================================================
    # Serialization (Physics State Only)
    # =========================================================================
    
    def to_dict(self):
        """Serialize physics state."""
        return {
            'count': int(self.count),
            'world_size': float(self.world_size),
            'pos_x': self.pos_x[:self.count].tolist(),
            'pos_y': self.pos_y[:self.count].tolist(),
            'vel_x': self.vel_x[:self.count].tolist(),
            'vel_y': self.vel_y[:self.count].tolist(),
            'is_static': self.is_static[:self.count].tolist(),
            'kinematic_props': self.kinematic_props[:self.count].tolist(),
            'atom_sigma': self.atom_sigma[:self.count].tolist(),
            'atom_eps_sqrt': self.atom_eps_sqrt[:self.count].tolist(),
            'atom_color': self.atom_color[:self.count].tolist(),
        }

    def restore(self, data):
        """Restore physics state from dict."""
        self.count = data.get('count', 0)
        self.world_size = data.get('world_size', config.DEFAULT_WORLD_SIZE)
        
        if self.count > self.capacity:
            while self.capacity < self.count:
                self.capacity *= 2
            self._resize_arrays()
        
        if 'pos_x' in data:
            self.pos_x[:self.count] = np.array(data['pos_x'], dtype=np.float32)
        if 'pos_y' in data:
            self.pos_y[:self.count] = np.array(data['pos_y'], dtype=np.float32)
        if 'vel_x' in data:
            self.vel_x[:self.count] = np.array(data['vel_x'], dtype=np.float32)
        if 'vel_y' in data:
            self.vel_y[:self.count] = np.array(data['vel_y'], dtype=np.float32)
        if 'is_static' in data:
            self.is_static[:self.count] = np.array(data['is_static'], dtype=np.int32)
        if 'kinematic_props' in data:
            self.kinematic_props[:self.count] = np.array(data['kinematic_props'], dtype=np.float32)
        if 'atom_sigma' in data:
            self.atom_sigma[:self.count] = np.array(data['atom_sigma'], dtype=np.float32)
        if 'atom_eps_sqrt' in data:
            self.atom_eps_sqrt[:self.count] = np.array(data['atom_eps_sqrt'], dtype=np.float32)
        if 'atom_color' in data:
            self.atom_color[:self.count] = np.array(data['atom_color'], dtype=np.uint8)

        self.rebuild_next = True
        self.pair_count = 0

    # =========================================================================
    # Entity Sync Interface (Fast Path for Geometry Motion)
    # =========================================================================

    def sync_entity_arrays(self, entities):
        """
        Update entity position arrays WITHOUT rebuilding atoms.

        This is the FAST PATH for geometry motion. Called when entities move
        but topology is unchanged (no add/delete). Updates anchor positions
        for tether force calculations.

        Does NOT modify:
        - Atom positions (pos_x, pos_y)
        - Atom velocities (vel_x, vel_y)
        - Atom count or tether relationships

        Args:
            entities: List of Entity objects from Sketch
        """
        from model.geometry import Line, Circle, Point

        # Resize entity arrays if needed
        if len(entities) > self.max_entities:
            self.max_entities = len(entities) * 2
            self.entity_positions = np.zeros((self.max_entities, 4), dtype=np.float32)
            self.entity_forces = np.zeros((self.max_entities, 3), dtype=np.float32)
            self.entity_com = np.zeros((self.max_entities, 2), dtype=np.float32)
            self.entity_types = np.zeros(self.max_entities, dtype=np.int32)

        self.entity_count = len(entities)

        for i, entity in enumerate(entities):
            if isinstance(entity, Line):
                self.entity_types[i] = ENTITY_TYPE_LINE
                self.entity_positions[i, 0] = entity.start[0]
                self.entity_positions[i, 1] = entity.start[1]
                self.entity_positions[i, 2] = entity.end[0]
                self.entity_positions[i, 3] = entity.end[1]
                com = entity.get_center_of_mass()
                self.entity_com[i, 0] = com[0]
                self.entity_com[i, 1] = com[1]
            elif isinstance(entity, Circle):
                self.entity_types[i] = ENTITY_TYPE_CIRCLE
                self.entity_positions[i, 0] = entity.center[0]
                self.entity_positions[i, 1] = entity.center[1]
                self.entity_positions[i, 2] = entity.radius
                self.entity_positions[i, 3] = 0.0  # Unused
                self.entity_com[i, 0] = entity.center[0]
                self.entity_com[i, 1] = entity.center[1]
            elif isinstance(entity, Point):
                self.entity_types[i] = ENTITY_TYPE_POINT
                self.entity_positions[i, 0] = entity.pos[0]
                self.entity_positions[i, 1] = entity.pos[1]
                self.entity_positions[i, 2] = 0.0
                self.entity_positions[i, 3] = 0.0
                self.entity_com[i, 0] = entity.pos[0]
                self.entity_com[i, 1] = entity.pos[1]

    def clear_entity_forces(self):
        """Zero out entity force accumulators before physics step."""
        self.entity_forces[:self.entity_count] = 0

    def get_entity_forces(self):
        """
        Retrieve accumulated forces and torques for all entities.

        Returns:
            numpy array of shape (entity_count, 3) containing [fx, fy, torque]
            for each entity. This is a view into the internal array.
        """
        return self.entity_forces[:self.entity_count]

    def apply_tether_forces(self):
        """
        Apply tether spring forces between atoms and entities.

        This is the core of two-way coupling:
        - Tethered atoms feel forces pulling them toward their anchors
        - Entities accumulate reaction forces from their tethered atoms

        Must call clear_entity_forces() before this to reset accumulators.
        Must call sync_entity_arrays() before this to update anchor positions.
        """
        if self.count == 0 or self.entity_count == 0:
            return

        apply_tether_forces_pbd(
            self.pos_x[:self.count],
            self.pos_y[:self.count],
            self.force_x[:self.count],
            self.force_y[:self.count],
            self.is_static[:self.count],
            self.tether_entity_idx[:self.count],
            self.tether_local_pos[:self.count],
            self.tether_stiffness[:self.count],
            self.entity_positions[:self.entity_count],
            self.entity_forces[:self.entity_count],
            self.entity_com[:self.entity_count],
            self.entity_types[:self.entity_count]
        )

    def sync_static_atoms_to_geometry(self):
        """
        Teleport static atoms to match their parent entity's current position.

        This fixes the "atoms left behind" bug: when a user drags a physical
        but static entity, the atoms need to move with it instantly.

        Only affects static atoms (is_static == 1) that have a valid
        tether_entity_idx (>= 0). Recalculates world position from the
        entity's current geometry using local coordinates.

        Called by Scene.update() when geometry has moved but topology is unchanged.
        """
        if self.count == 0 or self.entity_count == 0:
            return

        for i in range(self.count):
            # Only process static atoms with valid entity linkage
            if self.is_static[i] != 1:
                continue

            ent_idx = self.tether_entity_idx[i]
            if ent_idx < 0 or ent_idx >= self.entity_count:
                continue

            local_t = self.tether_local_pos[i, 0]
            ent_type = self.entity_types[ent_idx]

            if ent_type == ENTITY_TYPE_LINE:
                # Line: lerp between start and end using t
                start_x = self.entity_positions[ent_idx, 0]
                start_y = self.entity_positions[ent_idx, 1]
                end_x = self.entity_positions[ent_idx, 2]
                end_y = self.entity_positions[ent_idx, 3]
                self.pos_x[i] = start_x + local_t * (end_x - start_x)
                self.pos_y[i] = start_y + local_t * (end_y - start_y)

            elif ent_type == ENTITY_TYPE_CIRCLE:
                # Circle: center + radius * (cos(theta), sin(theta))
                center_x = self.entity_positions[ent_idx, 0]
                center_y = self.entity_positions[ent_idx, 1]
                radius = self.entity_positions[ent_idx, 2]
                theta = local_t  # For circles, local_t stores the angle
                self.pos_x[i] = center_x + radius * math.cos(theta)
                self.pos_y[i] = center_y + radius * math.sin(theta)

            elif ent_type == ENTITY_TYPE_POINT:
                # Point: anchor is the point itself
                self.pos_x[i] = self.entity_positions[ent_idx, 0]
                self.pos_y[i] = self.entity_positions[ent_idx, 1]

        # Mark neighbor list for rebuild since atoms moved
        self.rebuild_next = True

    def snap_tethered_atoms_to_anchors(self):
        """
        Snap tethered atoms to their anchor positions and zero velocities.

        This ensures a "cold start" with zero initial energy when an entity
        becomes dynamic. Without this, floating-point precision differences
        between compile-time position calculation and runtime anchor calculation
        can create small initial displacements that grow into oscillations.

        Only affects tethered atoms (is_static == 3) with valid entity linkage.
        Called after rebuild() when topology changes.
        """
        if self.count == 0 or self.entity_count == 0:
            return

        for i in range(self.count):
            # Only process tethered atoms with valid entity linkage
            if self.is_static[i] != 3:
                continue

            ent_idx = self.tether_entity_idx[i]
            if ent_idx < 0 or ent_idx >= self.entity_count:
                continue

            local_t = self.tether_local_pos[i, 0]
            ent_type = self.entity_types[ent_idx]

            # Calculate anchor position using SAME math as tether kernel
            if ent_type == ENTITY_TYPE_LINE:
                start_x = self.entity_positions[ent_idx, 0]
                start_y = self.entity_positions[ent_idx, 1]
                end_x = self.entity_positions[ent_idx, 2]
                end_y = self.entity_positions[ent_idx, 3]
                self.pos_x[i] = start_x + local_t * (end_x - start_x)
                self.pos_y[i] = start_y + local_t * (end_y - start_y)

            elif ent_type == ENTITY_TYPE_CIRCLE:
                center_x = self.entity_positions[ent_idx, 0]
                center_y = self.entity_positions[ent_idx, 1]
                radius = self.entity_positions[ent_idx, 2]
                theta = local_t
                self.pos_x[i] = center_x + radius * math.cos(theta)
                self.pos_y[i] = center_y + radius * math.sin(theta)

            elif ent_type == ENTITY_TYPE_POINT:
                self.pos_x[i] = self.entity_positions[ent_idx, 0]
                self.pos_y[i] = self.entity_positions[ent_idx, 1]

            # Zero velocities for cold start
            self.vel_x[i] = 0.0
            self.vel_y[i] = 0.0

            # Zero forces to prevent any residual acceleration
            self.force_x[i] = 0.0
            self.force_y[i] = 0.0

        self.rebuild_next = True

    # =========================================================================
    # Compiler Interface (Called by Scene/Compiler)
    # =========================================================================

    def compact_arrays(self, keep_indices):
        """
        Compact particle arrays to remove gaps.
        Called by Compiler during rebuild.
        """
        indices = np.array(keep_indices, dtype=np.int32)
        new_count = len(indices)

        self.pos_x[:new_count] = self.pos_x[indices]
        self.pos_y[:new_count] = self.pos_y[indices]
        self.vel_x[:new_count] = self.vel_x[indices]
        self.vel_y[:new_count] = self.vel_y[indices]
        self.is_static[:new_count] = self.is_static[indices]
        self.kinematic_props[:new_count] = self.kinematic_props[indices]
        self.atom_sigma[:new_count] = self.atom_sigma[indices]
        self.atom_eps_sqrt[:new_count] = self.atom_eps_sqrt[indices]
        self.atom_color[:new_count] = self.atom_color[indices]

        # Tether arrays
        self.tether_entity_idx[:new_count] = self.tether_entity_idx[indices]
        self.tether_local_pos[:new_count] = self.tether_local_pos[indices]
        self.tether_stiffness[:new_count] = self.tether_stiffness[indices]

        self.count = new_count

    # =========================================================================
    # Low-Level Particle Primitives (Used by ParticleBrush, Compiler, Sources)
    # =========================================================================

    def _add_particle(self, x, y, vx=0.0, vy=0.0, is_static=0, sigma=None, epsilon=None):
        """
        Add a single particle to the simulation.
        
        This is a low-level primitive used by ParticleBrush, Compiler, and Sources.
        For brush operations, use ParticleBrush.paint() instead.
        
        Args:
            x, y: Position
            vx, vy: Velocity (default 0)
            is_static: 0=dynamic, 1=static, 2=kinematic
            sigma: Particle size (default: self.sigma)
            epsilon: LJ energy parameter (default: self.epsilon)
            
        Returns:
            Index of the new particle, or -1 if failed
        """
        if self.count >= self.capacity:
            self._resize_arrays()
        
        if sigma is None:
            sigma = self.sigma
        if epsilon is None:
            epsilon = self.epsilon
        
        idx = self.count
        self.pos_x[idx] = x
        self.pos_y[idx] = y
        self.vel_x[idx] = vx
        self.vel_y[idx] = vy
        self.is_static[idx] = is_static
        self.atom_sigma[idx] = sigma
        self.atom_eps_sqrt[idx] = math.sqrt(epsilon)
        self.count += 1
        self.rebuild_next = True
        
        return idx

    def _check_overlap(self, x, y, threshold):
        """
        Check if a position overlaps existing particles.
        
        This is a low-level primitive used by ParticleBrush.
        
        Args:
            x, y: Position to check
            threshold: Distance threshold for overlap
            
        Returns:
            True if position overlaps, False otherwise
        """
        threshold_sq = threshold * threshold
        for i in range(self.count):
            dx = self.pos_x[i] - x
            dy = self.pos_y[i] - y
            if dx * dx + dy * dy < threshold_sq:
                return True
        return False

    def has_particle_near(self, x, y, threshold):
        """
        Check if any particle exists within threshold distance of (x, y).
        
        This is the public interface for overlap detection, used by
        ProcessObject Sources for rejection sampling during particle spawning.
        
        Args:
            x, y: Position to check (world coordinates)
            threshold: Distance threshold (particles closer than this = overlap)
            
        Returns:
            True if any particle is within threshold distance
        """
        return self._check_overlap(x, y, threshold)

    # =========================================================================
    # Physics Step
    # =========================================================================

    def step(self, steps_to_run=1):
        """
        Run physics integration steps.
        
        Args:
            steps_to_run: Number of integration sub-steps to run
        """
        # Check for spatial displacement exceeding threshold
        should_rebuild = self.rebuild_next
        if not should_rebuild and self.count > 0:
            should_rebuild = check_displacement(
                self.pos_x[:self.count], self.pos_y[:self.count],
                self.last_x[:self.count], self.last_y[:self.count],
                self.r_skin_sq_limit
            )
        
        # Rebuild neighbor list if needed
        if should_rebuild and self.count > 0:
            while True:
                count = build_neighbor_list(
                    self.pos_x[:self.count], self.pos_y[:self.count],
                    self.r_list2, self.cell_size, self.world_size,
                    self.pair_i, self.pair_j
                )
                if count >= self.max_pairs:
                    self.max_pairs *= 2
                    self.pair_i = np.zeros(self.max_pairs, dtype=np.int32)
                    self.pair_j = np.zeros(self.max_pairs, dtype=np.int32)
                    continue
                self.pair_count = count
                break
            
            self.last_x[:self.count] = self.pos_x[:self.count]
            self.last_y[:self.count] = self.pos_y[:self.count]
            self.rebuild_next = False
        
        # Run integration
        if self.count > 0:
            steps_done = integrate_n_steps(
                steps_to_run,
                self.pos_x[:self.count], self.pos_y[:self.count],
                self.vel_x[:self.count], self.vel_y[:self.count],
                self.force_x[:self.count], self.force_y[:self.count],
                self.last_x[:self.count], self.last_y[:self.count],
                self.is_static[:self.count], self.kinematic_props[:self.count],
                self.atom_sigma[:self.count], self.atom_eps_sqrt[:self.count],
                np.float32(config.ATOM_MASS),
                self.pair_i, self.pair_j, self.pair_count,
                self.tether_entity_idx[:self.count],  # For intra-entity exclusion
                np.float32(self.dt), np.float32(self.gravity),
                np.float32(self.r_cut_base**2), np.float32(self.r_skin_sq_limit),
                np.float32(self.world_size), self.use_boundaries,
                np.float32(self.damping)
            )
            
            self.total_steps += steps_done
            self.steps_accumulator += steps_done
            
            # Update SPS metric
            now = time.time()
            elapsed = now - self.last_sps_update
            if elapsed >= 0.5:
                self.sps = self.steps_accumulator / elapsed
                self.steps_accumulator = 0
                self.last_sps_update = now
            
            if steps_done < steps_to_run:
                self.rebuild_next = True
            
            # Apply thermostat if enabled
            if self.use_thermostat:
                apply_thermostat(
                    self.vel_x[:self.count], self.vel_y[:self.count],
                    np.float32(config.ATOM_MASS), self.is_static[:self.count],
                    np.float32(self.target_temp), np.float32(0.1)
                )
            
            # Remove particles that escaped the world
            active_x = self.pos_x[:self.count]
            active_y = self.pos_y[:self.count]
            w = self.world_size
            is_inside = (active_x >= 0) & (active_x <= w) & (active_y >= 0) & (active_y <= w)
            
            if not np.all(is_inside):
                keep_indices = np.where(is_inside)[0]
                self.compact_arrays(keep_indices)
                self.rebuild_next = True

    # =========================================================================
    # Array Management
    # =========================================================================

    def _resize_arrays(self):
        """Double the capacity of all particle arrays."""
        self.capacity *= 2
        print(f"Resizing simulation capacity to {self.capacity}")

        self.pos_x = np.resize(self.pos_x, self.capacity)
        self.pos_y = np.resize(self.pos_y, self.capacity)
        self.vel_x = np.resize(self.vel_x, self.capacity)
        self.vel_y = np.resize(self.vel_y, self.capacity)
        self.force_x = np.resize(self.force_x, self.capacity)
        self.force_y = np.resize(self.force_y, self.capacity)
        self.is_static = np.resize(self.is_static, self.capacity)
        self.kinematic_props = np.resize(self.kinematic_props, (self.capacity, 3))
        self.atom_sigma = np.resize(self.atom_sigma, self.capacity)
        self.atom_eps_sqrt = np.resize(self.atom_eps_sqrt, self.capacity)
        old_color = self.atom_color
        self.atom_color = np.zeros((self.capacity, 3), dtype=np.uint8)
        self.atom_color[:len(old_color)] = old_color
        self.last_x = np.resize(self.last_x, self.capacity)
        self.last_y = np.resize(self.last_y, self.capacity)

        # Tether arrays
        old_tether_idx = self.tether_entity_idx
        self.tether_entity_idx = np.full(self.capacity, -1, dtype=np.int32)
        self.tether_entity_idx[:len(old_tether_idx)] = old_tether_idx

        old_tether_local = self.tether_local_pos
        self.tether_local_pos = np.zeros((self.capacity, 2), dtype=np.float32)
        self.tether_local_pos[:len(old_tether_local)] = old_tether_local

        self.tether_stiffness = np.resize(self.tether_stiffness, self.capacity)
