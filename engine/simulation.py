"""
Simulation - The Physics Domain (PURE)

This class is responsible for PARTICLE PHYSICS ONLY:
- Particle arrays (position, velocity, force)
- Physics parameters (gravity, dt, damping)
- Neighbor lists and spatial optimization
- Force calculation and integration

It does NOT know about:
- Lines, Circles, Points (geometry types)
- Constraints (CAD relationships)
- Materials (physical properties of geometry)
- Compilation (that's the Compiler's job)
- Orchestration (that's the Scene's job)

Those belong to the Sketch (CAD domain). The Compiler bridges them.
The Scene orchestrates all operations.

NOTE: Brush operations (paint/erase particles) are now handled by 
ParticleBrush in engine/particle_brush.py. The deprecated 
add_particles_brush() and delete_particles_brush() methods have been removed.
"""

import numpy as np
import math
import time
import core.config as config

from engine.physics_core import (
    integrate_n_steps, build_neighbor_list, check_displacement, 
    apply_thermostat, spatial_sort
)


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
        
        self.rebuild_next = True
        self.pair_count = 0

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
        
        self.count = new_count

    # =========================================================================
    # Low-Level Particle Primitives (Used by ParticleBrush and Compiler)
    # =========================================================================

    def _add_particle(self, x, y, vx=0.0, vy=0.0, is_static=0, sigma=None, epsilon=None):
        """
        Add a single particle to the simulation.
        
        This is a low-level primitive used by ParticleBrush and Compiler.
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
            threshold: Minimum distance to consider overlap
            
        Returns:
            True if position overlaps, False otherwise
        """
        if self.count == 0:
            return False
        
        threshold_sq = threshold * threshold
        dx = self.pos_x[:self.count] - x
        dy = self.pos_y[:self.count] - y
        dist_sq = dx*dx + dy*dy
        
        return np.any(dist_sq < threshold_sq)

    # =========================================================================
    # Physics Step (PURE PHYSICS)
    # =========================================================================

    def step(self, steps_to_run):
        """
        Run physics simulation steps.
        
        This method is PURE PHYSICS:
        - Updates particle positions and velocities
        - Calculates forces
        - Handles collisions and boundaries
        
        It does NOT:
        - Update geometry (that's Sketch's job)
        - Update constraints (that's Sketch's job)
        - Compile geometry (that's Compiler's job)
        - Animate anything (that's done via constraint drivers before step())
        """
        if self.paused:
            return
        
        # Update dynamic particle properties from UI sliders
        is_dyn = self.is_static[:self.count] == 0
        if np.any(is_dyn):
            self.atom_sigma[:self.count][is_dyn] = self.sigma
            self.atom_eps_sqrt[:self.count][is_dyn] = math.sqrt(self.epsilon)
        
        self._update_derived_params()

        # Periodic spatial sort for cache efficiency
        if self.total_steps % 100 == 0 and self.count > 0:
            spatial_sort(
                self.pos_x[:self.count], self.pos_y[:self.count],
                self.vel_x[:self.count], self.vel_y[:self.count],
                self.force_x[:self.count], self.force_y[:self.count],
                self.is_static[:self.count], self.kinematic_props[:self.count],
                self.atom_sigma[:self.count], self.atom_eps_sqrt[:self.count],
                self.world_size, self.cell_size
            )
            self.rebuild_next = True
        
        # Check if neighbor list needs rebuilding
        should_rebuild = False
        if self.pair_count == 0 or self.rebuild_next:
            should_rebuild = True
        elif self.count > 0:
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
        self.last_x = np.resize(self.last_x, self.capacity)
        self.last_y = np.resize(self.last_y, self.capacity)
