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
    integrate_n_steps, integrate_n_steps_newton3,
    build_neighbor_list, check_displacement,
    apply_thermostat, spatial_sort, apply_tether_forces_pbd,
    build_atom_neighbor_csr,
    BOUNDARY_OPEN, BOUNDARY_REFLECTING, BOUNDARY_PERIODIC,
)
from numba import get_num_threads

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
        # Boundary mode: BOUNDARY_OPEN / REFLECTING / PERIODIC. Canonical
        # attribute; the bool `use_boundaries` is a backward-compat property
        # that toggles between OPEN and REFLECTING for the existing UI button.
        self.boundary_mode = BOUNDARY_OPEN
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
        self.atom_sigma = np.zeros(self.capacity, dtype=np.float32)
        self.atom_eps_sqrt = np.zeros(self.capacity, dtype=np.float32)
        self.atom_mass = np.full(self.capacity, config.ATOM_MASS, dtype=np.float32)
        self.atom_color = np.zeros((self.capacity, 3), dtype=np.uint8)  # RGB per particle
        # Stable per-atom material index into Sketch.materials' insertion-order
        # key list. -1 means "no material" — the LJ kernel falls back to
        # per-atom ε_sqrt geometric mean (legacy L-B behaviour). Atoms spawned
        # from Source/brush/compiler with a known material name carry a real
        # index, which lets the kernel look up cross-pair ε in eps_ij_matrix.
        self.atom_material_id = np.full(self.capacity, -1, dtype=np.int32)

        # (M, M) effective LJ ε matrix, indexed by material_id pair. Set by
        # Scene.rebuild from Sketch.build_eps_ij_matrix(). Zero-shape default
        # disables the matrix path — kernel falls back to per-atom ε_sqrt for
        # everyone. Headless tests and the pre-Scene warmup path operate with
        # this default.
        self.eps_ij_matrix = np.zeros((0, 0), dtype=np.float32)

        # --- Tether Arrays (for Dynamic Two-Way Coupling) ---
        # is_static=3 means tethered atom
        # tether_entity_idx: which entity this atom is tethered to (-1 = none)
        # tether_local_pos: [t, 0] for lines (t=0..1), [theta, 0] for circles
        # tether_stiffness: spring constant for tether
        self.tether_entity_idx = np.full(self.capacity, -1, dtype=np.int32)
        self.tether_local_pos = np.zeros((self.capacity, 2), dtype=np.float32)
        self.tether_stiffness = np.zeros(self.capacity, dtype=np.float32)

        # --- Joint Arrays (for Coincident Constraint LJ Exclusion) ---
        # joint_ids: atoms with same non-zero ID skip LJ forces between each other
        # This prevents physics explosions at coincident joints where atoms overlap
        self.joint_ids = np.zeros(self.capacity, dtype=np.int32)

        # --- Bond Arrays (Harmonic Spring Bonds, intramolecular) ---
        # Each bond couples two atoms (bond_i[b], bond_j[b]) via F = -k*(r - r_eq)*r_hat.
        # Bonds are independent of atom arrays — they're indexed b=0..bond_count-1
        # and grown separately. compact_arrays remaps bond indices when atoms
        # are removed; bonds that reference a removed atom are dropped.
        self.bond_capacity = 100
        self.bond_count = 0
        self.bond_i = np.zeros(self.bond_capacity, dtype=np.int32)
        self.bond_j = np.zeros(self.bond_capacity, dtype=np.int32)
        self.bond_k = np.zeros(self.bond_capacity, dtype=np.float32)
        self.bond_r_eq = np.zeros(self.bond_capacity, dtype=np.float32)

        # --- Angle Arrays (Three-Body Angle Springs, intramolecular) ---
        # Each angle binds three atoms (a, b, c) with b as the apex via
        # U(θ) = k*(θ - θ_eq)² where θ is the angle b→a vs b→c.
        # Indexed n=0..angle_count-1; grown separately from atom and bond
        # arrays. compact_arrays remaps angle indices the same way as bonds.
        self.angle_capacity = 100
        self.angle_count = 0
        self.angle_a = np.zeros(self.angle_capacity, dtype=np.int32)
        self.angle_b = np.zeros(self.angle_capacity, dtype=np.int32)
        self.angle_c = np.zeros(self.angle_capacity, dtype=np.int32)
        self.angle_k = np.zeros(self.angle_capacity, dtype=np.float32)
        self.angle_theta_eq = np.zeros(self.angle_capacity, dtype=np.float32)

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
        # Atom-centric CSR view of the neighbour list, derived from the
        # half-pair list each rebuild. Drives the parallel atom-centric
        # LJ pair loop — each atom's neighbours live in
        # nbr_idx[nbr_start[i] : nbr_start[i+1]]. nbr_idx is sized for the
        # full pair list (each pair counted twice).
        self.nbr_start = np.zeros(self.capacity + 1, dtype=np.int32)
        self.nbr_idx = np.zeros(2 * self.max_pairs, dtype=np.int32)
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

        # Force-kernel selector. False = classic atom-centric LJ loop (pairs
        # computed twice for race-free parallelism). True = half-pair Newton-3
        # path with thread-local force accumulators (each pair computed once).
        # The local_force_x/y buffers are lazy-allocated on first use sized to
        # (T, capacity) where T = numba.get_num_threads().
        self.use_newton3 = False
        self._local_force_x = None
        self._local_force_y = None

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

    # ---- Backward-compat boundary toggle ----------------------------------
    # The "Bounds" UI button (and any other binary-state caller) reads/writes
    # `use_boundaries` as a bool. The canonical attribute is `boundary_mode`
    # (int, 0/1/2). This property maps the bool onto OPEN/REFLECTING and
    # treats PERIODIC as truthy on read so a UI showing "boundaries on" stays
    # consistent. To switch into PERIODIC, callers set `boundary_mode`
    # directly or use `cycle_boundary_mode()`.
    @property
    def use_boundaries(self):
        return self.boundary_mode != BOUNDARY_OPEN

    @use_boundaries.setter
    def use_boundaries(self, value):
        if value:
            # Don't downgrade an active PERIODIC mode to REFLECTING when the
            # bool stays True (e.g., the UI button writes True every frame).
            if self.boundary_mode == BOUNDARY_OPEN:
                self.boundary_mode = BOUNDARY_REFLECTING
        else:
            self.boundary_mode = BOUNDARY_OPEN

    def cycle_boundary_mode(self):
        """Cycle OPEN → REFLECTING → PERIODIC → OPEN. Refuses PERIODIC if the
        domain is too small for a PBC cell list (n_cells < 3) and skips that
        mode in the cycle."""
        next_mode = (self.boundary_mode + 1) % 3
        if next_mode == BOUNDARY_PERIODIC and not self._pbc_safe():
            print("PBC unavailable: world_size too small for cell list "
                  "(need world_size >= 3 * cell_size). Skipping PERIODIC.")
            next_mode = BOUNDARY_OPEN
        self.boundary_mode = next_mode
        return self.boundary_mode

    def _pbc_safe(self):
        """True iff the cell list has at least 3 cells along each axis, the
        minimum for non-double-counting cell-neighbour walks under PBC."""
        n_cells = int(self.world_size // self.cell_size) + 1
        return n_cells >= 3

    def _ensure_local_force_buffers(self):
        """Lazy-allocate the (T, capacity) thread-local force buffers required
        by the Newton-3 integrator. Resized when capacity grows (resize_world
        and other expansion paths zero them out via reallocation)."""
        T = get_num_threads()
        cap = self.capacity
        need_alloc = (
            self._local_force_x is None
            or self._local_force_x.shape != (T, cap)
        )
        if need_alloc:
            self._local_force_x = np.zeros((T, cap), dtype=np.float32)
            self._local_force_y = np.zeros((T, cap), dtype=np.float32)

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
        self.atom_mass[:2] = config.ATOM_MASS
        
        build_neighbor_list(
            self.pos_x[:2], self.pos_y[:2], self.r_list2,
            self.cell_size, self.world_size, self.pair_i, self.pair_j,
            np.int32(self.boundary_mode),
        )
        # Build the atom-centric CSR view (input to the parallel LJ loop).
        build_atom_neighbor_csr(
            2, self.pair_i, self.pair_j, self.pair_count,
            self.nbr_start[:3], self.nbr_idx,
        )

        f32_vals = [
            np.float32(x) for x in [
                config.ATOM_MASS, self.dt, self.gravity,
                self.r_cut_base**2, self.r_skin_sq_limit,
                self.world_size, self.damping
            ]
        ]

        # Warmup with empty bond + angle lists — exercises the zero-length
        # loops so the kernel is JIT-compiled for the bondless/angle-less
        # fast path that's typical of brush-only scenes.
        empty_bi = np.zeros(0, dtype=np.int32)
        empty_bj = np.zeros(0, dtype=np.int32)
        empty_bk = np.zeros(0, dtype=np.float32)
        empty_br = np.zeros(0, dtype=np.float32)
        empty_aa = np.zeros(0, dtype=np.int32)
        empty_ab = np.zeros(0, dtype=np.int32)
        empty_ac = np.zeros(0, dtype=np.int32)
        empty_ak = np.zeros(0, dtype=np.float32)
        empty_ate = np.zeros(0, dtype=np.float32)
        # Warmup with an empty eps_ij_matrix (and material_id slice that's all
        # -1 from the np.full default in __init__) so the kernel JITs both the
        # matrix branch and the per-atom ε_sqrt fallback. Pass shape (0, 0).
        empty_eps_matrix = np.zeros((0, 0), dtype=np.float32)
        integrate_n_steps(
            1, self.pos_x[:2], self.pos_y[:2],
            self.vel_x[:2], self.vel_y[:2],
            self.force_x[:2], self.force_y[:2],
            self.last_x[:2], self.last_y[:2],
            self.is_static[:2],
            self.atom_sigma[:2], self.atom_eps_sqrt[:2],
            self.atom_mass[:2],
            self.atom_material_id[:2], empty_eps_matrix,
            self.nbr_start[:3], self.nbr_idx,
            self.tether_entity_idx[:2],  # For intra-entity exclusion
            self.joint_ids[:2],  # For coincident constraint LJ exclusion
            empty_bi, empty_bj, empty_bk, empty_br,
            empty_aa, empty_ab, empty_ac, empty_ak, empty_ate,
            f32_vals[1], f32_vals[2], f32_vals[3], f32_vals[4],
            f32_vals[5], np.int32(self.boundary_mode), f32_vals[6]
        )

        spatial_sort(
            self.pos_x[:2], self.pos_y[:2], self.vel_x[:2], self.vel_y[:2],
            self.force_x[:2], self.force_y[:2], self.is_static[:2],
            self.atom_sigma[:2], self.atom_eps_sqrt[:2], self.atom_mass[:2],
            self.atom_material_id[:2],
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
            'atom_sigma': np.copy(self.atom_sigma[:self.count]),
            'atom_eps_sqrt': np.copy(self.atom_eps_sqrt[:self.count]),
            'atom_mass': np.copy(self.atom_mass[:self.count]),
            'atom_color': np.copy(self.atom_color[:self.count]),
            'atom_material_id': np.copy(self.atom_material_id[:self.count]),
            'world_size': self.world_size,
            'bond_count': self.bond_count,
            'bond_i': np.copy(self.bond_i[:self.bond_count]),
            'bond_j': np.copy(self.bond_j[:self.bond_count]),
            'bond_k': np.copy(self.bond_k[:self.bond_count]),
            'bond_r_eq': np.copy(self.bond_r_eq[:self.bond_count]),
            'angle_count': self.angle_count,
            'angle_a': np.copy(self.angle_a[:self.angle_count]),
            'angle_b': np.copy(self.angle_b[:self.angle_count]),
            'angle_c': np.copy(self.angle_c[:self.angle_count]),
            'angle_k': np.copy(self.angle_k[:self.angle_count]),
            'angle_theta_eq': np.copy(self.angle_theta_eq[:self.angle_count]),
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
        self.atom_sigma[:self.count] = state['atom_sigma']
        self.atom_eps_sqrt[:self.count] = state['atom_eps_sqrt']
        if 'atom_mass' in state:
            self.atom_mass[:self.count] = state['atom_mass']
        if 'atom_color' in state:
            self.atom_color[:self.count] = state['atom_color']
        # Pre-R1 snapshots have no atom_material_id key — fall back to -1
        # so the kernel uses the per-atom ε_sqrt path for restored atoms.
        if 'atom_material_id' in state:
            self.atom_material_id[:self.count] = state['atom_material_id']
        else:
            self.atom_material_id[:self.count] = -1
        self.world_size = state['world_size']
        # Bonds (back-compat: snapshots from before R1 won't have these keys —
        # treat absence as "no bonds in the saved state" rather than failing).
        if 'bond_count' in state:
            new_bond_count = state['bond_count']
            while self.bond_capacity < new_bond_count:
                self._resize_bond_arrays()
            self.bond_count = new_bond_count
            self.bond_i[:new_bond_count] = state['bond_i']
            self.bond_j[:new_bond_count] = state['bond_j']
            self.bond_k[:new_bond_count] = state['bond_k']
            self.bond_r_eq[:new_bond_count] = state['bond_r_eq']
        else:
            self.bond_count = 0
        # Angles (back-compat: pre-angle snapshots fall through to angle_count=0)
        if 'angle_count' in state:
            new_angle_count = state['angle_count']
            while self.angle_capacity < new_angle_count:
                self._resize_angle_arrays()
            self.angle_count = new_angle_count
            self.angle_a[:new_angle_count] = state['angle_a']
            self.angle_b[:new_angle_count] = state['angle_b']
            self.angle_c[:new_angle_count] = state['angle_c']
            self.angle_k[:new_angle_count] = state['angle_k']
            self.angle_theta_eq[:new_angle_count] = state['angle_theta_eq']
        else:
            self.angle_count = 0
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
            'atom_sigma': np.copy(self.atom_sigma[:self.count]),
            'atom_eps_sqrt': np.copy(self.atom_eps_sqrt[:self.count]),
            'atom_mass': np.copy(self.atom_mass[:self.count]),
            'atom_color': np.copy(self.atom_color[:self.count]),
            'atom_material_id': np.copy(self.atom_material_id[:self.count]),
            'world_size': self.world_size,
            'bond_count': self.bond_count,
            'bond_i': np.copy(self.bond_i[:self.bond_count]),
            'bond_j': np.copy(self.bond_j[:self.bond_count]),
            'bond_k': np.copy(self.bond_k[:self.bond_count]),
            'bond_r_eq': np.copy(self.bond_r_eq[:self.bond_count]),
            'angle_count': self.angle_count,
            'angle_a': np.copy(self.angle_a[:self.angle_count]),
            'angle_b': np.copy(self.angle_b[:self.angle_count]),
            'angle_c': np.copy(self.angle_c[:self.angle_count]),
            'angle_k': np.copy(self.angle_k[:self.angle_count]),
            'angle_theta_eq': np.copy(self.angle_theta_eq[:self.angle_count]),
        }
        stack.append(state)

    # =========================================================================
    # World Management
    # =========================================================================

    def resize_world(self, new_size):
        """Resize the simulation world. Performs a full physics reset.

        All particles are cleared and physics settings (gravity, dt,
        damping, sigma, epsilon, skin_distance, target_temp) return to
        defaults. The new world_size overrides the default set by reset().

        Note: reset() snapshots internally (see line 292), so no outer
        snapshot() call is needed here — that would push two undo states
        per resize and require two Ctrl+Z presses to revert.

        Callers (see AppController.action_resize_world) are responsible
        for triggering scene.rebuild() afterwards to regenerate the
        Compiler-emitted static and tethered atoms that reset() wipes.
        """
        if new_size < 10.0:
            new_size = 10.0
        self.reset()                  # full physics reset; reset() snapshots internally
        self.world_size = new_size    # override the default set by reset()

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
        # Bonds and angles reference atom indices that are now gone — drop them all.
        self.bond_count = 0
        self.angle_count = 0
        self.rebuild_next = True

    def reset(self):
        """Reset physics to defaults."""
        self.snapshot()
        self.world_size = config.DEFAULT_WORLD_SIZE
        self.dt = config.DEFAULT_DT
        self.gravity = config.DEFAULT_GRAVITY
        self.target_temp = 0.5
        self.damping = config.DEFAULT_DAMPING
        self.boundary_mode = BOUNDARY_OPEN
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
            'atom_sigma': self.atom_sigma[:self.count].tolist(),
            'atom_eps_sqrt': self.atom_eps_sqrt[:self.count].tolist(),
            'atom_mass': self.atom_mass[:self.count].tolist(),
            'atom_color': self.atom_color[:self.count].tolist(),
            'atom_material_id': self.atom_material_id[:self.count].tolist(),
            'bond_count': int(self.bond_count),
            'bond_i': self.bond_i[:self.bond_count].tolist(),
            'bond_j': self.bond_j[:self.bond_count].tolist(),
            'bond_k': self.bond_k[:self.bond_count].tolist(),
            'bond_r_eq': self.bond_r_eq[:self.bond_count].tolist(),
            'angle_count': int(self.angle_count),
            'angle_a': self.angle_a[:self.angle_count].tolist(),
            'angle_b': self.angle_b[:self.angle_count].tolist(),
            'angle_c': self.angle_c[:self.angle_count].tolist(),
            'angle_k': self.angle_k[:self.angle_count].tolist(),
            'angle_theta_eq': self.angle_theta_eq[:self.angle_count].tolist(),
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
        if 'atom_sigma' in data:
            self.atom_sigma[:self.count] = np.array(data['atom_sigma'], dtype=np.float32)
        if 'atom_eps_sqrt' in data:
            self.atom_eps_sqrt[:self.count] = np.array(data['atom_eps_sqrt'], dtype=np.float32)
        if 'atom_mass' in data:
            self.atom_mass[:self.count] = np.array(data['atom_mass'], dtype=np.float32)
        # Guard against the count==0 case: np.array([], dtype=uint8) has shape (0,)
        # which doesn't broadcast into atom_color[:0] of shape (0, 3).
        if 'atom_color' in data and self.count > 0:
            self.atom_color[:self.count] = np.array(data['atom_color'], dtype=np.uint8)
        # Pre-R1 saves predate atom_material_id — default to -1 so restored
        # atoms use the per-atom ε_sqrt fallback path in the LJ kernel.
        if 'atom_material_id' in data:
            self.atom_material_id[:self.count] = np.array(data['atom_material_id'], dtype=np.int32)
        else:
            self.atom_material_id[:self.count] = -1

        # Bonds (back-compat: pre-R1 saves won't carry them — default to none).
        new_bond_count = data.get('bond_count', 0)
        while self.bond_capacity < new_bond_count:
            self._resize_bond_arrays()
        self.bond_count = new_bond_count
        if new_bond_count > 0:
            self.bond_i[:new_bond_count] = np.array(data['bond_i'], dtype=np.int32)
            self.bond_j[:new_bond_count] = np.array(data['bond_j'], dtype=np.int32)
            self.bond_k[:new_bond_count] = np.array(data['bond_k'], dtype=np.float32)
            self.bond_r_eq[:new_bond_count] = np.array(data['bond_r_eq'], dtype=np.float32)

        # Angles (back-compat: pre-R5-angle saves get angle_count=0)
        new_angle_count = data.get('angle_count', 0)
        while self.angle_capacity < new_angle_count:
            self._resize_angle_arrays()
        self.angle_count = new_angle_count
        if new_angle_count > 0:
            self.angle_a[:new_angle_count] = np.array(data['angle_a'], dtype=np.int32)
            self.angle_b[:new_angle_count] = np.array(data['angle_b'], dtype=np.int32)
            self.angle_c[:new_angle_count] = np.array(data['angle_c'], dtype=np.int32)
            self.angle_k[:new_angle_count] = np.array(data['angle_k'], dtype=np.float32)
            self.angle_theta_eq[:new_angle_count] = np.array(data['angle_theta_eq'], dtype=np.float32)

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

        Bonds are remapped via an old→new index table built from
        keep_indices: any bond touching a removed atom is dropped (swap-with-
        last). Bonds where both endpoints survive have bond_i and bond_j
        rewritten to their new positions. This preserves the bond between
        atoms even when atoms get reordered by compaction.
        """
        indices = np.array(keep_indices, dtype=np.int32)
        new_count = len(indices)

        # Build old→new index remap BEFORE overwriting the atom arrays.
        # remap[old_idx] = new_idx, or -1 if the atom was removed.
        remap = np.full(self.count, -1, dtype=np.int32)
        for new_idx, old_idx in enumerate(indices):
            remap[old_idx] = new_idx

        self.pos_x[:new_count] = self.pos_x[indices]
        self.pos_y[:new_count] = self.pos_y[indices]
        self.vel_x[:new_count] = self.vel_x[indices]
        self.vel_y[:new_count] = self.vel_y[indices]
        self.is_static[:new_count] = self.is_static[indices]
        self.atom_sigma[:new_count] = self.atom_sigma[indices]
        self.atom_eps_sqrt[:new_count] = self.atom_eps_sqrt[indices]
        self.atom_mass[:new_count] = self.atom_mass[indices]
        self.atom_color[:new_count] = self.atom_color[indices]
        self.atom_material_id[:new_count] = self.atom_material_id[indices]

        # Tether arrays
        self.tether_entity_idx[:new_count] = self.tether_entity_idx[indices]
        self.tether_local_pos[:new_count] = self.tether_local_pos[indices]
        self.tether_stiffness[:new_count] = self.tether_stiffness[indices]

        # Joint arrays
        self.joint_ids[:new_count] = self.joint_ids[indices]

        self.count = new_count

        # Remap or drop bonds. Iterate from the end so swap-with-last is safe.
        b = self.bond_count - 1
        while b >= 0:
            new_i = remap[self.bond_i[b]]
            new_j = remap[self.bond_j[b]]
            if new_i < 0 or new_j < 0:
                self.remove_bond(b)
            else:
                self.bond_i[b] = new_i
                self.bond_j[b] = new_j
            b -= 1

        # Remap or drop angles — same iterate-from-end / swap-with-last pattern.
        n = self.angle_count - 1
        while n >= 0:
            new_a = remap[self.angle_a[n]]
            new_b = remap[self.angle_b[n]]
            new_c = remap[self.angle_c[n]]
            if new_a < 0 or new_b < 0 or new_c < 0:
                self.remove_angle(n)
            else:
                self.angle_a[n] = new_a
                self.angle_b[n] = new_b
                self.angle_c[n] = new_c
            n -= 1

    # =========================================================================
    # Low-Level Particle Primitives (Used by ParticleBrush, Compiler, Sources)
    # =========================================================================

    def set_eps_ij_matrix(self, matrix):
        """Replace the effective LJ ε matrix used by the kernel for atoms
        with valid material_id. Accepts a (M, M) numpy array (will coerce
        to float32) or None / empty → matrix-path disabled, kernel falls
        back to per-atom ε_sqrt for everyone. Called by Scene.rebuild after
        building the matrix from Sketch.materials + Sketch.lj_cross_overrides.
        """
        if matrix is None:
            self.eps_ij_matrix = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(matrix, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"eps_ij_matrix must be square 2D; got shape {arr.shape}"
                )
            self.eps_ij_matrix = arr

    def _add_particle(self, x, y, vx=0.0, vy=0.0, is_static=0, sigma=None, epsilon=None,
                      mass=None, color=(50, 150, 255), material_id=-1):
        """
        Add a single particle to the simulation.

        This is a low-level primitive used by ParticleBrush, Compiler, and Sources.
        For brush operations, use ParticleBrush.paint() instead.

        Args:
            x, y: Position
            vx, vy: Velocity (default 0)
            is_static: 0=dynamic, 1=static, 3=tethered
            sigma: Particle size (default: self.sigma)
            epsilon: LJ energy parameter (default: self.epsilon)
            mass: Particle mass (default: config.ATOM_MASS). Per-particle so
                mixed-material scenes (Mercury vs Water) integrate with the
                correct inertia per atom.
            color: RGB tuple for the atom's render color. Defaults to the
                project's water-blue. The slot's previous color is otherwise
                retained — Source emissions used to inherit residue from
                whatever atom (often a wall) had previously occupied the
                slot, which surfaced as wall-coloured "free" atoms.

        Returns:
            Index of the new particle, or -1 if failed
        """
        if self.count >= self.capacity:
            self._resize_arrays()

        if sigma is None:
            sigma = self.sigma
        if epsilon is None:
            epsilon = self.epsilon
        if mass is None:
            mass = config.ATOM_MASS

        idx = self.count
        self.pos_x[idx] = x
        self.pos_y[idx] = y
        self.vel_x[idx] = vx
        self.vel_y[idx] = vy
        self.is_static[idx] = is_static
        self.atom_sigma[idx] = sigma
        self.atom_eps_sqrt[idx] = math.sqrt(epsilon)
        self.atom_mass[idx] = mass
        self.atom_color[idx] = color
        self.atom_material_id[idx] = material_id
        self.count += 1
        self.rebuild_next = True

        return idx

    def add_bond(self, i, j, k, r_eq):
        """Add a harmonic spring bond between atoms i and j.

        Args:
            i, j: Atom indices into pos_x/pos_y/... (must be < self.count).
            k: Spring stiffness.
            r_eq: Equilibrium bond length.

        Returns:
            Index of the new bond (0..bond_count-1 after insertion), or -1
            if i == j (degenerate). Self-bonds are silently rejected because
            the force computation would divide by zero on identical positions
            and the result is meaningless physically.
        """
        if i == j:
            return -1
        if self.bond_count >= self.bond_capacity:
            self._resize_bond_arrays()
        b = self.bond_count
        self.bond_i[b] = i
        self.bond_j[b] = j
        self.bond_k[b] = k
        self.bond_r_eq[b] = r_eq
        self.bond_count += 1
        return b

    def remove_bond(self, b):
        """Remove the bond at index b via swap-with-last."""
        if b < 0 or b >= self.bond_count:
            return
        last = self.bond_count - 1
        if b != last:
            self.bond_i[b] = self.bond_i[last]
            self.bond_j[b] = self.bond_j[last]
            self.bond_k[b] = self.bond_k[last]
            self.bond_r_eq[b] = self.bond_r_eq[last]
        self.bond_count -= 1

    def clear_bonds(self):
        """Drop all bonds (used by clear/reset)."""
        self.bond_count = 0

    def _resize_bond_arrays(self):
        """Double the capacity of bond arrays. Preserves existing data."""
        self.bond_capacity *= 2
        self.bond_i = np.resize(self.bond_i, self.bond_capacity)
        self.bond_j = np.resize(self.bond_j, self.bond_capacity)
        self.bond_k = np.resize(self.bond_k, self.bond_capacity)
        self.bond_r_eq = np.resize(self.bond_r_eq, self.bond_capacity)

    def add_angle(self, a, b, c, k, theta_eq):
        """Add a three-body angle spring with apex b between legs a and c.

        Args:
            a, b, c: Atom indices (b is the apex; the angle is between
                vectors b→a and b→c). All three must be distinct.
            k: Angular stiffness.
            theta_eq: Equilibrium angle in radians (0 to π).

        Returns:
            Index of the new angle, or -1 if the triplet is degenerate
            (any two indices equal).
        """
        if a == b or b == c or a == c:
            return -1
        if self.angle_count >= self.angle_capacity:
            self._resize_angle_arrays()
        n = self.angle_count
        self.angle_a[n] = a
        self.angle_b[n] = b
        self.angle_c[n] = c
        self.angle_k[n] = k
        self.angle_theta_eq[n] = theta_eq
        self.angle_count += 1
        return n

    def remove_angle(self, n):
        """Remove the angle at index n via swap-with-last."""
        if n < 0 or n >= self.angle_count:
            return
        last = self.angle_count - 1
        if n != last:
            self.angle_a[n] = self.angle_a[last]
            self.angle_b[n] = self.angle_b[last]
            self.angle_c[n] = self.angle_c[last]
            self.angle_k[n] = self.angle_k[last]
            self.angle_theta_eq[n] = self.angle_theta_eq[last]
        self.angle_count -= 1

    def clear_angles(self):
        """Drop all angles (used by clear/reset)."""
        self.angle_count = 0

    def _resize_angle_arrays(self):
        """Double the capacity of angle arrays. Preserves existing data."""
        self.angle_capacity *= 2
        self.angle_a = np.resize(self.angle_a, self.angle_capacity)
        self.angle_b = np.resize(self.angle_b, self.angle_capacity)
        self.angle_c = np.resize(self.angle_c, self.angle_capacity)
        self.angle_k = np.resize(self.angle_k, self.angle_capacity)
        self.angle_theta_eq = np.resize(self.angle_theta_eq, self.angle_capacity)

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
            mode_arg = np.int32(self.boundary_mode)
            while True:
                count = build_neighbor_list(
                    self.pos_x[:self.count], self.pos_y[:self.count],
                    self.r_list2, self.cell_size, self.world_size,
                    self.pair_i, self.pair_j,
                    mode_arg,
                )
                if count >= self.max_pairs:
                    self.max_pairs *= 2
                    self.pair_i = np.zeros(self.max_pairs, dtype=np.int32)
                    self.pair_j = np.zeros(self.max_pairs, dtype=np.int32)
                    self.nbr_idx = np.zeros(2 * self.max_pairs, dtype=np.int32)
                    continue
                self.pair_count = count
                break

            self.last_x[:self.count] = self.pos_x[:self.count]
            self.last_y[:self.count] = self.pos_y[:self.count]
            self.rebuild_next = False

            # Convert the half-pair list to atom-centric CSR. Cheap
            # (O(N + pair_count)) and only runs when the neighbour list
            # is rebuilt — same cadence as the half-pair list itself.
            build_atom_neighbor_csr(
                self.count, self.pair_i, self.pair_j, self.pair_count,
                self.nbr_start[:self.count + 1], self.nbr_idx,
            )

        # Run integration
        if self.count > 0:
            # Bond / angle views: kernels infer count from .shape[0], so
            # empty lists skip cleanly inside the kernel.
            b = self.bond_count
            a = self.angle_count
            if self.use_newton3:
                self._ensure_local_force_buffers()
                steps_done = integrate_n_steps_newton3(
                    steps_to_run,
                    self.pos_x[:self.count], self.pos_y[:self.count],
                    self.vel_x[:self.count], self.vel_y[:self.count],
                    self.force_x[:self.count], self.force_y[:self.count],
                    self.last_x[:self.count], self.last_y[:self.count],
                    self.is_static[:self.count],
                    self.atom_sigma[:self.count], self.atom_eps_sqrt[:self.count],
                    self.atom_mass[:self.count],
                    self.atom_material_id[:self.count], self.eps_ij_matrix,
                    self.pair_i, self.pair_j, self.pair_count,
                    self.tether_entity_idx[:self.count],
                    self.joint_ids[:self.count],
                    self.bond_i[:b], self.bond_j[:b],
                    self.bond_k[:b], self.bond_r_eq[:b],
                    self.angle_a[:a], self.angle_b[:a], self.angle_c[:a],
                    self.angle_k[:a], self.angle_theta_eq[:a],
                    np.float32(self.dt), np.float32(self.gravity),
                    np.float32(self.r_cut_base**2), np.float32(self.r_skin_sq_limit),
                    np.float32(self.world_size), np.int32(self.boundary_mode),
                    np.float32(self.damping),
                    self._local_force_x[:, :self.count],
                    self._local_force_y[:, :self.count],
                )
            else:
                steps_done = integrate_n_steps(
                    steps_to_run,
                    self.pos_x[:self.count], self.pos_y[:self.count],
                    self.vel_x[:self.count], self.vel_y[:self.count],
                    self.force_x[:self.count], self.force_y[:self.count],
                    self.last_x[:self.count], self.last_y[:self.count],
                    self.is_static[:self.count],
                    self.atom_sigma[:self.count], self.atom_eps_sqrt[:self.count],
                    self.atom_mass[:self.count],
                    self.atom_material_id[:self.count], self.eps_ij_matrix,
                    self.nbr_start[:self.count + 1], self.nbr_idx,
                    self.tether_entity_idx[:self.count],  # For intra-entity exclusion
                    self.joint_ids[:self.count],  # For coincident constraint LJ exclusion
                    self.bond_i[:b], self.bond_j[:b],
                    self.bond_k[:b], self.bond_r_eq[:b],
                    self.angle_a[:a], self.angle_b[:a], self.angle_c[:a],
                    self.angle_k[:a], self.angle_theta_eq[:a],
                    np.float32(self.dt), np.float32(self.gravity),
                    np.float32(self.r_cut_base**2), np.float32(self.r_skin_sq_limit),
                    np.float32(self.world_size), np.int32(self.boundary_mode),
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
                    self.atom_mass[:self.count], self.is_static[:self.count],
                    np.float32(self.target_temp), np.float32(0.1)
                )
            
            # Remove particles that escaped the world. Only dynamic atoms
            # (is_static==0) are subject to this filter — static (1) and
            # tethered (3) atoms must be retained because their positions are
            # managed by the Compiler and sync_static_atoms_to_geometry path,
            # and removing them would orphan tether linkage indices.
            #
            # Under PERIODIC boundaries dynamic atoms wrap inside the kernel
            # and never legitimately fall outside [0, world_size]. The escape
            # filter is skipped (any straggler outside the box from a borderline
            # float landed mid-step is not a leak — the next substep wraps it).
            if self.boundary_mode != BOUNDARY_PERIODIC:
                active_x = self.pos_x[:self.count]
                active_y = self.pos_y[:self.count]
                active_static = self.is_static[:self.count]
                w = self.world_size
                is_inside = (active_x >= 0) & (active_x <= w) & (active_y >= 0) & (active_y <= w)
                keep = is_inside | (active_static != 0)

                if not np.all(keep):
                    keep_indices = np.where(keep)[0]
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
        self.atom_sigma = np.resize(self.atom_sigma, self.capacity)
        self.atom_eps_sqrt = np.resize(self.atom_eps_sqrt, self.capacity)
        # New slots inherit config.ATOM_MASS so unset entries don't divide
        # by zero in the integrator.
        old_mass = self.atom_mass
        self.atom_mass = np.full(self.capacity, config.ATOM_MASS, dtype=np.float32)
        self.atom_mass[:len(old_mass)] = old_mass
        old_color = self.atom_color
        self.atom_color = np.zeros((self.capacity, 3), dtype=np.uint8)
        self.atom_color[:len(old_color)] = old_color
        # Per-atom material_id grows alongside the rest; new slots default to
        # -1 (no material → per-atom ε_sqrt fallback in the LJ kernel).
        old_material_id = self.atom_material_id
        self.atom_material_id = np.full(self.capacity, -1, dtype=np.int32)
        self.atom_material_id[:len(old_material_id)] = old_material_id
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

        # Joint arrays
        old_joint_ids = self.joint_ids
        self.joint_ids = np.zeros(self.capacity, dtype=np.int32)
        self.joint_ids[:len(old_joint_ids)] = old_joint_ids

        # Atom-centric CSR neighbour view: nbr_start indexes by atom so it
        # must grow with capacity. nbr_idx scales with max_pairs and is
        # grown there (in step() during the build_neighbor_list overflow loop).
        self.nbr_start = np.zeros(self.capacity + 1, dtype=np.int32)
