import math
import numpy as np
from numba import njit, prange, get_thread_id, get_num_threads
import core.config as config

# Module-level constants for numba JIT functions (evaluated at import time)
LJ_HARD_CORE_FRACTION = config.LJ_HARD_CORE_FRACTION
SUBSTEP_SAFETY_CHECK_FREQ = config.SUBSTEP_SAFETY_CHECK_FREQ
TETHER_DAMPING = config.TETHER_DAMPING
MAX_TETHER_FORCE = config.MAX_TETHER_FORCE

# Boundary modes — passed as int to JIT kernels.
# OPEN: dynamic atoms can leave [0, world_size]^2 (escape filter trims them in step()).
# REFLECTING: dynamic + tethered atoms reflect off the world walls with wall_damping.
# PERIODIC: dynamic atoms wrap; pair distances use minimum-image; cell list wraps.
#           Requires n_cells >= 3 along each axis (i.e. world_size >= 3 * cell_size)
#           to avoid the cell-list double-counting same pair via wrap and direct
#           neighbour. Tethered atoms (is_static==3) do NOT wrap under PERIODIC —
#           their positions are driven by spring forces toward in-domain anchors.
BOUNDARY_OPEN = 0
BOUNDARY_REFLECTING = 1
BOUNDARY_PERIODIC = 2

@njit(fastmath=True)
def force_LJ_mixed(r2, s_ij2, e_24):
    """
    LJ Force for mixed particle types.
    s_ij2: (sigma_i + sigma_j)/2 squared
    e_24: 24 * sqrt(eps_i * eps_j)
    """
    # 1. Clamp r2 to avoid singularity.
    # Use a fraction of s_ij2 as the hard core limit.
    min_r2 = LJ_HARD_CORE_FRACTION * s_ij2
    eff_r2 = r2 if r2 > min_r2 else min_r2

    inv_r2 = 1.0 / eff_r2
    s2_inv_r2 = s_ij2 * inv_r2
    inv_r6 = s2_inv_r2 * s2_inv_r2 * s2_inv_r2
    inv_r12 = inv_r6 * inv_r6
    
    return e_24 * (2.0 * inv_r12 - inv_r6) * inv_r2

@njit(fastmath=True, parallel=True)
def check_displacement(pos_x, pos_y, last_x, last_y, limit_sq):
    """
    Parallelized displacement check using max reduction.
    """
    N = pos_x.shape[0]
    max_d2 = 0.0
    
    # prange automatically handles the max reduction for max_d2
    for i in prange(N):
        dx = pos_x[i] - last_x[i]
        dy = pos_y[i] - last_y[i]
        d2 = dx*dx + dy*dy
        max_d2 = max(max_d2, d2)
        
    return max_d2 > limit_sq

@njit(fastmath=True, parallel=True)
def build_neighbor_list(pos_x, pos_y, r_list2, cell_size, world_size, pair_i, pair_j, boundary_mode):
    """Cell-list neighbour search, parallelised across atoms.

    The previous implementation used a linked-list cell layout (head + next_idx)
    that couldn't be parallelised because per-cell head writes raced. This
    version replaces the linked list with a counting-sort CSR cell layout
    (cell_start + cell_atoms) so each cell's atoms live in a contiguous range
    and parallel readers don't conflict.

    Pair generation is also two-pass to avoid an atomic counter on pair_i/j:
      1. Each atom counts its own pairs (j > i within r_list).
      2. Prefix-sum the per-atom counts to get write offsets.
      3. Each atom writes its pairs to its dedicated segment.
    Each pair condition is evaluated TWICE per pair location (once during
    count, once during fill) — same pattern as build_atom_neighbor_csr's
    consumer (integrate_n_steps' atom-centric LJ loop). Net win comes from
    full data-parallelism and no atomic adds.

    Returns the total number of pairs found. If the total exceeds
    pair_i.shape[0] no pairs are written and the caller is expected to
    resize and retry — same contract as the previous serial version.

    boundary_mode: BOUNDARY_OPEN/REFLECTING (0/1) — neighbour cells are clipped
    at the domain edge. BOUNDARY_PERIODIC (2) — neighbour cells wrap and pair
    distances use the minimum-image convention. Caller must ensure
    n_cells >= 3 along each axis when using PERIODIC, otherwise a single
    neighbour cell can be reached from two (dx, dy) offsets and pairs will
    be double-counted.
    """
    N = pos_x.shape[0]
    n_cells = int(world_size // cell_size) + 1
    n_cells2 = n_cells * n_cells
    inv_cell = np.float32(1.0 / cell_size)
    max_pairs = pair_i.shape[0]
    half_world = np.float32(0.5 * world_size)

    # Phase 1: each atom computes its cell key (parallel)
    atom_cell = np.empty(N, dtype=np.int32)
    for i in prange(N):
        cx = int(pos_x[i] * inv_cell)
        cy = int(pos_y[i] * inv_cell)
        if cx < 0:
            cx = 0
        elif cx >= n_cells:
            cx = n_cells - 1
        if cy < 0:
            cy = 0
        elif cy >= n_cells:
            cy = n_cells - 1
        atom_cell[i] = cy * n_cells + cx

    # Phase 2: per-cell occupancy via counting sort (sequential — O(N + n_cells²))
    cell_start = np.zeros(n_cells2 + 1, dtype=np.int32)
    for i in range(N):
        cell_start[atom_cell[i] + 1] += 1
    for c in range(1, n_cells2 + 1):
        cell_start[c] += cell_start[c - 1]
    # cell_start[c] is now the start offset of cell c; cell_start[c+1] is its end.

    # Scatter atoms into cell-sorted array (sequential — O(N))
    write_pos = np.empty(n_cells2, dtype=np.int32)
    for c in range(n_cells2):
        write_pos[c] = cell_start[c]
    cell_atoms = np.empty(N, dtype=np.int32)
    for i in range(N):
        c = atom_cell[i]
        cell_atoms[write_pos[c]] = i
        write_pos[c] += 1

    # Phase 3: each atom counts its pairs (j > i, within r_list) — parallel, no writes shared
    pair_count_per_atom = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        cx = int(pos_x[i] * inv_cell)
        cy = int(pos_y[i] * inv_cell)
        if cx < 0:
            cx = 0
        elif cx >= n_cells:
            cx = n_cells - 1
        if cy < 0:
            cy = 0
        elif cy >= n_cells:
            cy = n_cells - 1

        cnt = 0
        xi = pos_x[i]
        yi = pos_y[i]
        for dx in range(-1, 2):
            nx = cx + dx
            if boundary_mode == BOUNDARY_PERIODIC:
                if nx < 0:
                    nx += n_cells
                elif nx >= n_cells:
                    nx -= n_cells
            elif nx < 0 or nx >= n_cells:
                continue
            for dy in range(-1, 2):
                ny = cy + dy
                if boundary_mode == BOUNDARY_PERIODIC:
                    if ny < 0:
                        ny += n_cells
                    elif ny >= n_cells:
                        ny -= n_cells
                elif ny < 0 or ny >= n_cells:
                    continue
                nc = ny * n_cells + nx
                k_start = cell_start[nc]
                k_end = cell_start[nc + 1]
                for k in range(k_start, k_end):
                    j = cell_atoms[k]
                    if i < j:
                        px = xi - pos_x[j]
                        py = yi - pos_y[j]
                        if boundary_mode == BOUNDARY_PERIODIC:
                            if px > half_world:
                                px -= world_size
                            elif px < -half_world:
                                px += world_size
                            if py > half_world:
                                py -= world_size
                            elif py < -half_world:
                                py += world_size
                        r2 = px * px + py * py
                        if r2 < r_list2:
                            cnt += 1
        pair_count_per_atom[i] = cnt

    # Phase 4: prefix-sum per-atom counts → write offsets (sequential — O(N))
    atom_pair_start = np.empty(N + 1, dtype=np.int32)
    total = 0
    for i in range(N):
        atom_pair_start[i] = total
        total += pair_count_per_atom[i]
    atom_pair_start[N] = total

    # Overflow contract: caller resizes pair_i/pair_j and retries. No partial writes.
    if total > max_pairs:
        return total

    # Phase 5: each atom writes its pairs to its dedicated segment (parallel, no atomics)
    for i in prange(N):
        cx = int(pos_x[i] * inv_cell)
        cy = int(pos_y[i] * inv_cell)
        if cx < 0:
            cx = 0
        elif cx >= n_cells:
            cx = n_cells - 1
        if cy < 0:
            cy = 0
        elif cy >= n_cells:
            cy = n_cells - 1

        wp = atom_pair_start[i]
        xi = pos_x[i]
        yi = pos_y[i]
        for dx in range(-1, 2):
            nx = cx + dx
            if boundary_mode == BOUNDARY_PERIODIC:
                if nx < 0:
                    nx += n_cells
                elif nx >= n_cells:
                    nx -= n_cells
            elif nx < 0 or nx >= n_cells:
                continue
            for dy in range(-1, 2):
                ny = cy + dy
                if boundary_mode == BOUNDARY_PERIODIC:
                    if ny < 0:
                        ny += n_cells
                    elif ny >= n_cells:
                        ny -= n_cells
                elif ny < 0 or ny >= n_cells:
                    continue
                nc = ny * n_cells + nx
                k_start = cell_start[nc]
                k_end = cell_start[nc + 1]
                for k in range(k_start, k_end):
                    j = cell_atoms[k]
                    if i < j:
                        px = xi - pos_x[j]
                        py = yi - pos_y[j]
                        if boundary_mode == BOUNDARY_PERIODIC:
                            if px > half_world:
                                px -= world_size
                            elif px < -half_world:
                                px += world_size
                            if py > half_world:
                                py -= world_size
                            elif py < -half_world:
                                py += world_size
                        r2 = px * px + py * py
                        if r2 < r_list2:
                            pair_i[wp] = i
                            pair_j[wp] = j
                            wp += 1

    return total

@njit(fastmath=True)
def build_atom_neighbor_csr(N, pair_i, pair_j, pair_count, nbr_start, nbr_idx):
    """Convert a half-pair list (i<j) into an atom-centric CSR neighbour list.

    nbr_start has length N+1 and is filled with start offsets per atom.
    nbr_idx has length >= 2*pair_count and is filled with each atom's
    neighbour indices, contiguous per atom: atom i's neighbours are at
    nbr_idx[nbr_start[i] : nbr_start[i+1]].

    Used to drive the parallel atom-centric LJ pair loop in
    integrate_n_steps. Each atom-iteration only writes to its own
    force_x/force_y slot, so no atomic adds or race conditions.
    """
    # 1. Count degrees (slots [1..N], slot 0 stays zero so the prefix
    #    sum below produces correct start offsets).
    for i in range(N + 1):
        nbr_start[i] = 0
    for k in range(pair_count):
        nbr_start[pair_i[k] + 1] += 1
        nbr_start[pair_j[k] + 1] += 1

    # 2. Prefix sum → start offsets.
    for i in range(1, N + 1):
        nbr_start[i] += nbr_start[i - 1]

    # 3. Scatter both directions of each pair into the neighbour buckets.
    #    write_pos[i] is the next free slot for atom i's bucket; we
    #    advance it as we write. Stored on a transient stack array.
    write_pos = np.empty(N, dtype=np.int32)
    for i in range(N):
        write_pos[i] = nbr_start[i]
    for k in range(pair_count):
        i = pair_i[k]
        j = pair_j[k]
        nbr_idx[write_pos[i]] = j
        write_pos[i] += 1
        nbr_idx[write_pos[j]] = i
        write_pos[j] += 1


@njit(fastmath=True, parallel=True)
def integrate_n_steps(
    steps_to_run,
    pos_x, pos_y, vel_x, vel_y, force_x, force_y,
    last_x, last_y,
    is_static,
    atom_sigma, atom_eps_sqrt, atom_mass,
    nbr_start, nbr_idx,             # atom-centric CSR neighbour list
    tether_entity_idx,  # For intra-entity force exclusion
    joint_ids,  # For coincident constraint LJ exclusion
    dt, gravity, r_cut2_base,
    skin_limit_sq,
    world_size,
    boundary_mode,
    wall_damping
):
    """Verlet integrator with cell-list LJ pair forces.

    atom_mass: per-particle mass array (matches atom_sigma / atom_eps_sqrt).
        Each particle's integration uses its own mass; previously a scalar.

    boundary_mode: 0=open (no wall, escapes filtered post-step),
                   1=reflecting walls (existing behaviour, wall_damping applied),
                   2=periodic — dynamic atoms wrap the [0, world_size]^2 domain
                   and pair distances use minimum-image. Tethered atoms
                   (is_static==3) are NOT wrapped under PBC; they continue to
                   reflect under mode 1 because their positions are driven by
                   spring forces toward in-domain anchors.
    """
    N = pos_x.shape[0]
    half_dt = 0.5 * dt
    half_dt2 = 0.5 * dt * dt
    half_world = 0.5 * world_size

    steps_done = 0

    for step in range(steps_to_run):
        # 0. Safety Check
        if step > 0 and step % SUBSTEP_SAFETY_CHECK_FREQ == 0:
            if check_displacement(pos_x, pos_y, last_x, last_y, skin_limit_sq):
                return steps_done

        # 1. Integration (Pos + Half Vel) - PARALLEL
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                # Dynamic Particle Integration
                inv_mi = 1.0 / atom_mass[i]
                dt2_2m_i = half_dt2 * inv_mi
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m_i
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m_i

                # Boundaries
                if boundary_mode == BOUNDARY_REFLECTING:
                    if xi >= world_size:
                        xi = 2.0 * world_size - xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    elif xi < 0.0:
                        xi = -xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    if yi >= world_size:
                        yi = 2.0 * world_size - yi
                        vel_y[i] = -vel_y[i] * wall_damping
                    elif yi < 0.0:
                        yi = -yi
                        vel_y[i] = -vel_y[i] * wall_damping
                elif boundary_mode == BOUNDARY_PERIODIC:
                    # Wrap (single subtraction is enough when atom motion per
                    # substep stays below world_size; the safety-check above
                    # rebuilds the neighbour list before larger displacements).
                    if xi >= world_size:
                        xi -= world_size
                    elif xi < 0.0:
                        xi += world_size
                    if yi >= world_size:
                        yi -= world_size
                    elif yi < 0.0:
                        yi += world_size

                pos_x[i] = xi
                pos_y[i] = yi
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt

            elif st == 3:
                # Tethered Particle Integration (bound to geometry via spring)
                # Integrates like dynamic but no gravity (follows geometry).
                # Tethered atoms reflect under REFLECTING but do not wrap under
                # PERIODIC — their anchors live at fixed in-domain positions.
                inv_mi = 1.0 / atom_mass[i]
                dt2_2m_i = half_dt2 * inv_mi
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m_i
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m_i

                if boundary_mode == BOUNDARY_REFLECTING:
                    if xi >= world_size:
                        xi = 2.0 * world_size - xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    elif xi < 0.0:
                        xi = -xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    if yi >= world_size:
                        yi = 2.0 * world_size - yi
                        vel_y[i] = -vel_y[i] * wall_damping
                    elif yi < 0.0:
                        yi = -yi
                        vel_y[i] = -vel_y[i] * wall_damping

                pos_x[i] = xi
                pos_y[i] = yi
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt

        # 2. Reset Forces - PARALLEL
        # All forces reset here. Tether forces were already used in Section 1.
        # Section 4 will only use LJ forces (if any) for tethered atoms.
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                # Dynamic: reset and apply per-particle gravity (F = m·g)
                force_x[i] = 0.0
                force_y[i] = atom_mass[i] * gravity
            else:
                # Static/Tethered: reset to zero (no gravity)
                force_x[i] = 0.0
                force_y[i] = 0.0

        # 3. Forces (Mixed Properties) - PARALLEL atom-centric
        # Each atom-iteration accumulates into its own force_x[i] / force_y[i]
        # slot only — no race conditions, no atomic adds. The trade-off is
        # that each pair (i,j) is now computed twice (once from i's view,
        # once from j's). Newton's third law is preserved by sign flip via
        # dx swapping when the pair is computed from j's side. Static atoms
        # (is_static==1) are skipped because their forces are never used
        # (positions and velocities are pinned).
        for i in prange(N):
            if is_static[i] == 1:
                continue
            fx_i = 0.0
            fy_i = 0.0
            sigma_i = atom_sigma[i]
            eps_sqrt_i = atom_eps_sqrt[i]
            jid_i = joint_ids[i]
            ent_i = tether_entity_idx[i]
            st_i_is_tethered = is_static[i] == 3

            for k in range(nbr_start[i], nbr_start[i + 1]):
                j = nbr_idx[k]

                # Skip intra-entity forces: tethered atoms on the same rigid
                # body should not exert LJ forces on each other.
                if st_i_is_tethered and is_static[j] == 3:
                    if ent_i >= 0 and ent_i == tether_entity_idx[j]:
                        continue

                # Skip joint forces: atoms at coincident joints share the
                # same non-zero joint_id and should not repel each other.
                if jid_i != 0 and jid_i == joint_ids[j]:
                    continue

                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                if boundary_mode == BOUNDARY_PERIODIC:
                    # Minimum-image: shorten any component greater than L/2
                    # so the pair force matches the closest periodic image.
                    if dx > half_world:
                        dx -= world_size
                    elif dx < -half_world:
                        dx += world_size
                    if dy > half_world:
                        dy -= world_size
                    elif dy < -half_world:
                        dy += world_size
                r2 = dx * dx + dy * dy

                if r2 < r_cut2_base:
                    s_ij = 0.5 * (sigma_i + atom_sigma[j])
                    s_ij2 = s_ij * s_ij
                    e_24 = 24.0 * eps_sqrt_i * atom_eps_sqrt[j]

                    f_scal = force_LJ_mixed(r2, s_ij2, e_24)
                    fx_i += f_scal * dx
                    fy_i += f_scal * dy

            force_x[i] += fx_i
            force_y[i] += fy_i

        # 4. Integration (Half Vel for Dynamic & Tethered) - PARALLEL
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                inv_mi = 1.0 / atom_mass[i]
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt
            elif st == 3:
                # Tethered: complete velocity update then apply damping
                # Damping prevents spring oscillation
                inv_mi = 1.0 / atom_mass[i]
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt
                vel_x[i] *= TETHER_DAMPING
                vel_y[i] *= TETHER_DAMPING

        steps_done += 1

    return steps_done


@njit(fastmath=True, parallel=True)
def integrate_n_steps_newton3(
    steps_to_run,
    pos_x, pos_y, vel_x, vel_y, force_x, force_y,
    last_x, last_y,
    is_static,
    atom_sigma, atom_eps_sqrt, atom_mass,
    pair_i, pair_j, pair_count,    # half-pair list (i < j); each pair appears once
    tether_entity_idx, joint_ids,
    dt, gravity, r_cut2_base,
    skin_limit_sq,
    world_size,
    boundary_mode,
    wall_damping,
    local_force_x, local_force_y,  # shape (T, N) — pre-allocated thread-local buffers
):
    """Verlet integrator using a HALF-pair list with Newton's-3rd-law and
    thread-local force accumulators.

    Differs from `integrate_n_steps` only in the force calculation phase:

    - The classic kernel iterates the atom-centric CSR (`nbr_start`, `nbr_idx`)
      so each pair (i,j) is computed twice — once when atom i scans its
      neighbours, once when atom j scans its neighbours. That duplication
      buys race-free parallelism (every thread writes only to its own atom's
      force slot) but doubles the LJ force work.

    - This kernel iterates `pair_i, pair_j` directly (each pair appears once)
      and uses thread-local force buffers (`local_force_x[t]`,
      `local_force_y[t]`) to avoid races. Each thread accumulates into its
      own slice, then a final per-atom merge sums the T contributions.

    The thread-local buffers must be pre-allocated to shape (T, N) where
    T = numba.get_num_threads(); they are passed in to keep allocation off
    the hot path. Buffers are zeroed at the start of each substep so prior
    data does not bleed in.

    Position/velocity update phases are line-for-line equivalent to the
    classic kernel; only the force phase changes.
    """
    N = pos_x.shape[0]
    T = local_force_x.shape[0]
    half_dt = 0.5 * dt
    half_dt2 = 0.5 * dt * dt
    half_world = 0.5 * world_size

    steps_done = 0

    for step in range(steps_to_run):
        # 0. Safety Check (matches classic kernel)
        if step > 0 and step % SUBSTEP_SAFETY_CHECK_FREQ == 0:
            if check_displacement(pos_x, pos_y, last_x, last_y, skin_limit_sq):
                return steps_done

        # 1. Integration (Pos + Half Vel) — copied verbatim from classic kernel.
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                inv_mi = 1.0 / atom_mass[i]
                dt2_2m_i = half_dt2 * inv_mi
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m_i
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m_i

                if boundary_mode == BOUNDARY_REFLECTING:
                    if xi >= world_size:
                        xi = 2.0 * world_size - xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    elif xi < 0.0:
                        xi = -xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    if yi >= world_size:
                        yi = 2.0 * world_size - yi
                        vel_y[i] = -vel_y[i] * wall_damping
                    elif yi < 0.0:
                        yi = -yi
                        vel_y[i] = -vel_y[i] * wall_damping
                elif boundary_mode == BOUNDARY_PERIODIC:
                    if xi >= world_size:
                        xi -= world_size
                    elif xi < 0.0:
                        xi += world_size
                    if yi >= world_size:
                        yi -= world_size
                    elif yi < 0.0:
                        yi += world_size

                pos_x[i] = xi
                pos_y[i] = yi
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt

            elif st == 3:
                inv_mi = 1.0 / atom_mass[i]
                dt2_2m_i = half_dt2 * inv_mi
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m_i
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m_i

                if boundary_mode == BOUNDARY_REFLECTING:
                    if xi >= world_size:
                        xi = 2.0 * world_size - xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    elif xi < 0.0:
                        xi = -xi
                        vel_x[i] = -vel_x[i] * wall_damping
                    if yi >= world_size:
                        yi = 2.0 * world_size - yi
                        vel_y[i] = -vel_y[i] * wall_damping
                    elif yi < 0.0:
                        yi = -yi
                        vel_y[i] = -vel_y[i] * wall_damping

                pos_x[i] = xi
                pos_y[i] = yi
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt

        # 2. Zero thread-local force buffers in parallel (per-thread strip).
        for t in prange(T):
            for i in range(N):
                local_force_x[t, i] = 0.0
                local_force_y[t, i] = 0.0

        # 3. Half-pair LJ force loop with Newton-3 thread-local accumulation.
        # Each pair contributes equal-and-opposite forces to its two atoms.
        # `get_thread_id()` returns the worker index; each worker writes only
        # to its own slice `local_force_*[t]`, so there are no races.
        for k in prange(pair_count):
            t = get_thread_id()
            i = pair_i[k]
            j = pair_j[k]

            si = is_static[i]
            sj = is_static[j]
            # Both static: no force needed (both pinned).
            if si == 1 and sj == 1:
                continue

            # Joint exclusion: atoms at coincident joints share the same
            # non-zero joint_id and should not repel each other.
            jid_i = joint_ids[i]
            if jid_i != 0 and jid_i == joint_ids[j]:
                continue

            # Tether intra-entity exclusion: tethered atoms on the same rigid
            # body should not exert LJ forces on each other.
            if si == 3 and sj == 3:
                ent_i = tether_entity_idx[i]
                if ent_i >= 0 and ent_i == tether_entity_idx[j]:
                    continue

            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            if boundary_mode == BOUNDARY_PERIODIC:
                if dx > half_world:
                    dx -= world_size
                elif dx < -half_world:
                    dx += world_size
                if dy > half_world:
                    dy -= world_size
                elif dy < -half_world:
                    dy += world_size
            r2 = dx * dx + dy * dy

            if r2 < r_cut2_base:
                s_ij = 0.5 * (atom_sigma[i] + atom_sigma[j])
                s_ij2 = s_ij * s_ij
                e_24 = 24.0 * atom_eps_sqrt[i] * atom_eps_sqrt[j]
                f_scal = force_LJ_mixed(r2, s_ij2, e_24)
                fx = f_scal * dx
                fy = f_scal * dy

                # Newton's 3rd law: equal and opposite. Skip writes for static
                # atoms (their forces are never used; pinned positions).
                if si != 1:
                    local_force_x[t, i] += fx
                    local_force_y[t, i] += fy
                if sj != 1:
                    local_force_x[t, j] -= fx
                    local_force_y[t, j] -= fy

        # 4. Merge thread-local forces into global force_x/y. Apply gravity for
        # dynamic atoms. Static atoms zeroed for defensive cleanliness.
        for i in prange(N):
            st = is_static[i]
            if st == 1:
                force_x[i] = 0.0
                force_y[i] = 0.0
                continue
            fx_sum = 0.0
            fy_sum = atom_mass[i] * gravity if st == 0 else 0.0
            for t in range(T):
                fx_sum += local_force_x[t, i]
                fy_sum += local_force_y[t, i]
            force_x[i] = fx_sum
            force_y[i] = fy_sum

        # 5. Integration (Half Vel for Dynamic & Tethered) — verbatim from classic.
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                inv_mi = 1.0 / atom_mass[i]
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt
            elif st == 3:
                inv_mi = 1.0 / atom_mass[i]
                vel_x[i] += force_x[i] * inv_mi * half_dt
                vel_y[i] += force_y[i] * inv_mi * half_dt
                vel_x[i] *= TETHER_DAMPING
                vel_y[i] *= TETHER_DAMPING

        steps_done += 1

    return steps_done


@njit(fastmath=True)
def spatial_sort(pos_x, pos_y, vel_x, vel_y, force_x, force_y, is_static,
                 atom_sigma, atom_eps_sqrt, atom_mass, world_size, cell_size):
    # Sorting is usually fast enough in serial, and parallel sort is complex to implement.
    N = pos_x.shape[0]
    n_cells = int(world_size // cell_size) + 1
    inv_cell = 1.0 / cell_size
    keys = np.zeros(N, dtype=np.int32)
    for i in range(N):
        cx = int(pos_x[i] * inv_cell)
        cy = int(pos_y[i] * inv_cell)
        keys[i] = cy * n_cells + cx
    perm = np.argsort(keys)

    pos_x[:] = pos_x[perm]
    pos_y[:] = pos_y[perm]
    vel_x[:] = vel_x[perm]
    vel_y[:] = vel_y[perm]
    force_x[:] = force_x[perm]
    force_y[:] = force_y[perm]
    is_static[:] = is_static[perm]
    atom_sigma[:] = atom_sigma[perm]
    atom_eps_sqrt[:] = atom_eps_sqrt[perm]
    atom_mass[:] = atom_mass[perm]

@njit(fastmath=True)
def apply_thermostat(vel_x, vel_y, atom_mass, is_static, target_temp, mix):
    """Berendsen velocity-rescaling thermostat (single-threaded JIT).

    Was previously @njit(parallel=True) with two prange loops. A
    microbenchmark across N from 700 to 220 000 showed the parallel
    version paid a fixed ~75-90 µs thread-dispatch cost that dominated
    the actual O(N) arithmetic up to N ≈ 30 000-35 000. Below that
    crossover the serial version is 2-30× faster (29× at N=700, the
    common case for today's sims). The two paths cross at the 10×
    simulation-size target; if we ever sustain N >> 35 000 we should
    revisit this with a per-call branch or a different thermostatting
    strategy (e.g., DPD pair thermostat folded into the LJ loop).
    """
    ke = 0.0
    count = 0
    N = vel_x.shape[0]

    # KE reduction — per-particle mass so mixed-mass scenes thermostat correctly
    for i in range(N):
        if is_static[i] == 0:
            ke += 0.5 * atom_mass[i] * (vel_x[i]**2 + vel_y[i]**2)
            count += 1

    if count == 0: return
    current_T = ke / count
    if current_T <= 1e-6: return
    scale = math.sqrt(target_temp / current_T)
    eff_scale = 1.0 + mix * (scale - 1.0)

    # Velocity rescale
    for i in range(N):
        if is_static[i] == 0:
            vel_x[i] *= eff_scale
            vel_y[i] *= eff_scale


# =============================================================================
# Tether Force Kernel (Two-Way Coupling)
# =============================================================================

# Entity type constants (must match simulation.py)
ENTITY_TYPE_LINE = 0
ENTITY_TYPE_CIRCLE = 1
ENTITY_TYPE_POINT = 2


@njit(fastmath=True)
def apply_tether_forces_pbd(
    # Particle arrays
    pos_x, pos_y,
    force_x, force_y,
    is_static,
    tether_entity_idx,
    tether_local_pos,
    tether_stiffness,
    # Entity arrays
    entity_positions,   # [N_ent, 4] - position data
    entity_forces,      # [N_ent, 3] - [fx, fy, torque] accumulator (OUTPUT)
    entity_com,         # [N_ent, 2] - center of mass
    entity_types        # [N_ent] - 0=Line, 1=Circle, 2=Point
):
    """
    Apply tether spring forces between atoms and their parent entities.

    For each tethered atom (is_static == 3):
    1. Compute the anchor point on the entity using local coordinates
    2. Calculate spring force: F = -k * (atom_pos - anchor_pos)
    3. Apply +F to atom force accumulator
    4. Apply -F to entity force accumulator (Newton's 3rd law)
    5. Calculate torque: τ = (anchor - COM) × (-F)

    This kernel is SERIAL because entity_forces writes could race.
    For now, correctness over speed.

    Args:
        pos_x, pos_y: Particle positions
        force_x, force_y: Particle force accumulators (modified)
        is_static: Particle types (0=dynamic, 1=static, 3=tethered)
        tether_entity_idx: Entity index for each particle (-1 if not tethered)
        tether_local_pos: Local coordinates [t or theta, unused]
        tether_stiffness: Spring constant k for each particle
        entity_positions: Entity geometry data
        entity_forces: Entity force accumulators [fx, fy, torque] (modified)
        entity_com: Entity centers of mass
        entity_types: Entity type codes
    """
    N = pos_x.shape[0]

    for i in range(N):
        # Only process tethered atoms
        if is_static[i] != 3:
            continue

        ent_idx = tether_entity_idx[i]
        if ent_idx < 0:
            continue

        local_t = tether_local_pos[i, 0]
        k = tether_stiffness[i]
        ent_type = entity_types[ent_idx]

        # Compute anchor position based on entity type
        anchor_x = 0.0
        anchor_y = 0.0

        if ent_type == ENTITY_TYPE_LINE:
            # Line: lerp between start and end using t
            start_x = entity_positions[ent_idx, 0]
            start_y = entity_positions[ent_idx, 1]
            end_x = entity_positions[ent_idx, 2]
            end_y = entity_positions[ent_idx, 3]
            anchor_x = start_x + local_t * (end_x - start_x)
            anchor_y = start_y + local_t * (end_y - start_y)

        elif ent_type == ENTITY_TYPE_CIRCLE:
            # Circle: center + radius * (cos(theta), sin(theta))
            center_x = entity_positions[ent_idx, 0]
            center_y = entity_positions[ent_idx, 1]
            radius = entity_positions[ent_idx, 2]
            theta = local_t  # For circles, local_t stores the angle
            anchor_x = center_x + radius * math.cos(theta)
            anchor_y = center_y + radius * math.sin(theta)

        elif ent_type == ENTITY_TYPE_POINT:
            # Point: anchor is the point itself
            anchor_x = entity_positions[ent_idx, 0]
            anchor_y = entity_positions[ent_idx, 1]

        # Calculate spring displacement (drift from anchor)
        dx = pos_x[i] - anchor_x
        dy = pos_y[i] - anchor_y

        # Spring force: F = -k * displacement
        fx = -k * dx
        fy = -k * dy

        # Clamp force magnitude to prevent explosions
        f_mag_sq = fx * fx + fy * fy
        if f_mag_sq > MAX_TETHER_FORCE * MAX_TETHER_FORCE:
            f_mag = math.sqrt(f_mag_sq)
            scale = MAX_TETHER_FORCE / f_mag
            fx *= scale
            fy *= scale

        # Apply force to atom (pulls it toward anchor)
        force_x[i] += fx
        force_y[i] += fy

        # Apply reaction force to entity (Newton's 3rd law)
        # Entity feels -F (opposite direction)
        entity_forces[ent_idx, 0] -= fx
        entity_forces[ent_idx, 1] -= fy

        # Calculate torque: τ = r × F (2D cross product)
        # r = anchor position relative to center of mass
        # F = force on entity = -[fx, fy]
        com_x = entity_com[ent_idx, 0]
        com_y = entity_com[ent_idx, 1]
        rx = anchor_x - com_x
        ry = anchor_y - com_y
        # 2D cross product: r × F = rx * Fy - ry * Fx
        # Force on entity is (-fx, -fy)
        torque = rx * (-fy) - ry * (-fx)
        entity_forces[ent_idx, 2] += torque