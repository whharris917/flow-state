import math
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def force_LJ_mixed(r2, s_ij2, e_24):
    """
    LJ Force for mixed particle types.
    s_ij2: (sigma_i + sigma_j)/2 squared
    e_24: 24 * sqrt(eps_i * eps_j)
    """
    # 1. Clamp r2 to avoid singularity. 
    # Use a fraction of s_ij2 as the hard core limit.
    min_r2 = 0.64 * s_ij2
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

@njit(fastmath=True)
def build_neighbor_list(pos_x, pos_y, r_list2, cell_size, world_size, pair_i, pair_j):
    # NOTE: Kept serial because writing to the linked list 'head' array 
    # creates race conditions that are hard to solve without a full algorithm rewrite.
    N = pos_x.shape[0]
    n_cells = int(world_size // cell_size) + 1
    
    head = np.full((n_cells, n_cells), -1, dtype=np.int32)
    next_idx = np.full(N, -1, dtype=np.int32)
    
    inv_cell = 1.0 / cell_size
    for i in range(N):
        cx = int(pos_x[i] * inv_cell)
        cy = int(pos_y[i] * inv_cell)
        if cx < 0: cx = 0
        elif cx >= n_cells: cx = n_cells - 1
        if cy < 0: cy = 0
        elif cy >= n_cells: cy = n_cells - 1
        
        next_idx[i] = head[cx, cy]
        head[cx, cy] = i
        
    count = 0
    max_pairs = pair_i.shape[0]
    
    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx, cy]
            while i != -1:
                for dx in (-1, 0, 1):
                    nx = cx + dx
                    if nx < 0 or nx >= n_cells: continue
                    for dy in (-1, 0, 1):
                        ny = cy + dy
                        if ny < 0 or ny >= n_cells: continue
                        
                        j = head[nx, ny]
                        while j != -1:
                            if i < j:
                                px = pos_x[i] - pos_x[j]
                                py = pos_y[i] - pos_y[j]
                                r2 = px*px + py*py
                                if r2 < r_list2:
                                    if count < max_pairs:
                                        pair_i[count] = i
                                        pair_j[count] = j
                                        count += 1
                            j = next_idx[j]
                i = next_idx[i]
    return count

@njit(fastmath=True, parallel=True)
def integrate_n_steps(
    steps_to_run,
    pos_x, pos_y, vel_x, vel_y, force_x, force_y,
    last_x, last_y,
    is_static,
    kinematic_props,
    atom_sigma, atom_eps_sqrt, mass,
    pair_i, pair_j, pair_count,
    tether_entity_idx,  # For intra-entity force exclusion
    dt, gravity, r_cut2_base,
    skin_limit_sq,
    world_size,
    use_boundaries,
    wall_damping
):
    N = pos_x.shape[0]
    half_dt = 0.5 * dt
    dt2_2m = 0.5 * dt * dt / mass
    inv_mass = 1.0 / mass
    
    steps_done = 0
    
    for step in range(steps_to_run):
        # 0. Safety Check
        if step > 0 and step % 20 == 0:
            if check_displacement(pos_x, pos_y, last_x, last_y, skin_limit_sq):
                return steps_done

        # 1. Integration (Pos + Half Vel) - PARALLEL
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                # Dynamic Particle Integration
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m
                
                # Boundaries
                if use_boundaries:
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
                vel_x[i] += force_x[i] * inv_mass * half_dt
                vel_y[i] += force_y[i] * inv_mass * half_dt
            
            elif st == 2:
                # Kinematic (Rotating) Particle Update
                pivot_x = kinematic_props[i, 0]
                pivot_y = kinematic_props[i, 1]
                omega = kinematic_props[i, 2]

                d_theta = omega * dt
                c = math.cos(d_theta)
                s = math.sin(d_theta)

                rx = pos_x[i] - pivot_x
                ry = pos_y[i] - pivot_y

                pos_x[i] = pivot_x + rx * c - ry * s
                pos_y[i] = pivot_y + rx * s + ry * c

                rx = pos_x[i] - pivot_x
                ry = pos_y[i] - pivot_y
                vel_x[i] = -omega * ry
                vel_y[i] =  omega * rx

            elif st == 3:
                # Tethered Particle Integration (bound to geometry via spring)
                # Integrates like dynamic but no gravity (follows geometry)
                xi = pos_x[i] + vel_x[i] * dt + force_x[i] * dt2_2m
                yi = pos_y[i] + vel_y[i] * dt + force_y[i] * dt2_2m

                # Boundaries
                if use_boundaries:
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
                vel_x[i] += force_x[i] * inv_mass * half_dt
                vel_y[i] += force_y[i] * inv_mass * half_dt

        # 2. Reset Forces - PARALLEL
        # All forces reset here. Tether forces were already used in Section 1.
        # Section 4 will only use LJ forces (if any) for tethered atoms.
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                # Dynamic: reset and apply gravity
                force_x[i] = 0.0
                force_y[i] = mass * gravity
            else:
                # Static/Kinematic/Tethered: reset to zero (no gravity)
                force_x[i] = 0.0
                force_y[i] = 0.0

        # 3. Forces (Mixed Properties) - SERIAL
        # NOTE: Cannot easily parallelize without atomic adds or race conditions.
        for k in range(pair_count):
            i = pair_i[k]
            j = pair_j[k]

            # Skip intra-entity forces: tethered atoms on the same rigid body
            # should not exert LJ forces on each other
            if is_static[i] == 3 and is_static[j] == 3:
                ent_i = tether_entity_idx[i]
                ent_j = tether_entity_idx[j]
                if ent_i >= 0 and ent_i == ent_j:
                    continue

            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            r2 = dx*dx + dy*dy

            if r2 < r_cut2_base:
                s_ij = 0.5 * (atom_sigma[i] + atom_sigma[j])
                s_ij2 = s_ij * s_ij
                e_ij = atom_eps_sqrt[i] * atom_eps_sqrt[j]
                e_24 = 24.0 * e_ij

                f_scal = force_LJ_mixed(r2, s_ij2, e_24)
                fx = f_scal * dx
                fy = f_scal * dy

                force_x[i] += fx
                force_y[i] += fy
                force_x[j] -= fx
                force_y[j] -= fy

        # 4. Integration (Half Vel for Dynamic & Tethered) - PARALLEL
        for i in prange(N):
            st = is_static[i]
            if st == 0:
                vel_x[i] += force_x[i] * inv_mass * half_dt
                vel_y[i] += force_y[i] * inv_mass * half_dt
            elif st == 3:
                # Tethered: complete velocity update then apply damping
                # Damping prevents spring oscillation
                TETHER_DAMPING = 0.90
                vel_x[i] += force_x[i] * inv_mass * half_dt
                vel_y[i] += force_y[i] * inv_mass * half_dt
                vel_x[i] *= TETHER_DAMPING
                vel_y[i] *= TETHER_DAMPING

        steps_done += 1
        
    return steps_done

@njit(fastmath=True)
def spatial_sort(pos_x, pos_y, vel_x, vel_y, force_x, force_y, is_static, kinematic_props, atom_sigma, atom_eps_sqrt, world_size, cell_size):
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
    kinematic_props[:] = kinematic_props[perm]
    atom_sigma[:] = atom_sigma[perm]
    atom_eps_sqrt[:] = atom_eps_sqrt[perm]

@njit(fastmath=True, parallel=True)
def apply_thermostat(vel_x, vel_y, mass, is_static, target_temp, mix):
    ke = 0.0
    count = 0
    N = vel_x.shape[0]

    # Parallel reduction for Kinetic Energy
    for i in prange(N):
        if is_static[i] == 0:
            ke += 0.5 * mass * (vel_x[i]**2 + vel_y[i]**2)
            count += 1

    if count == 0: return
    current_T = ke / count
    if current_T <= 1e-6: return
    scale = math.sqrt(target_temp / current_T)
    eff_scale = 1.0 + mix * (scale - 1.0)

    # Parallel update of velocities
    for i in prange(N):
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
        is_static: Particle types (0=dynamic, 1=static, 2=kinematic, 3=tethered)
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
        MAX_FORCE = 10000.0
        f_mag_sq = fx * fx + fy * fy
        if f_mag_sq > MAX_FORCE * MAX_FORCE:
            f_mag = math.sqrt(f_mag_sq)
            scale = MAX_FORCE / f_mag
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