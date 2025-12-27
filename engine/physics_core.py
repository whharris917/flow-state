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

        # 2. Reset Forces - PARALLEL
        for i in prange(N):
            force_x[i] = 0.0
            if is_static[i] == 0:
                force_y[i] = mass * gravity
            else:
                force_y[i] = 0.0

        # 3. Forces (Mixed Properties) - SERIAL
        # NOTE: Cannot easily parallelize without atomic adds or race conditions.
        for k in range(pair_count):
            i = pair_i[k]
            j = pair_j[k]
            
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

        # 4. Integration (Half Vel for Dynamic) - PARALLEL
        for i in prange(N):
            if is_static[i] == 0:
                vel_x[i] += force_x[i] * inv_mass * half_dt
                vel_y[i] += force_y[i] * inv_mass * half_dt
        
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