"""
Solver Kernels - Numba-accelerated constraint solving functions

This module provides JIT-compiled kernels for Position-Based Dynamics (PBD)
constraint solving. These are the low-level numerical functions called by
the hybrid solver path.

All functions use @njit(cache=True) for maximum performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def solve_length_pbd(pos, pairs, lengths, inv_masses, stiffness=1.0):
    """
    Solve length constraints using Position-Based Dynamics.

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        pairs: ndarray (M, 2) - Index pairs (p1_idx, p2_idx) for each constraint
        lengths: ndarray (M,) - Target lengths for each constraint
        inv_masses: ndarray (N,) - Inverse masses for all points
        stiffness: float - Constraint stiffness (0.0 to 1.0)

    Modifies pos in-place.
    """
    num_constraints = pairs.shape[0]

    for i in range(num_constraints):
        idx1 = pairs[i, 0]
        idx2 = pairs[i, 1]
        target_len = lengths[i]

        w1 = inv_masses[idx1]
        w2 = inv_masses[idx2]
        w_sum = w1 + w2

        if w_sum < 1e-10:
            continue

        # Current vector and length
        dx = pos[idx2, 0] - pos[idx1, 0]
        dy = pos[idx2, 1] - pos[idx1, 1]
        curr_len = np.sqrt(dx * dx + dy * dy)

        if curr_len < 1e-10:
            continue

        # Compute correction
        diff = curr_len - target_len
        correction_scale = (diff / curr_len) * stiffness

        cx = dx * correction_scale
        cy = dy * correction_scale

        # Apply weighted corrections
        ratio1 = w1 / w_sum
        ratio2 = w2 / w_sum

        if w1 > 0:
            pos[idx1, 0] += cx * ratio1
            pos[idx1, 1] += cy * ratio1
        if w2 > 0:
            pos[idx2, 0] -= cx * ratio2
            pos[idx2, 1] -= cy * ratio2


@njit(cache=True)
def solve_coincident_pbd(pos, pairs, inv_masses):
    """
    Solve point-point coincident constraints using weighted averaging.

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        pairs: ndarray (M, 2) - Index pairs (p1_idx, p2_idx) for each constraint
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    num_constraints = pairs.shape[0]

    for i in range(num_constraints):
        idx1 = pairs[i, 0]
        idx2 = pairs[i, 1]

        w1 = inv_masses[idx1]
        w2 = inv_masses[idx2]
        w_sum = w1 + w2

        if w_sum < 1e-10:
            continue

        p1x, p1y = pos[idx1, 0], pos[idx1, 1]
        p2x, p2y = pos[idx2, 0], pos[idx2, 1]

        # Compute weighted target position
        if w1 == 0:
            # p1 is anchored, snap p2 to p1
            target_x, target_y = p1x, p1y
        elif w2 == 0:
            # p2 is anchored, snap p1 to p2
            target_x, target_y = p2x, p2y
        else:
            # Weighted average
            target_x = (p1x * w1 + p2x * w2) / w_sum
            target_y = (p1y * w1 + p2y * w2) / w_sum

        # Apply positions
        if w1 > 0:
            pos[idx1, 0] = target_x
            pos[idx1, 1] = target_y
        if w2 > 0:
            pos[idx2, 0] = target_x
            pos[idx2, 1] = target_y


@njit(cache=True)
def solve_collinear_pbd(pos, point_indices, line_start_indices, line_end_indices, inv_masses):
    """
    Solve point-on-line (collinear) constraints.

    Projects the point onto the infinite line defined by start-end,
    then distributes corrections based on inverse masses.

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        point_indices: ndarray (M,) - Indices of points to constrain
        line_start_indices: ndarray (M,) - Indices of line start points
        line_end_indices: ndarray (M,) - Indices of line end points
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    num_constraints = point_indices.shape[0]

    for i in range(num_constraints):
        pt_idx = point_indices[i]
        a_idx = line_start_indices[i]
        b_idx = line_end_indices[i]

        # Get positions
        px, py = pos[pt_idx, 0], pos[pt_idx, 1]
        ax, ay = pos[a_idx, 0], pos[a_idx, 1]
        bx, by = pos[b_idx, 0], pos[b_idx, 1]

        # Line vector
        ux = bx - ax
        uy = by - ay
        len_sq = ux * ux + uy * uy

        if len_sq < 1e-10:
            continue

        # Project point onto line (parametric t)
        vx = px - ax
        vy = py - ay
        t = (vx * ux + vy * uy) / len_sq

        # Closest point on line
        sx = ax + t * ux
        sy = ay + t * uy

        # Error vector (from point to line)
        err_x = sx - px
        err_y = sy - py

        # Inverse masses
        w_p = inv_masses[pt_idx]
        w_a = inv_masses[a_idx]
        w_b = inv_masses[b_idx]

        # Weighted denominator (accounting for t position along line)
        t1 = 1.0 - t
        W = w_p + w_a * t1 * t1 + w_b * t * t

        if W < 1e-10:
            continue

        lambda_val = 1.0 / W

        # Apply corrections
        if w_p > 0:
            pos[pt_idx, 0] += w_p * lambda_val * err_x
            pos[pt_idx, 1] += w_p * lambda_val * err_y
        if w_a > 0:
            pos[a_idx, 0] -= w_a * t1 * lambda_val * err_x
            pos[a_idx, 1] -= w_a * t1 * lambda_val * err_y
        if w_b > 0:
            pos[b_idx, 0] -= w_b * t * lambda_val * err_x
            pos[b_idx, 1] -= w_b * t * lambda_val * err_y


@njit(cache=True)
def solve_horizontal_pbd(pos, pairs, inv_masses):
    """
    Solve horizontal constraints (both points share same Y coordinate).

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        pairs: ndarray (M, 2) - Index pairs (p1_idx, p2_idx) for each constraint
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    num_constraints = pairs.shape[0]

    for i in range(num_constraints):
        idx1 = pairs[i, 0]
        idx2 = pairs[i, 1]

        w1 = inv_masses[idx1]
        w2 = inv_masses[idx2]

        y1 = pos[idx1, 1]
        y2 = pos[idx2, 1]

        if w1 == 0:
            avg_y = y1
        elif w2 == 0:
            avg_y = y2
        elif w1 + w2 > 0:
            avg_y = (y1 + y2) / 2.0
        else:
            continue

        if w1 > 0:
            pos[idx1, 1] = avg_y
        if w2 > 0:
            pos[idx2, 1] = avg_y


@njit(cache=True)
def solve_vertical_pbd(pos, pairs, inv_masses):
    """
    Solve vertical constraints (both points share same X coordinate).

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        pairs: ndarray (M, 2) - Index pairs (p1_idx, p2_idx) for each constraint
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    num_constraints = pairs.shape[0]

    for i in range(num_constraints):
        idx1 = pairs[i, 0]
        idx2 = pairs[i, 1]

        w1 = inv_masses[idx1]
        w2 = inv_masses[idx2]

        x1 = pos[idx1, 0]
        x2 = pos[idx2, 0]

        if w1 == 0:
            avg_x = x1
        elif w2 == 0:
            avg_x = x2
        elif w1 + w2 > 0:
            avg_x = (x1 + x2) / 2.0
        else:
            continue

        if w1 > 0:
            pos[idx1, 0] = avg_x
        if w2 > 0:
            pos[idx2, 0] = avg_x


@njit(cache=True)
def solve_interaction_pbd(pos, target, indices, handle_t, inv_masses):
    """
    Solve the mouse interaction constraint (User Servo).

    Pulls the grabbed point(s) toward the mouse target with high stiffness.
    This acts like a 0-length spring that negotiates with other constraints.

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        target: ndarray (2,) - Target world position (mouse location)
        indices: ndarray (2,) - Point indices [idx0, idx1]. If idx1 == -1, single point mode.
        handle_t: float - Parameter t (0.0-1.0) for line body drag. If < 0, use single point mode.
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    # High stiffness for responsive feel
    stiffness = 0.8

    idx0 = indices[0]
    idx1 = indices[1]

    if idx1 < 0 or handle_t < 0:
        # Single point mode (Point, Circle, or EDIT mode on Line endpoint)
        w = inv_masses[idx0]
        if w > 0:
            delta_x = target[0] - pos[idx0, 0]
            delta_y = target[1] - pos[idx0, 1]
            pos[idx0, 0] += delta_x * stiffness
            pos[idx0, 1] += delta_y * stiffness
    else:
        # Line body drag mode: both endpoints, weighted by handle_t
        # Current handle position = A * (1-t) + B * t
        t = handle_t
        ax, ay = pos[idx0, 0], pos[idx0, 1]
        bx, by = pos[idx1, 0], pos[idx1, 1]

        handle_x = ax * (1.0 - t) + bx * t
        handle_y = ay * (1.0 - t) + by * t

        # Error from handle to target
        delta_x = target[0] - handle_x
        delta_y = target[1] - handle_y

        # Distribute correction to endpoints based on t
        w0 = inv_masses[idx0]
        w1 = inv_masses[idx1]

        # Weight by proximity to grab point
        weight0 = 1.0 - t
        weight1 = t

        if w0 > 0:
            pos[idx0, 0] += delta_x * stiffness * weight0
            pos[idx0, 1] += delta_y * stiffness * weight0
        if w1 > 0:
            pos[idx1, 0] += delta_x * stiffness * weight1
            pos[idx1, 1] += delta_y * stiffness * weight1
