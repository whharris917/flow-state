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
def solve_parallel_pbd(pos, line1_pairs, line2_pairs, inv_masses):
    """
    Solve parallel constraints using cross-product minimization.

    Two lines are parallel when their direction vectors have zero cross-product.
    This kernel rotates both lines toward the average angle.

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        line1_pairs: ndarray (M, 2) - Index pairs (start_idx, end_idx) for first lines
        line2_pairs: ndarray (M, 2) - Index pairs (start_idx, end_idx) for second lines
        inv_masses: ndarray (N,) - Inverse masses for all points

    Modifies pos in-place.
    """
    num_constraints = line1_pairs.shape[0]
    stiffness = 0.5

    for i in range(num_constraints):
        # Line 1 indices
        a1_idx = line1_pairs[i, 0]
        b1_idx = line1_pairs[i, 1]
        # Line 2 indices
        a2_idx = line2_pairs[i, 0]
        b2_idx = line2_pairs[i, 1]

        # Get positions
        a1x, a1y = pos[a1_idx, 0], pos[a1_idx, 1]
        b1x, b1y = pos[b1_idx, 0], pos[b1_idx, 1]
        a2x, a2y = pos[a2_idx, 0], pos[a2_idx, 1]
        b2x, b2y = pos[b2_idx, 0], pos[b2_idx, 1]

        # Direction vectors
        v1x = b1x - a1x
        v1y = b1y - a1y
        v2x = b2x - a2x
        v2y = b2y - a2y

        # Lengths
        len1 = np.sqrt(v1x * v1x + v1y * v1y)
        len2 = np.sqrt(v2x * v2x + v2y * v2y)

        if len1 < 1e-10 or len2 < 1e-10:
            continue

        # Current angles
        angle1 = np.arctan2(v1y, v1x)
        angle2 = np.arctan2(v2y, v2x)

        # Handle anti-parallel case: flip angle2 if lines point opposite
        diff = angle1 - angle2
        if diff > np.pi / 2:
            angle2 += np.pi
        elif diff < -np.pi / 2:
            angle2 -= np.pi

        # Target angle: average
        avg_angle = (angle1 + angle2) / 2.0

        # Calculate rotation deltas
        delta1 = (avg_angle - angle1) * stiffness
        delta2 = (avg_angle - angle2) * stiffness

        # Inverse masses for weighting
        w_a1 = inv_masses[a1_idx]
        w_b1 = inv_masses[b1_idx]
        w_a2 = inv_masses[a2_idx]
        w_b2 = inv_masses[b2_idx]

        # Rotate line 1 around its center (or anchored point)
        if w_a1 + w_b1 > 0:
            if w_a1 == 0:
                # Pivot around a1
                cos_d = np.cos(delta1)
                sin_d = np.sin(delta1)
                new_v1x = v1x * cos_d - v1y * sin_d
                new_v1y = v1x * sin_d + v1y * cos_d
                if w_b1 > 0:
                    pos[b1_idx, 0] = a1x + new_v1x
                    pos[b1_idx, 1] = a1y + new_v1y
            elif w_b1 == 0:
                # Pivot around b1
                cos_d = np.cos(delta1)
                sin_d = np.sin(delta1)
                new_v1x = v1x * cos_d - v1y * sin_d
                new_v1y = v1x * sin_d + v1y * cos_d
                if w_a1 > 0:
                    pos[a1_idx, 0] = b1x - new_v1x
                    pos[a1_idx, 1] = b1y - new_v1y
            else:
                # Pivot around center
                cx1 = (a1x + b1x) / 2.0
                cy1 = (a1y + b1y) / 2.0
                half_len1 = len1 / 2.0
                new_angle1 = angle1 + delta1
                cos_a = np.cos(new_angle1)
                sin_a = np.sin(new_angle1)
                pos[a1_idx, 0] = cx1 - cos_a * half_len1
                pos[a1_idx, 1] = cy1 - sin_a * half_len1
                pos[b1_idx, 0] = cx1 + cos_a * half_len1
                pos[b1_idx, 1] = cy1 + sin_a * half_len1

        # Rotate line 2 around its center (or anchored point)
        if w_a2 + w_b2 > 0:
            if w_a2 == 0:
                # Pivot around a2
                cos_d = np.cos(delta2)
                sin_d = np.sin(delta2)
                new_v2x = v2x * cos_d - v2y * sin_d
                new_v2y = v2x * sin_d + v2y * cos_d
                if w_b2 > 0:
                    pos[b2_idx, 0] = a2x + new_v2x
                    pos[b2_idx, 1] = a2y + new_v2y
            elif w_b2 == 0:
                # Pivot around b2
                cos_d = np.cos(delta2)
                sin_d = np.sin(delta2)
                new_v2x = v2x * cos_d - v2y * sin_d
                new_v2y = v2x * sin_d + v2y * cos_d
                if w_a2 > 0:
                    pos[a2_idx, 0] = b2x - new_v2x
                    pos[a2_idx, 1] = b2y - new_v2y
            else:
                # Pivot around center
                cx2 = (a2x + b2x) / 2.0
                cy2 = (a2y + b2y) / 2.0
                half_len2 = len2 / 2.0
                new_angle2 = angle2 + delta2
                cos_a = np.cos(new_angle2)
                sin_a = np.sin(new_angle2)
                pos[a2_idx, 0] = cx2 - cos_a * half_len2
                pos[a2_idx, 1] = cy2 - sin_a * half_len2
                pos[b2_idx, 0] = cx2 + cos_a * half_len2
                pos[b2_idx, 1] = cy2 + sin_a * half_len2


@njit(cache=True)
def solve_equal_length_pbd(pos, line1_pairs, line2_pairs, inv_masses, stiffness=1.0):
    """
    Solve equal length constraints using PBD constraint projection.

    Both lines are adjusted to satisfy L1 = L2 by distributing the error
    to all 4 endpoints based on inverse masses. This prevents the "phantom
    length" effect when one line has other constraints (like FixedLength).

    Parameters:
        pos: ndarray (N, 2) - Flattened positions of all points
        line1_pairs: ndarray (M, 2) - Index pairs (start_idx, end_idx) for first lines
        line2_pairs: ndarray (M, 2) - Index pairs (start_idx, end_idx) for second lines
        inv_masses: ndarray (N,) - Inverse masses for all points
        stiffness: float - Constraint stiffness (0.0 to 1.0)

    Modifies pos in-place (both lines).
    """
    num_constraints = line1_pairs.shape[0]

    for i in range(num_constraints):
        # Line 1 indices
        a1_idx = line1_pairs[i, 0]
        b1_idx = line1_pairs[i, 1]
        # Line 2 indices
        a2_idx = line2_pairs[i, 0]
        b2_idx = line2_pairs[i, 1]

        # Get Line 1 positions and length
        a1x, a1y = pos[a1_idx, 0], pos[a1_idx, 1]
        b1x, b1y = pos[b1_idx, 0], pos[b1_idx, 1]
        v1x = b1x - a1x
        v1y = b1y - a1y
        len1 = np.sqrt(v1x * v1x + v1y * v1y)

        # Get Line 2 positions and length
        a2x, a2y = pos[a2_idx, 0], pos[a2_idx, 1]
        b2x, b2y = pos[b2_idx, 0], pos[b2_idx, 1]
        v2x = b2x - a2x
        v2y = b2y - a2y
        len2 = np.sqrt(v2x * v2x + v2y * v2y)

        if len1 < 1e-10 or len2 < 1e-10:
            continue

        # Constraint error: C = L1 - L2 (we want L1 = L2)
        C = len1 - len2

        if abs(C) < 1e-8:
            continue  # Already equal

        # Inverse masses for all 4 endpoints
        w_a1 = inv_masses[a1_idx]
        w_b1 = inv_masses[b1_idx]
        w_a2 = inv_masses[a2_idx]
        w_b2 = inv_masses[b2_idx]

        # Total inverse mass per line
        w1_total = w_a1 + w_b1
        w2_total = w_a2 + w_b2
        w_sum = w1_total + w2_total

        if w_sum < 1e-10:
            continue

        # PBD correction factor
        lambda_val = (C / w_sum) * stiffness

        # Normalized direction vectors
        n1x = v1x / len1
        n1y = v1y / len1
        n2x = v2x / len2
        n2y = v2y / len2

        # Apply corrections to Line 1 (shrink if C > 0)
        if w1_total > 0:
            corr1 = lambda_val * w1_total
            if w_a1 > 0:
                ratio = w_a1 / w1_total
                pos[a1_idx, 0] += n1x * corr1 * ratio
                pos[a1_idx, 1] += n1y * corr1 * ratio
            if w_b1 > 0:
                ratio = w_b1 / w1_total
                pos[b1_idx, 0] -= n1x * corr1 * ratio
                pos[b1_idx, 1] -= n1y * corr1 * ratio

        # Apply corrections to Line 2 (grow if C > 0)
        if w2_total > 0:
            corr2 = lambda_val * w2_total
            if w_a2 > 0:
                ratio = w_a2 / w2_total
                pos[a2_idx, 0] -= n2x * corr2 * ratio
                pos[a2_idx, 1] -= n2y * corr2 * ratio
            if w_b2 > 0:
                ratio = w_b2 / w2_total
                pos[b2_idx, 0] += n2x * corr2 * ratio
                pos[b2_idx, 1] += n2y * corr2 * ratio


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
