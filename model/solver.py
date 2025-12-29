import math
import numpy as np
from model.geometry import Line, Circle, Point

# Import Numba kernels (lazy import handled in solve_numba)
_kernels = None

def _get_kernels():
    """Lazy import of Numba kernels to avoid import-time JIT compilation."""
    global _kernels
    if _kernels is None:
        from model import solver_kernels as _kernels
    return _kernels


class Solver:
    @staticmethod
    def solve(constraints, entities, iterations=10, use_numba=False, interaction_data=None):
        """
        Iteratively solves the provided constraints against the entities.

        Parameters:
            constraints: List of Constraint objects
            entities: List of Entity objects (Point, Line, Circle)
            iterations: Number of solver iterations
            use_numba: If True, use the Numba-accelerated path; else use legacy OOP path
            interaction_data: Optional mouse interaction constraint (User Servo)
                Format: {'entity_idx': int, 'point_idx': int/None, 'handle_t': float/None, 'target': (x,y)}
        """
        if use_numba:
            Solver.solve_numba(constraints, entities, iterations, interaction_data)
        else:
            # Legacy OOP path with phased constraint execution
            # Separate constraints into unary (establish truth) and binary (propagate)
            unary_types = {'LENGTH', 'RADIUS', 'HORIZONTAL', 'VERTICAL'}
            unary_constraints = [c for c in constraints if c.type in unary_types]
            binary_constraints = [c for c in constraints if c.type not in unary_types]

            for _ in range(iterations):
                # Phase 1: Interaction (Mouse Input) - highest priority
                if interaction_data:
                    Solver._solve_interaction(entities, interaction_data)

                # Phase 2: Unary Constraints (establish individual entity truth)
                # These run first so FixedLength/Radius set the correct values
                for c in unary_constraints:
                    Solver.solve_single(c, entities)

                # Phase 3: Binary Constraints (propagate between entities)
                # These run after unary so they read the corrected values
                for c in binary_constraints:
                    Solver.solve_single(c, entities)

    @staticmethod
    def solve_numba(constraints, entities, iterations=10, interaction_data=None):
        """
        Numba-accelerated constraint solver using Data-Oriented Design.

        This method:
        1. Flattens entities into contiguous arrays
        2. Converts constraints to index-based arrays
        3. Runs JIT-compiled kernels (including interaction constraint)
        4. Writes results back to entity objects
        """
        kernels = _get_kernels()

        # === Step 1: Flatten Entities ===
        # Build mapping: (entity_idx, point_idx) -> flat_array_idx
        # and extract positions + inverse masses

        point_map = {}  # (ent_idx, pt_idx) -> flat_idx
        flat_positions = []
        flat_inv_masses = []
        flat_idx = 0

        for ent_idx, entity in enumerate(entities):
            if isinstance(entity, Point):
                point_map[(ent_idx, 0)] = flat_idx
                flat_positions.append(entity.pos.copy())
                flat_inv_masses.append(entity.get_inv_mass(0))
                flat_idx += 1
            elif isinstance(entity, Line):
                # Line has 2 points: index 0 (start) and index 1 (end)
                point_map[(ent_idx, 0)] = flat_idx
                flat_positions.append(entity.start.copy())
                flat_inv_masses.append(entity.get_inv_mass(0))
                flat_idx += 1

                point_map[(ent_idx, 1)] = flat_idx
                flat_positions.append(entity.end.copy())
                flat_inv_masses.append(entity.get_inv_mass(1))
                flat_idx += 1
            elif isinstance(entity, Circle):
                # Circle has 1 point: index 0 (center)
                point_map[(ent_idx, 0)] = flat_idx
                flat_positions.append(entity.center.copy())
                flat_inv_masses.append(entity.get_inv_mass(0))
                flat_idx += 1

        if flat_idx == 0:
            return  # No points to solve

        pos_array = np.array(flat_positions, dtype=np.float64)
        inv_mass_array = np.array(flat_inv_masses, dtype=np.float64)

        # === Step 2: Flatten Constraints by Type ===
        length_pairs = []
        length_values = []

        coincident_pairs = []

        collinear_points = []
        collinear_line_starts = []
        collinear_line_ends = []

        horizontal_pairs = []
        vertical_pairs = []
        parallel_line1_pairs = []
        parallel_line2_pairs = []
        equal_line1_pairs = []
        equal_line2_pairs = []

        for c in constraints:
            ctype = c.type

            if ctype == 'LENGTH':
                # indices = [entity_idx], value = target_length
                ent_idx = c.indices[0]
                # Length constraints apply to lines (start and end points)
                if (ent_idx, 0) in point_map and (ent_idx, 1) in point_map:
                    idx1 = point_map[(ent_idx, 0)]
                    idx2 = point_map[(ent_idx, 1)]
                    length_pairs.append([idx1, idx2])
                    length_values.append(c.value)

            elif ctype == 'COINCIDENT':
                # indices = [(e1, p1), (e2, p2)]
                if len(c.indices) == 2:
                    idx1_tuple = c.indices[0]
                    idx2_tuple = c.indices[1]
                    # Check it's point-point (both have valid pt_idx != -1)
                    if (isinstance(idx1_tuple, (list, tuple)) and
                        isinstance(idx2_tuple, (list, tuple)) and
                        idx1_tuple[1] != -1 and idx2_tuple[1] != -1):
                        key1 = (idx1_tuple[0], idx1_tuple[1])
                        key2 = (idx2_tuple[0], idx2_tuple[1])
                        if key1 in point_map and key2 in point_map:
                            coincident_pairs.append([point_map[key1], point_map[key2]])

            elif ctype == 'COLLINEAR':
                # indices = [(pt_entity_idx, pt_idx), line_entity_idx]
                pt_ref = c.indices[0]
                line_idx = c.indices[1]
                pt_key = (pt_ref[0], pt_ref[1])
                line_start_key = (line_idx, 0)
                line_end_key = (line_idx, 1)
                if (pt_key in point_map and
                    line_start_key in point_map and
                    line_end_key in point_map):
                    collinear_points.append(point_map[pt_key])
                    collinear_line_starts.append(point_map[line_start_key])
                    collinear_line_ends.append(point_map[line_end_key])

            elif ctype == 'HORIZONTAL':
                # Single line constraint
                if len(c.indices) >= 1:
                    ent_idx = c.indices[0]
                    if (ent_idx, 0) in point_map and (ent_idx, 1) in point_map:
                        horizontal_pairs.append([point_map[(ent_idx, 0)],
                                                 point_map[(ent_idx, 1)]])

            elif ctype == 'VERTICAL':
                # Single line constraint
                if len(c.indices) >= 1:
                    ent_idx = c.indices[0]
                    if (ent_idx, 0) in point_map and (ent_idx, 1) in point_map:
                        vertical_pairs.append([point_map[(ent_idx, 0)],
                                               point_map[(ent_idx, 1)]])

            elif ctype == 'PARALLEL':
                # Two-line constraint: indices = [entity_idx1, entity_idx2]
                if len(c.indices) >= 2:
                    ent1_idx = c.indices[0]
                    ent2_idx = c.indices[1]
                    if ((ent1_idx, 0) in point_map and (ent1_idx, 1) in point_map and
                        (ent2_idx, 0) in point_map and (ent2_idx, 1) in point_map):
                        parallel_line1_pairs.append([point_map[(ent1_idx, 0)],
                                                     point_map[(ent1_idx, 1)]])
                        parallel_line2_pairs.append([point_map[(ent2_idx, 0)],
                                                     point_map[(ent2_idx, 1)]])

            elif ctype == 'EQUAL':
                # Two-line constraint: indices = [entity_idx1, entity_idx2]
                if len(c.indices) >= 2:
                    ent1_idx = c.indices[0]
                    ent2_idx = c.indices[1]
                    if ((ent1_idx, 0) in point_map and (ent1_idx, 1) in point_map and
                        (ent2_idx, 0) in point_map and (ent2_idx, 1) in point_map):
                        equal_line1_pairs.append([point_map[(ent1_idx, 0)],
                                                  point_map[(ent1_idx, 1)]])
                        equal_line2_pairs.append([point_map[(ent2_idx, 0)],
                                                  point_map[(ent2_idx, 1)]])

        # Convert to numpy arrays
        length_pairs_arr = np.array(length_pairs, dtype=np.int64) if length_pairs else np.empty((0, 2), dtype=np.int64)
        length_values_arr = np.array(length_values, dtype=np.float64) if length_values else np.empty(0, dtype=np.float64)
        coincident_pairs_arr = np.array(coincident_pairs, dtype=np.int64) if coincident_pairs else np.empty((0, 2), dtype=np.int64)
        collinear_points_arr = np.array(collinear_points, dtype=np.int64) if collinear_points else np.empty(0, dtype=np.int64)
        collinear_starts_arr = np.array(collinear_line_starts, dtype=np.int64) if collinear_line_starts else np.empty(0, dtype=np.int64)
        collinear_ends_arr = np.array(collinear_line_ends, dtype=np.int64) if collinear_line_ends else np.empty(0, dtype=np.int64)
        horizontal_pairs_arr = np.array(horizontal_pairs, dtype=np.int64) if horizontal_pairs else np.empty((0, 2), dtype=np.int64)
        vertical_pairs_arr = np.array(vertical_pairs, dtype=np.int64) if vertical_pairs else np.empty((0, 2), dtype=np.int64)
        parallel_line1_arr = np.array(parallel_line1_pairs, dtype=np.int64) if parallel_line1_pairs else np.empty((0, 2), dtype=np.int64)
        parallel_line2_arr = np.array(parallel_line2_pairs, dtype=np.int64) if parallel_line2_pairs else np.empty((0, 2), dtype=np.int64)
        equal_line1_arr = np.array(equal_line1_pairs, dtype=np.int64) if equal_line1_pairs else np.empty((0, 2), dtype=np.int64)
        equal_line2_arr = np.array(equal_line2_pairs, dtype=np.int64) if equal_line2_pairs else np.empty((0, 2), dtype=np.int64)

        # === Prepare Interaction Data ===
        interaction_target = None
        interaction_indices = None
        interaction_handle_t = -1.0  # -1 means no handle_t (use point_idx mode)

        if interaction_data:
            entity_idx = interaction_data.get('entity_idx')
            point_idx = interaction_data.get('point_idx')
            handle_t = interaction_data.get('handle_t')
            target = interaction_data.get('target')

            if entity_idx is not None and target is not None:
                interaction_target = np.array(target, dtype=np.float64)

                if point_idx is not None:
                    # EDIT mode: single point
                    key = (entity_idx, point_idx)
                    if key in point_map:
                        interaction_indices = np.array([point_map[key], -1], dtype=np.int64)
                elif handle_t is not None:
                    # Body drag mode: both endpoints of a line
                    key0 = (entity_idx, 0)
                    key1 = (entity_idx, 1)
                    if key0 in point_map and key1 in point_map:
                        interaction_indices = np.array([point_map[key0], point_map[key1]], dtype=np.int64)
                        interaction_handle_t = handle_t
                else:
                    # Point or Circle: single point at index 0
                    key = (entity_idx, 0)
                    if key in point_map:
                        interaction_indices = np.array([point_map[key], -1], dtype=np.int64)

        # === Step 3: Run Numba Kernels (Phased Execution) ===
        for _ in range(iterations):
            # Phase 1: Interaction (Mouse Input) - highest priority
            if interaction_target is not None and interaction_indices is not None:
                kernels.solve_interaction_pbd(pos_array, interaction_target,
                                              interaction_indices, interaction_handle_t, inv_mass_array)

            # Phase 2: Unary Constraints (establish individual entity truth)
            # These run first so FixedLength/Radius set the correct values
            if length_pairs_arr.shape[0] > 0:
                kernels.solve_length_pbd(pos_array, length_pairs_arr, length_values_arr, inv_mass_array)
            if horizontal_pairs_arr.shape[0] > 0:
                kernels.solve_horizontal_pbd(pos_array, horizontal_pairs_arr, inv_mass_array)
            if vertical_pairs_arr.shape[0] > 0:
                kernels.solve_vertical_pbd(pos_array, vertical_pairs_arr, inv_mass_array)

            # Phase 3: Binary Constraints (propagate between entities)
            # These run after unary so they read the corrected values
            if coincident_pairs_arr.shape[0] > 0:
                kernels.solve_coincident_pbd(pos_array, coincident_pairs_arr, inv_mass_array)
            if collinear_points_arr.shape[0] > 0:
                kernels.solve_collinear_pbd(pos_array, collinear_points_arr,
                                            collinear_starts_arr, collinear_ends_arr, inv_mass_array)
            if parallel_line1_arr.shape[0] > 0:
                kernels.solve_parallel_pbd(pos_array, parallel_line1_arr, parallel_line2_arr, inv_mass_array)
            if equal_line1_arr.shape[0] > 0:
                kernels.solve_equal_length_pbd(pos_array, equal_line1_arr, equal_line2_arr, inv_mass_array)

        # === Step 4: Write Back to Entities ===
        for (ent_idx, pt_idx), flat_idx in point_map.items():
            entity = entities[ent_idx]
            new_pos = pos_array[flat_idx]
            entity.set_point(pt_idx, new_pos)

    @staticmethod
    def solve_single(constraint, entities):
        ctype = constraint.type
        indices = constraint.indices
        
        # --- COINCIDENT ---
        if ctype == 'COINCIDENT':
            # Case 1: Point-Point (2 indices, each is (ent_idx, pt_idx))
            if len(indices) == 2 and isinstance(indices[0], (list, tuple)) and isinstance(indices[1], (list, tuple)):
                Solver._solve_coincident_pt_pt(constraint, entities)
            # Case 2: Point-Entity (2 indices, one is (ent_idx, pt_idx), one is ent_idx)
            else:
                Solver._solve_coincident_pt_ent(constraint, entities)

        # --- LENGTH ---
        elif ctype == 'LENGTH':
            Solver._solve_length(constraint, entities)

        # --- RADIUS ---
        elif ctype == 'RADIUS':
            Solver._solve_radius(constraint, entities)

        # --- EQUAL LENGTH ---
        elif ctype == 'EQUAL':
            Solver._solve_equal_length(constraint, entities)

        # --- ANGLE / PARALLEL / PERPENDICULAR ---
        elif ctype in ['ANGLE', 'PARALLEL', 'PERPENDICULAR', 'HORIZONTAL', 'VERTICAL']:
            Solver._solve_angle(constraint, entities)
            
        # --- MIDPOINT ---
        elif ctype == 'MIDPOINT':
            Solver._solve_midpoint(constraint, entities)
            
        # --- COLLINEAR ---
        elif ctype == 'COLLINEAR':
            Solver._solve_collinear(constraint, entities)

    # --- Internal Solvers ---

    @staticmethod
    def _get_point_and_inv_mass(entity, pt_idx):
        p = entity.get_point(pt_idx)
        im = entity.get_inv_mass(pt_idx)
        return p, im

    @staticmethod
    def _solve_coincident_pt_pt(c, entities):
        idx1, idx2 = c.indices[0], c.indices[1]
        try:
            e1 = entities[idx1[0]]
            e2 = entities[idx2[0]]
        except IndexError: return

        p1, w1 = Solver._get_point_and_inv_mass(e1, idx1[1])
        p2, w2 = Solver._get_point_and_inv_mass(e2, idx2[1])
        
        w_sum = w1 + w2
        if w_sum == 0: return

        # Weighted average position
        target = (p1 * w1 + p2 * w2) / w_sum
        
        # Priority snap: if one is fully anchored (w=0), snap to it
        if w1 == 0: target = p1
        elif w2 == 0: target = p2

        if w1 > 0: e1.set_point(idx1[1], target)
        if w2 > 0: e2.set_point(idx2[1], target)

    @staticmethod
    def _solve_coincident_pt_ent(c, entities):
        # Identify which index is the point and which is the entity
        idx1, idx2 = c.indices[0], c.indices[1]
        
        # Helper to unpack (ent_idx, pt_idx) vs (ent_idx, -1)
        # In constraints.py Coincident: indices are [(e1, p1), (e2, p2)]
        # If p2 is -1, it's a point-entity constraint
        
        # Let's assume standard format from the old Coincident class:
        # One index is (ent_idx, pt_idx), the other is (ent_idx, -1)
        
        pt_ref = idx1 if idx1[1] != -1 else idx2
        ent_ref = idx1 if idx1[1] == -1 else idx2
        
        try:
            pt_ent = entities[pt_ref[0]]
            target_ent = entities[ent_ref[0]]
        except IndexError: return
        
        pt_idx = pt_ref[1]
        P, w_p = Solver._get_point_and_inv_mass(pt_ent, pt_idx)
        
        if isinstance(target_ent, Circle):
            C = target_ent.center
            R = target_ent.radius
            diff = P - C
            dist = np.linalg.norm(diff)
            if dist < 1e-6: diff = np.array([1.0, 0.0]); dist = 1.0
            
            error = dist - R
            dir_vec = diff / dist
            
            w_c = target_ent.get_inv_mass(0)
            w_r = 1.0 # Radius "mass"
            w_sum = w_p + w_c + w_r
            if w_sum == 0: return
            
            lambda_val = error / w_sum
            
            if w_p > 0: pt_ent.set_point(pt_idx, P - dir_vec * (w_p * lambda_val))
            if w_c > 0: target_ent.set_point(0, C + dir_vec * (w_c * lambda_val))
            # Optional: Resize circle (w_r logic) if we allowed radius changes here
            
        elif isinstance(target_ent, Line):
            A = target_ent.start
            B = target_ent.end
            u = B - A
            len_sq = np.dot(u, u)
            if len_sq < 1e-8: return
            
            v = P - A
            t = np.dot(v, u) / len_sq
            t_clamped = max(0.0, min(1.0, t))
            
            closest = A + t_clamped * u
            diff = closest - P
            
            w_a = target_ent.get_inv_mass(0)
            w_b = target_ent.get_inv_mass(1)
            
            # Simplified Position Based Dynamics for point-on-line
            # Just move point for now if line is heavy, or split
            # Using specific weights from old logic:
            W = w_p + w_a * (1.0 - t_clamped)**2 + w_b * t_clamped**2
            if W == 0: return
            
            lambda_val = 1.0 / W
            
            if w_p > 0: pt_ent.set_point(pt_idx, P + w_p * lambda_val * diff)
            if w_a > 0: target_ent.set_point(0, A - w_a * (1.0 - t_clamped) * lambda_val * diff)
            if w_b > 0: target_ent.set_point(1, B - w_b * t_clamped * lambda_val * diff)

    @staticmethod
    def _solve_length(c, entities):
        try: e = entities[c.indices[0]]
        except IndexError: return
        
        p1 = e.get_point(0)
        p2 = e.get_point(1)
        w1 = e.get_inv_mass(0)
        w2 = e.get_inv_mass(1)
        
        if w1 + w2 == 0: return
        
        curr_vec = p2 - p1
        curr_len = np.linalg.norm(curr_vec)
        if curr_len < 1e-6: return
        
        diff = curr_len - c.value
        correction = (curr_vec / curr_len) * diff
        
        if w1 > 0: e.set_point(0, p1 + correction * (w1 / (w1+w2)))
        if w2 > 0: e.set_point(1, p2 - correction * (w2 / (w1+w2)))

    @staticmethod
    def _solve_radius(c, entities):
        try:
            e = entities[c.indices[0]]
        except IndexError:
            return

        # Only applies to circles
        if not isinstance(e, Circle):
            return

        # Directly set the radius to the constraint value
        e.radius = c.value

    @staticmethod
    def _solve_equal_length(c, entities):
        """
        Solve equal length constraint using PBD constraint projection.

        This implements a proper Position-Based Dynamics approach where:
        - Error C = L1 - L2 is distributed to ALL 4 endpoints
        - Both lines negotiate toward equal length
        - Works correctly with other constraints (e.g., FixedLength on one line)

        The key insight is that we don't copy one length to the other.
        Instead, we project both lines toward satisfying L1 = L2.
        """
        try:
            e1 = entities[c.indices[0]]
            e2 = entities[c.indices[1]]
        except IndexError:
            return

        # Get current positions
        p1_start = e1.get_point(0)
        p1_end = e1.get_point(1)
        p2_start = e2.get_point(0)
        p2_end = e2.get_point(1)

        # Calculate current lengths
        v1 = p1_end - p1_start
        v2 = p2_end - p2_start
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-6 or len2 < 1e-6:
            return

        # Constraint error: we want L1 = L2, so C = L1 - L2
        C = len1 - len2

        if abs(C) < 1e-8:
            return  # Already equal

        # Get inverse masses for all 4 endpoints
        w1_start = e1.get_inv_mass(0)
        w1_end = e1.get_inv_mass(1)
        w2_start = e2.get_inv_mass(0)
        w2_end = e2.get_inv_mass(1)

        # Total inverse mass (weighted by how much each point affects length)
        # For a length constraint, each endpoint contributes equally (factor 0.5 each)
        # But we have 2 lines, so we need the sum of their "length mobilities"
        w1_total = w1_start + w1_end  # Line 1's ability to change length
        w2_total = w2_start + w2_end  # Line 2's ability to change length

        w_sum = w1_total + w2_total
        if w_sum < 1e-10:
            return  # Both lines fully anchored

        # PBD correction factor: how much total length change to distribute
        # Positive C means L1 > L2, so we need to shrink L1 and/or grow L2
        lambda_val = C / w_sum

        # Normalized direction vectors
        n1 = v1 / len1
        n2 = v2 / len2

        # Apply corrections to Line 1 (shrink if C > 0)
        # Line 1 correction magnitude: lambda * w1_total
        # Distributed to endpoints based on their individual masses
        if w1_total > 0:
            corr1 = lambda_val * w1_total
            if w1_start > 0:
                # Move start toward end (shrinks line)
                e1.set_point(0, p1_start + n1 * (corr1 * w1_start / w1_total))
            if w1_end > 0:
                # Move end toward start (shrinks line)
                e1.set_point(1, p1_end - n1 * (corr1 * w1_end / w1_total))

        # Apply corrections to Line 2 (grow if C > 0)
        # Line 2 correction magnitude: lambda * w2_total (but opposite direction)
        if w2_total > 0:
            corr2 = lambda_val * w2_total
            if w2_start > 0:
                # Move start away from end (grows line)
                e2.set_point(0, p2_start - n2 * (corr2 * w2_start / w2_total))
            if w2_end > 0:
                # Move end away from start (grows line)
                e2.set_point(1, p2_end + n2 * (corr2 * w2_end / w2_total))

    @staticmethod
    def _solve_angle(c, entities):
        idx = c.indices
        if c.type in ['HORIZONTAL', 'VERTICAL']:
            try: e = entities[idx[0]]
            except IndexError: return
            p1 = e.get_point(0); p2 = e.get_point(1)
            w1 = e.get_inv_mass(0); w2 = e.get_inv_mass(1)
            if w1 + w2 == 0: return
            
            if c.type == 'HORIZONTAL':
                avg_y = (p1[1] + p2[1]) / 2.0
                if w1 == 0: avg_y = p1[1]
                elif w2 == 0: avg_y = p2[1]
                if w1 > 0: e.set_point(0, np.array([p1[0], avg_y]))
                if w2 > 0: e.set_point(1, np.array([p2[0], avg_y]))
            else: # Vertical
                avg_x = (p1[0] + p2[0]) / 2.0
                if w1 == 0: avg_x = p1[0]
                elif w2 == 0: avg_x = p2[0]
                if w1 > 0: e.set_point(0, np.array([avg_x, p1[1]]))
                if w2 > 0: e.set_point(1, np.array([avg_x, p2[1]]))
                
        elif c.type in ['PARALLEL', 'PERPENDICULAR', 'ANGLE']:
            try: 
                e1 = entities[idx[0]]
                e2 = entities[idx[1]]
            except IndexError: return
            
            v1 = e1.end - e1.start
            v2 = e2.end - e2.start
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6: return
            
            a1 = math.atan2(v1[1], v1[0])
            a2 = math.atan2(v2[1], v2[0])
            
            target_a1, target_a2 = a1, a2
            
            if c.type == 'ANGLE':
                # Fixed relative angle
                curr_angle = a1 - a2 # Simplified
                # For fixed angle, we usually want e2 to be at e1 + angle
                # This needs a more robust implementation matching the old one
                # Re-using the logic from old FixedAngle:
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                curr_rad = math.atan2(cross, dot)
                target_rad = math.radians(c.value)
                diff = curr_rad - target_rad
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                
                # Apply diff/2 to each
                Solver._rotate_entity(e1, diff * 0.5)
                Solver._rotate_entity(e2, -diff * 0.5)
                return

            # Parallel/Perp
            if c.type == 'PARALLEL':
                if abs(a1 - a2) > math.pi/2:
                    if a2 < a1: a2 += math.pi 
                    else: a2 -= math.pi
                avg = (a1 + a2) / 2.0
                target_a1, target_a2 = avg, avg
            elif c.type == 'PERPENDICULAR':
                diff = a2 - a1
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                goal = math.pi/2 if diff > 0 else -math.pi/2
                corr = (goal - diff) / 2.0
                target_a1 = a1 - corr
                target_a2 = a2 + corr
            
            Solver._rotate_entity(e1, target_a1 - a1)
            Solver._rotate_entity(e2, target_a2 - a2)

    @staticmethod
    def _solve_midpoint(c, entities):
        try:
            pt_ent = entities[c.indices[0][0]]
            line_ent = entities[c.indices[1]]
        except IndexError: return

        pt_idx = c.indices[0][1]
        P = pt_ent.get_point(pt_idx)
        A = line_ent.get_point(0); B = line_ent.get_point(1)
        
        w_p = pt_ent.get_inv_mass(pt_idx)
        w_a = line_ent.get_inv_mass(0)
        w_b = line_ent.get_inv_mass(1)
        
        denom = w_p + 0.25 * (w_a + w_b)
        if denom == 0: return
        
        M = (A + B) * 0.5
        diff = M - P
        lambda_val = diff / denom
        
        if w_p > 0: pt_ent.set_point(pt_idx, P + w_p * lambda_val)
        if w_a > 0: line_ent.set_point(0, A - 0.5 * w_a * lambda_val)
        if w_b > 0: line_ent.set_point(1, B - 0.5 * w_b * lambda_val)

    @staticmethod
    def _solve_collinear(c, entities):
        try:
            pt_ent = entities[c.indices[0][0]]
            line_ent = entities[c.indices[1]]
        except IndexError: return

        pt_idx = c.indices[0][1]
        P = pt_ent.get_point(pt_idx)
        A = line_ent.get_point(0); B = line_ent.get_point(1)
        
        u = B - A
        len_sq = np.dot(u, u)
        if len_sq < 1e-8: return
        
        v = P - A
        t = np.dot(v, u) / len_sq
        S = A + t * u
        err = S - P
        
        w_p = pt_ent.get_inv_mass(pt_idx)
        w_a = line_ent.get_inv_mass(0)
        w_b = line_ent.get_inv_mass(1)
        
        W = w_p + w_a * (1.0 - t)**2 + w_b * t**2
        if W == 0: return
        
        lambda_val = 1.0 / W
        
        if w_p > 0: pt_ent.set_point(pt_idx, P + w_p * lambda_val * err)
        if w_a > 0: line_ent.set_point(0, A - w_a * (1.0 - t) * lambda_val * err)
        if w_b > 0: line_ent.set_point(1, B - w_b * t * lambda_val * err)

    @staticmethod
    def _rotate_entity(e, angle_delta, stiffness=0.5):
        if e.get_inv_mass(0) == 0 and e.get_inv_mass(1) == 0: return
        
        v = e.end - e.start
        length = np.linalg.norm(v)
        if length < 1e-6: return
        
        current_ang = math.atan2(v[1], v[0])
        target = current_ang + angle_delta * stiffness
        
        # Determine Pivot
        if e.get_inv_mass(0) == 0: pivot = e.start
        elif e.get_inv_mass(1) == 0: pivot = e.end
        else: pivot = (e.start + e.end) * 0.5
        
        c = math.cos(target)
        s = math.sin(target)
        
        if np.array_equal(pivot, e.start):
            e.set_point(1, e.start + np.array([c*length, s*length]))
        elif np.array_equal(pivot, e.end):
            e.set_point(0, e.end - np.array([c*length, s*length]))
        else:
            half = length / 2.0
            e.set_point(0, pivot - np.array([c*half, s*half]))
            e.set_point(1, pivot + np.array([c*half, s*half]))

    # -------------------------------------------------------------------------
    # Interaction Constraint (User Servo / Mouse Drag)
    # -------------------------------------------------------------------------

    @staticmethod
    def _solve_interaction(entities, data):
        """
        Solve the mouse interaction as a high-priority position constraint.

        The 'User Servo' pulls the grabbed point/handle toward the mouse target,
        treating it like a 0-length spring with very high stiffness.

        Args:
            entities: List of Entity objects
            data: {'entity_idx': int, 'point_idx': int/None, 'handle_t': float/None, 'target': (x,y)}
        """
        entity_idx = data.get('entity_idx')
        point_idx = data.get('point_idx')
        handle_t = data.get('handle_t')
        target = data.get('target')

        if entity_idx is None or target is None:
            return
        if entity_idx < 0 or entity_idx >= len(entities):
            return

        entity = entities[entity_idx]
        target = np.array(target, dtype=np.float64)

        # High stiffness for responsive feel (acts like near-infinite spring)
        stiffness = 0.8

        if isinstance(entity, Point):
            # Point entity: move directly toward target
            w = entity.get_inv_mass(0)
            if w > 0:
                current = entity.get_point(0)
                delta = target - current
                entity.set_point(0, current + delta * stiffness)

        elif isinstance(entity, Line):
            if point_idx is not None:
                # Dragging a specific endpoint (EDIT mode)
                w = entity.get_inv_mass(point_idx)
                if w > 0:
                    current = entity.get_point(point_idx)
                    delta = target - current
                    entity.set_point(point_idx, current + delta * stiffness)
            elif handle_t is not None:
                # Dragging the line body at parameter t
                # Calculate current handle position
                A = entity.get_point(0)
                B = entity.get_point(1)
                t = handle_t
                current_handle = A * (1.0 - t) + B * t

                # Error vector from handle to target
                delta = target - current_handle

                # Distribute correction to both endpoints based on t
                # Point closer to handle gets more movement
                w0 = entity.get_inv_mass(0)
                w1 = entity.get_inv_mass(1)

                # Weight by distance from handle (1-t for start, t for end)
                weight0 = (1.0 - t)
                weight1 = t

                if w0 > 0:
                    entity.set_point(0, A + delta * stiffness * weight0)
                if w1 > 0:
                    entity.set_point(1, B + delta * stiffness * weight1)

        elif isinstance(entity, Circle):
            # Circle: move center toward target
            w = entity.get_inv_mass(0)
            if w > 0:
                current = entity.get_point(0)
                delta = target - current
                entity.set_point(0, current + delta * stiffness)