import math
import numpy as np
from model.geometry import Line, Circle

class Solver:
    @staticmethod
    def solve(constraints, entities, iterations=10):
        """
        Iteratively solves the provided constraints against the entities.
        """
        for _ in range(iterations):
            for c in constraints:
                Solver.solve_single(c, entities)

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
        try: 
            e1 = entities[c.indices[0]]
            e2 = entities[c.indices[1]]
        except IndexError: return
        
        l1 = e1.length()
        l2 = e2.length()
        
        # Approximate mass of the "Length" property
        im1 = e1.get_inv_mass(0) + e1.get_inv_mass(1)
        im2 = e2.get_inv_mass(0) + e2.get_inv_mass(1)
        
        if im1 + im2 == 0: return
        
        target = (l1 * im2 + l2 * im1) / (im1 + im2)
        
        # Apply target length to both
        c_temp = type('obj', (object,), {'indices': [c.indices[0]], 'value': target})
        Solver._solve_length(c_temp, entities)
        c_temp.indices = [c.indices[1]]
        Solver._solve_length(c_temp, entities)

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