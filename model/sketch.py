import copy
import numpy as np
import math
from model.geometry import Line, Circle, Point
from model.constraints import create_constraint
from model.solver import Solver
from model.properties import Material

class Sketch:
    """
    The Sketch is the 'Model' or 'Blueprint'.
    It holds Geometries, Constraints, and Materials.
    """
    def __init__(self):
        self.entities = []      # Lines, Circles, Points
        self.constraints = []   # Constraint Data Objects
        self.drivers = []       # Animation Drivers
        
        # Material Registry
        self.materials = {
            "Default": Material("Default", sigma=1.0, epsilon=1.0, color=(180, 180, 180), physical=True),
            "Ghost": Material("Ghost", sigma=1.0, epsilon=1.0, color=(100, 100, 100), physical=False),
            "Steel": Material("Steel", sigma=1.0, epsilon=2.0, color=(100, 150, 200), physical=True)
        }

    # --- Geometry Queries (New SoC Compliance) ---

    def find_entity_at(self, x, y, radius):
        """
        Finds the index of an entity intersecting the circle (x, y, radius).
        Moved from simulation_geometry.py to keep Model logic encapsulated.
        """
        best_dist = float('inf')
        best_idx = -1

        for i, w in enumerate(self.entities):
            dist = float('inf')

            if isinstance(w, Line):
                if np.array_equal(w.start, w.end):
                    continue
                p1 = w.start
                p2 = w.end
                p3 = np.array([x, y])

                # Point-Line Distance
                d_vec = p2 - p1
                len_sq = np.dot(d_vec, d_vec)
                if len_sq == 0:
                    dist = np.linalg.norm(p3 - p1)
                else:
                    t = np.dot(p3 - p1, d_vec) / len_sq
                    # For infinite ref lines, don't clamp t to segment bounds
                    if not (w.ref and w.infinite):
                        t = max(0, min(1, t))
                    proj = p1 + t * d_vec
                    dist = np.linalg.norm(p3 - proj)
                    
            elif isinstance(w, Circle):
                center_dist = math.hypot(x - w.center[0], y - w.center[1])
                dist = abs(center_dist - w.radius)
                
            elif isinstance(w, Point):
                 dist = math.hypot(x - w.pos[0], y - w.pos[1])

            if dist < radius and dist < best_dist:
                best_dist = dist
                best_idx = i
                
        return best_idx

    # --- Geometry API ---

    def add_line(self, start, end, is_ref=False, anchored=None, material_id="Default"):
        if anchored is None: anchored = [False, False]
        l = Line(start, end, is_ref, material_id)
        l.anchored = anchored
        self.entities.append(l)
        return len(self.entities) - 1

    def add_circle(self, center, radius, anchored=None, material_id="Default"):
        if anchored is None: anchored = [False]
        c = Circle(center, radius, material_id)
        c.anchored = anchored
        self.entities.append(c)
        return len(self.entities) - 1

    def update_entity(self, index, **kwargs):
        """
        Generic update for entity properties.
        """
        if 0 <= index < len(self.entities):
            e = self.entities[index]
            
            # Position updates
            if 'start' in kwargs and hasattr(e, 'start'): e.start[:] = kwargs['start']
            if 'end' in kwargs and hasattr(e, 'end'): e.end[:] = kwargs['end']
            if 'center' in kwargs and hasattr(e, 'center'): e.center[:] = kwargs['center']
            if 'radius' in kwargs and hasattr(e, 'radius'): e.radius = kwargs['radius']
            
            # Material Assignment
            if 'material_id' in kwargs: 
                e.material_id = kwargs['material_id']
            
            # Anchors
            if 'anchored' in kwargs: e.anchored = kwargs['anchored']
            
            # Trigger Solve
            self.solve()

    def remove_entity(self, index):
        """
        Removes an entity and any dependent constraints.
        """
        if 0 <= index < len(self.entities):
            self.constraints = [c for c in self.constraints if not self._constraint_involves(c, index)]
            for c in self.constraints:
                self._shift_constraint_indices(c, index)
            self.entities.pop(index)
            self.solve()

    def get_entity(self, index):
        if 0 <= index < len(self.entities):
            return self.entities[index]
        return None

    def toggle_anchor(self, entity_idx, point_idx):
        """
        Toggles the anchor state of a point on an entity.
        Anchored points are fixed in space during constraint solving.
        """
        if not (0 <= entity_idx < len(self.entities)):
            return
        
        entity = self.entities[entity_idx]
        
        if hasattr(entity, 'anchored'):
            if isinstance(entity.anchored, list):
                if 0 <= point_idx < len(entity.anchored):
                    entity.anchored[point_idx] = not entity.anchored[point_idx]
            else:
                # Point entity has single anchored bool
                entity.anchored = not entity.anchored

    # --- Compatibility Alias ---
    
    @property
    def walls(self):
        """Alias for entities - maintains compatibility with CONSTRAINT_DEFS."""
        return self.entities

    # --- Material API ---
    
    def add_material(self, material):
        if isinstance(material, Material):
            self.materials[material.name] = material

    def get_material(self, material_id):
        return self.materials.get(material_id, self.materials["Default"])

    # --- Constraint API ---

    def add_constraint(self, type_name, indices, value=None):
        data = {'type': type_name, 'indices': indices}
        if value is not None: data['value'] = value
        c = create_constraint(data)
        if c:
            self._handle_constraint_conflicts(c)
            self.constraints.append(c)
            self.solve(iterations=500)
            return c
        return None

    def remove_constraint(self, index):
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
            self.solve()

    def add_constraint_object(self, constraint, solve=True):
        """
        Adds a constraint object, handling conflicts with existing angle constraints.
        For angle-type constraints on the same entities, new replaces old.
        
        Args:
            constraint: The constraint to add
            solve: If True (default), run solver after adding. Set to False
                   for batch operations, then call solve() manually once.
        """
        angle_types = ['PARALLEL', 'PERPENDICULAR', 'HORIZONTAL', 'VERTICAL']
        
        if hasattr(constraint, 'type') and constraint.type in angle_types:
            new_indices = set(constraint.indices) if isinstance(constraint.indices, (list, tuple)) else {constraint.indices}
            
            keep = []
            for c in self.constraints:
                if getattr(c, 'type', '') in angle_types:
                    old_indices = set(c.indices) if isinstance(c.indices, (list, tuple)) else {c.indices}
                    if old_indices == new_indices:
                        continue  # Remove conflicting constraint
                keep.append(c)
            self.constraints = keep
        
        self.constraints.append(constraint)
        
        if solve:
            self.solve(iterations=500)

    def try_create_constraint(self, ctype, entity_idxs, point_idxs):
        """
        Attempts to create a constraint object WITHOUT adding it to the sketch.

        This is used by the command system to create constraints that can be
        properly tracked for undo/redo.

        Args:
            ctype: Constraint type string (e.g., 'LENGTH', 'PARALLEL')
            entity_idxs: List of entity indices
            point_idxs: List of (entity_idx, point_idx) tuples

        Returns:
            The constraint object if valid, None otherwise
        """
        from core.definitions import CONSTRAINT_DEFS

        rules = CONSTRAINT_DEFS.get(ctype, [])

        for rule in rules:
            if len(entity_idxs) == rule['w'] and len(point_idxs) == rule['p']:
                valid = True

                # Type check if required
                if rule.get('t'):
                    for idx in entity_idxs:
                        if idx < len(self.entities) and not isinstance(self.entities[idx], rule['t']):
                            valid = False
                            break

                if valid:
                    # Factory function expects (sketch, walls, pts)
                    return rule['f'](self, entity_idxs, point_idxs)

        return None

    def attempt_apply_constraint(self, ctype, entity_idxs, point_idxs):
        """
        Attempts to create and apply a constraint based on selection.

        WARNING: This method bypasses the command queue and should only be used
        for non-undoable operations. For undoable constraint creation, use
        try_create_constraint() with AddConstraintCommand.

        Args:
            ctype: Constraint type string (e.g., 'LENGTH', 'PARALLEL')
            entity_idxs: List of entity indices
            point_idxs: List of (entity_idx, point_idx) tuples

        Returns:
            True if constraint was successfully applied, False otherwise
        """
        constraint = self.try_create_constraint(ctype, entity_idxs, point_idxs)
        if constraint:
            self.add_constraint_object(constraint)
            return True
        return False

    def set_driver(self, constraint_index, driver_data):
        if 0 <= constraint_index < len(self.constraints):
            c = self.constraints[constraint_index]
            c.driver = driver_data
            if getattr(c, 'value', None) is not None:
                c.base_value = c.value

    def update_drivers(self, time):
        for c in self.constraints:
            if hasattr(c, 'driver') and c.driver:
                d = c.driver
                if c.base_value is None: c.base_value = c.value
                base = c.base_value
                t0 = getattr(c, 'base_time', 0.0)
                if t0 is None: t0 = 0.0
                dt_drive = time - t0
                
                if d['type'] == 'sin':
                    offset = d['amp'] * math.sin(2 * math.pi * d['freq'] * dt_drive + math.radians(d['phase']))
                    c.value = base + offset
                elif d['type'] == 'lin':
                    c.value = base + d['rate'] * dt_drive

    def solve(self, iterations=20):
        Solver.solve(self.constraints, self.entities, iterations)

    def clear(self):
        self.entities = []
        self.constraints = []
        # Materials persist!

    # --- Private Helpers ---

    def _constraint_involves(self, c, entity_idx):
        for idx in c.indices:
            if isinstance(idx, (list, tuple)):
                if idx[0] == entity_idx: return True
            else:
                if idx == entity_idx: return True
        return False

    def _shift_constraint_indices(self, c, removed_idx):
        new_indices = []
        for idx in c.indices:
            if isinstance(idx, (list, tuple)):
                w_idx, pt_idx = idx
                if w_idx > removed_idx: w_idx -= 1
                new_indices.append((w_idx, pt_idx))
            else:
                val = idx
                if val > removed_idx: val -= 1
                new_indices.append(val)
        c.indices = new_indices

    def _handle_constraint_conflicts(self, new_c):
        angle_types = ['PARALLEL', 'PERPENDICULAR', 'HORIZONTAL', 'VERTICAL']
        if new_c.type in angle_types:
            new_indices = set(new_c.indices) if isinstance(new_c.indices, (list, tuple)) else {new_c.indices}
            keep = []
            for c in self.constraints:
                if c.type in angle_types:
                    old_indices = set(c.indices) if isinstance(c.indices, (list, tuple)) else {c.indices}
                    if old_indices == new_indices: continue 
                keep.append(c)
            self.constraints = keep

    # --- Serialization ---

    def to_dict(self):
        return {
            'entities': [e.to_dict() for e in self.entities],
            'constraints': [c.to_dict() for c in self.constraints],
            'materials': {k: v.to_dict() for k, v in self.materials.items()}
        }

    def restore(self, data):
        # Local imports
        from model.geometry import Line, Circle, Point
        from model.properties import Material
        
        if 'materials' in data:
            self.materials = {}
            for k, v in data['materials'].items():
                self.materials[k] = Material.from_dict(v)
        if "Default" not in self.materials:
             self.materials["Default"] = Material("Default")

        self.entities = []
        for e_data in data.get('entities', []):
            if e_data['type'] == 'line': self.entities.append(Line.from_dict(e_data))
            elif e_data['type'] == 'circle': self.entities.append(Circle.from_dict(e_data))
            elif e_data['type'] == 'point': self.entities.append(Point.from_dict(e_data))
            
        self.constraints = []
        for c_data in data.get('constraints', []):
            c = create_constraint(c_data)
            if c: self.constraints.append(c)