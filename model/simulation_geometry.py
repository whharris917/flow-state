"""
Geometry Manager for Flow State

Handles instantiating geometry from data (import/placement).
Also handles exporting geometry for saving.

REMOVED (per SoC): Context Menu logic (→ Controller), Hit testing (→ Sketch), 
                   Layout logic (→ Renderer)
"""

import math
import numpy as np

from model.geometry import Line, Circle, Point


class GeometryManager:
    """
    Handles instantiating geometry from data (Command Logic).
    """
    def __init__(self, sketch_or_sim):
        # Robust initialization: Handle receiving Simulation OR Sketch
        if hasattr(sketch_or_sim, 'sketch'):
            self.sketch = sketch_or_sim.sketch
        else:
            self.sketch = sketch_or_sim

    def place_geometry(self, data, offset_x, offset_y, current_time=0.0):
        """
        Instantiates geometry data at a specific location.
        Used when importing/pasting geometry into the world.
        
        Args:
            data: Dict with 'walls' and optionally 'constraints', 'materials'
            offset_x: World X coordinate for placement center
            offset_y: World Y coordinate for placement center  
            current_time: Current animation time (for driver base_time)
        """
        if not data.get('walls'):
            return
        
        # Calculate centroid of imported data to center it at cursor
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for w in data['walls']:
            if w['type'] == 'line':
                min_x = min(min_x, w['start'][0], w['end'][0])
                max_x = max(max_x, w['start'][0], w['end'][0])
                min_y = min(min_y, w['start'][1], w['end'][1])
                max_y = max(max_y, w['start'][1], w['end'][1])
            elif w['type'] == 'circle':
                min_x = min(min_x, w['center'][0] - w['radius'])
                max_x = max(max_x, w['center'][0] + w['radius'])
                min_y = min(min_y, w['center'][1] - w['radius'])
                max_y = max(max_y, w['center'][1] + w['radius'])
            elif w['type'] == 'point':
                min_x = min(min_x, w['x'])
                max_x = max(max_x, w['x'])
                min_y = min(min_y, w['y'])
                max_y = max(max_y, w['y'])

        # Handle edge case where bounds weren't updated
        if min_x == float('inf'):
            return
            
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        dx = offset_x - center_x
        dy = offset_y - center_y
        
        base_idx = len(self.sketch.entities)
        
        # 1. Import Materials (if present)
        if 'materials' in data:
            from model.properties import Material
            for mat_name, mat_data in data['materials'].items():
                if mat_name not in self.sketch.materials:
                    self.sketch.materials[mat_name] = Material.from_dict(mat_data)
        
        # 2. Create Entities
        for w in data['walls']:
            if w['type'] == 'line':
                s = (w['start'][0] + dx, w['start'][1] + dy)
                e = (w['end'][0] + dx, w['end'][1] + dy)
                l = Line(s, e, w.get('ref', False), w.get('material_id', "Wall"))
                l.anchored = w.get('anchored', [False, False])
                if 'anim' in w:
                    l.anim = w['anim'].copy()
                    # Update animation reference positions for new location
                    if 'ref_start' in l.anim:
                        l.anim['ref_start'] = l.start.copy()
                        l.anim['ref_end'] = l.end.copy()
                self.sketch.entities.append(l)
                
            elif w['type'] == 'circle':
                c = (w['center'][0] + dx, w['center'][1] + dy)
                circ = Circle(c, w['radius'], w.get('material_id', "Wall"))
                circ.anchored = w.get('anchored', [False])
                if 'anim' in w:
                    circ.anim = w['anim'].copy()
                self.sketch.entities.append(circ)
                
            elif w['type'] == 'point':
                p = Point(w['x'] + dx, w['y'] + dy, w.get('anchored', False), w.get('material_id', "Wall"))
                self.sketch.entities.append(p)

        # 3. Import Constraints (with remapped indices)
        from model.constraints import create_constraint
        for c_data in data.get('constraints', []):
            remapped_data = self._remap_constraint_indices(c_data, base_idx)
            c = create_constraint(remapped_data)
            if c:
                # Reset animation base_time to current time
                if c.base_time is not None:
                    c.base_time = current_time
                self.sketch.constraints.append(c)

    def export_geometry_data(self):
        """
        Exports current geometry as a portable data dict.
        Used for saving geometry files and copy/paste operations.
        
        Returns:
            Dict with 'walls', 'constraints', and 'materials' keys
        """
        if not self.sketch.entities:
            return None
        
        data = {
            'walls': [e.to_dict() for e in self.sketch.entities],
            'constraints': [c.to_dict() for c in self.sketch.constraints],
            'materials': {k: v.to_dict() for k, v in self.sketch.materials.items()}
        }
        
        return data

    def export_selection(self, entity_indices, include_internal_constraints=True):
        """
        Exports a subset of geometry (for copy operations).
        
        Args:
            entity_indices: List/set of entity indices to export
            include_internal_constraints: If True, include constraints that 
                                          reference only selected entities
        
        Returns:
            Dict with 'walls', 'constraints', 'materials' (indices normalized to 0-based)
        """
        if not entity_indices:
            return None
        
        entity_indices = sorted(entity_indices)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(entity_indices)}
        
        # Export selected entities
        walls = []
        material_ids_used = set()
        
        for idx in entity_indices:
            if 0 <= idx < len(self.sketch.entities):
                e = self.sketch.entities[idx]
                walls.append(e.to_dict())
                material_ids_used.add(e.material_id)
        
        # Export constraints that only reference selected entities
        constraints = []
        if include_internal_constraints:
            for c in self.sketch.constraints:
                if self._constraint_fully_within(c, entity_indices):
                    remapped = self._remap_constraint_for_export(c, index_map)
                    constraints.append(remapped)
        
        # Export only used materials
        materials = {}
        for mat_id in material_ids_used:
            if mat_id in self.sketch.materials:
                materials[mat_id] = self.sketch.materials[mat_id].to_dict()
        
        return {
            'walls': walls,
            'constraints': constraints,
            'materials': materials
        }

    def _remap_constraint_indices(self, c_data, base_idx):
        """
        Remaps constraint indices by adding base_idx offset.
        Used when importing geometry into existing sketch.
        """
        c_data_copy = c_data.copy()
        new_indices = []
        
        indices = c_data.get('indices', [])
        for item in indices:
            if isinstance(item, (list, tuple)):
                # Tuple format: (entity_idx, point_idx)
                new_indices.append([item[0] + base_idx, item[1]])
            else:
                # Simple entity index
                new_indices.append(item + base_idx)
        
        c_data_copy['indices'] = new_indices
        return c_data_copy

    def _constraint_fully_within(self, constraint, entity_indices):
        """
        Checks if all entity references in a constraint are within the given set.
        """
        entity_set = set(entity_indices)
        
        for item in constraint.indices:
            if isinstance(item, (list, tuple)):
                if item[0] not in entity_set:
                    return False
            else:
                if item not in entity_set:
                    return False
        return True

    def _remap_constraint_for_export(self, constraint, index_map):
        """
        Creates a dict representation of constraint with remapped indices.
        Used when exporting a selection (indices become 0-based for the selection).
        """
        c_dict = constraint.to_dict()
        new_indices = []
        
        for item in c_dict['indices']:
            if isinstance(item, (list, tuple)):
                new_indices.append([index_map[item[0]], item[1]])
            else:
                new_indices.append(index_map[item])
        
        c_dict['indices'] = new_indices
        return c_dict