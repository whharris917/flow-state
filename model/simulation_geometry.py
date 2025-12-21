import math
import numpy as np
import core.utils as utils

from model.geometry import Line, Circle, Point

class GeometryManager:
    """
    Handles instantiating geometry from data (Command Logic).
    REMOVED: Context Menu logic (Controller) and Export logic (Persistence).
    REMOVED: Hit testing (moved to Sketch).
    REMOVED: Layout logic (moved to Renderer).
    """
    def __init__(self, sketch_or_sim):
        # Robust initialization: Handle receiving Simulation OR Sketch
        if hasattr(sketch_or_sim, 'sketch'):
            self.sketch = sketch_or_sim.sketch
        else:
            self.sketch = sketch_or_sim

    def place_geometry(self, data, offset_x, offset_y, current_time=0.0):
        """Instantiates geometry data at a specific location."""
        # Calculate centroid of imported data to center it at cursor
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        if not data.get('walls'): return
        
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

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        dx = offset_x - center_x
        dy = offset_y - center_y
        
        base_idx = len(self.sketch.entities)
        
        # 1. Create Entities
        for w in data['walls']:
            if w['type'] == 'line':
                s = (w['start'][0] + dx, w['start'][1] + dy)
                e = (w['end'][0] + dx, w['end'][1] + dy)
                l = Line(s, e, w.get('ref', False), w.get('material_id', "Default"))
                l.anchored = w.get('anchored', [False, False])
                if 'anim' in w: l.anim = w['anim']
                self.sketch.entities.append(l)
            elif w['type'] == 'circle':
                c = (w['center'][0] + dx, w['center'][1] + dy)
                circ = Circle(c, w['radius'], w.get('material_id', "Default"))
                circ.anchored = w.get('anchored', [False])
                if 'anim' in w: circ.anim = w['anim']
                self.sketch.entities.append(circ)

        # 2. Import Constraints (Offset indices)
        from model.constraints import create_constraint
        for c_data in data.get('constraints', []):
            # Remap indices
            new_indices = []
            if isinstance(c_data['indices'], list):
                if isinstance(c_data['indices'][0], list): # Nested (Entity, Pt)
                    for item in c_data['indices']:
                        if isinstance(item, list):
                            new_indices.append([item[0] + base_idx, item[1]])
                        else:
                            new_indices.append(item + base_idx)
                else: # Simple List
                     for item in c_data['indices']:
                         new_indices.append(item + base_idx)
            
            c_data_copy = c_data.copy()
            c_data_copy['indices'] = new_indices
            
            c = create_constraint(c_data_copy)
            if c:
                if c.base_time is not None: c.base_time = current_time
                self.sketch.constraints.append(c)