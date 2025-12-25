"""
Geometry - Core Geometric Entities

This module defines the basic geometric primitives:
- Point: A single position with optional anchoring
- Line: A line segment between two points
- Circle: A circle defined by center and radius

All entities support:
- Serialization (to_dict / from_dict)
- Constraint solving (get_point / set_point / get_inv_mass)
- Movement (move)
- Material assignment
"""

import numpy as np


class Entity:
    """Base class for all geometric entities."""
    
    def __init__(self, material_id="Default"):
        self.material_id = material_id
        self.anim = None  # Animation data (for drivers)

    def get_point(self, index):
        """Get the position of a point by index."""
        raise NotImplementedError

    def set_point(self, index, pos):
        """Set the position of a point by index."""
        raise NotImplementedError

    def get_inv_mass(self, index):
        """Get the inverse mass (0 = anchored/infinite mass)."""
        raise NotImplementedError
        
    def move(self, dx, dy, indices=None):
        """Move the entity by (dx, dy)."""
        raise NotImplementedError
    
    def to_dict(self):
        """Serialize to dictionary."""
        raise NotImplementedError


class Point(Entity):
    """
    A single point in space.
    
    Attributes:
        pos: numpy array [x, y]
        anchored: If True, point is fixed during constraint solving
        is_handle: If True, this Point is a ProcessObject handle
                   (skip during atomization, special deletion behavior)
    """
    
    def __init__(self, x, y, anchored=False, material_id="Default"):
        super().__init__(material_id)
        self.pos = np.array([x, y], dtype=np.float64)
        self.anchored = anchored
        
        # Handle system for ProcessObjects
        self.is_handle = False
        self._owner_process_object = None  # Set by ProcessObject.register_handles()

    def get_point(self, index):
        return self.pos

    def set_point(self, index, pos):
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]

    def get_inv_mass(self, index):
        return 0.0 if self.anchored else 1.0

    def move(self, dx, dy, indices=None):
        if not self.anchored:
            self.pos[0] += dx
            self.pos[1] += dy

    def to_dict(self):
        d = {
            'type': 'point', 
            'x': float(self.pos[0]), 
            'y': float(self.pos[1]), 
            'anchored': self.anchored, 
            'material_id': self.material_id
        }
        # Note: is_handle is NOT serialized here
        # ProcessObjects are serialized separately and recreate their handles
        return d

    @staticmethod
    def from_dict(data):
        p = Point(
            data['x'], 
            data['y'], 
            data.get('anchored', False), 
            data.get('material_id', "Default")
        )
        return p


class Line(Entity):
    """
    A line segment defined by start and end points.
    
    Attributes:
        start: numpy array [x, y] for start point
        end: numpy array [x, y] for end point
        ref: If True, this is a reference/construction line (not atomized)
        anchored: List [start_anchored, end_anchored]
    """
    
    def __init__(self, start, end, is_ref=False, material_id="Default"):
        super().__init__(material_id)
        self.start = np.array(start, dtype=np.float64)
        self.end = np.array(end, dtype=np.float64)
        self.ref = is_ref
        self.anchored = [False, False]

    def length(self):
        """Calculate the length of the line segment."""
        return np.linalg.norm(self.end - self.start)

    def get_point(self, index):
        return self.start if index == 0 else self.end

    def set_point(self, index, pos):
        if index == 0:
            self.start[:] = pos
        else:
            self.end[:] = pos

    def get_inv_mass(self, index):
        return 0.0 if self.anchored[index] else 1.0

    def move(self, dx, dy, indices=None):
        if indices is None:
            indices = [0, 1]
        if 0 in indices and not self.anchored[0]:
            self.start[0] += dx
            self.start[1] += dy
        if 1 in indices and not self.anchored[1]:
            self.end[0] += dx
            self.end[1] += dy

    def to_dict(self):
        d = {
            'type': 'line', 
            'start': self.start.tolist(), 
            'end': self.end.tolist(),
            'ref': self.ref, 
            'anchored': self.anchored,
            'material_id': self.material_id
        }
        if self.anim:
            d['anim'] = self.anim
        return d

    @staticmethod
    def from_dict(data):
        mat_id = data.get('material_id', "Default")
        l = Line(data['start'], data['end'], data.get('ref', False), mat_id)
        l.anchored = data.get('anchored', [False, False])
        l.anim = data.get('anim', None)
        return l


class Circle(Entity):
    """
    A circle defined by center and radius.
    
    Attributes:
        center: numpy array [x, y]
        radius: float
        anchored: List [center_anchored] (single element for consistency)
    """
    
    def __init__(self, center, radius, material_id="Default"):
        super().__init__(material_id)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.anchored = [False]

    def get_point(self, index):
        return self.center

    def set_point(self, index, pos):
        if index == 0:
            self.center[:] = pos

    def get_inv_mass(self, index):
        return 0.0 if self.anchored[0] else 1.0

    def move(self, dx, dy, indices=None):
        if not self.anchored[0]:
            self.center[0] += dx
            self.center[1] += dy

    def to_dict(self):
        d = {
            'type': 'circle', 
            'center': self.center.tolist(), 
            'radius': self.radius,
            'anchored': self.anchored,
            'material_id': self.material_id
        }
        if self.anim:
            d['anim'] = self.anim
        return d

    @staticmethod
    def from_dict(data):
        c = Circle(data['center'], data['radius'], data.get('material_id', "Default"))
        c.anchored = data.get('anchored', [False])
        c.anim = data.get('anim', None)
        return c
