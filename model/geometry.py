import numpy as np

class Entity:
    def __init__(self, material_id="Default"):
        self.material_id = material_id
        self.anim = None 

    def get_point(self, index):
        raise NotImplementedError

    def set_point(self, index, pos):
        raise NotImplementedError

    def get_inv_mass(self, index):
        raise NotImplementedError
        
    def move(self, dx, dy, indices=None):
        raise NotImplementedError
    
    def to_dict(self):
        raise NotImplementedError

class Point(Entity):
    def __init__(self, x, y, anchored=False, material_id="Default"):
        super().__init__(material_id)
        self.pos = np.array([x, y], dtype=np.float64)
        self.anchored = anchored

    def get_point(self, index):
        return self.pos

    def set_point(self, index, pos):
        self.pos[0] = pos[0]; self.pos[1] = pos[1]

    def get_inv_mass(self, index):
        return 0.0 if self.anchored else 1.0

    def move(self, dx, dy, indices=None):
        if not self.anchored:
            self.pos[0] += dx; self.pos[1] += dy

    def to_dict(self):
        return {
            'type': 'point', 
            'x': float(self.pos[0]), 'y': float(self.pos[1]), 
            'anchored': self.anchored, 
            'material_id': self.material_id
        }

    @staticmethod
    def from_dict(data):
        p = Point(data['x'], data['y'], data.get('anchored', False), data.get('material_id', "Default"))
        return p

class Line(Entity):
    def __init__(self, start, end, is_ref=False, material_id="Default"):
        super().__init__(material_id)
        self.start = np.array(start, dtype=np.float64)
        self.end = np.array(end, dtype=np.float64)
        self.ref = is_ref
        self.anchored = [False, False]

    def length(self):
        return np.linalg.norm(self.end - self.start)

    def get_point(self, index):
        return self.start if index == 0 else self.end

    def set_point(self, index, pos):
        if index == 0: self.start[:] = pos
        else: self.end[:] = pos

    def get_inv_mass(self, index):
        return 0.0 if self.anchored[index] else 1.0

    def move(self, dx, dy, indices=None):
        if indices is None: indices = [0, 1]
        if 0 in indices and not self.anchored[0]:
            self.start[0] += dx; self.start[1] += dy
        if 1 in indices and not self.anchored[1]:
            self.end[0] += dx; self.end[1] += dy

    def to_dict(self):
        d = {
            'type': 'line', 
            'start': self.start.tolist(), 
            'end': self.end.tolist(),
            'ref': self.ref, 
            'anchored': self.anchored,
            'material_id': self.material_id
        }
        if self.anim: d['anim'] = self.anim
        return d

    @staticmethod
    def from_dict(data):
        mat_id = data.get('material_id', "Default")
        l = Line(data['start'], data['end'], data.get('ref', False), mat_id)
        l.anchored = data.get('anchored', [False, False])
        l.anim = data.get('anim', None)
        return l

class Circle(Entity):
    def __init__(self, center, radius, material_id="Default"):
        super().__init__(material_id)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.anchored = [False] 

    def get_point(self, index):
        return self.center

    def set_point(self, index, pos):
        if index == 0: self.center[:] = pos

    def get_inv_mass(self, index):
        return 0.0 if self.anchored[0] else 1.0

    def move(self, dx, dy, indices=None):
        if not self.anchored[0]:
            self.center[0] += dx; self.center[1] += dy

    def to_dict(self):
        d = {
            'type': 'circle', 
            'center': self.center.tolist(), 
            'radius': self.radius,
            'anchored': self.anchored,
            'material_id': self.material_id
        }
        if self.anim: d['anim'] = self.anim
        return d

    @staticmethod
    def from_dict(data):
        c = Circle(data['center'], data['radius'], data.get('material_id', "Default"))
        c.anchored = data.get('anchored', [False])
        c.anim = data.get('anim', None)
        return c