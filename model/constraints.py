# Pure Data Models for Constraints
# Rendering logic moved to renderer.py

class Constraint:
    def __init__(self, type_name):
        self.type = type_name
        self.temp = False 
        self.driver = None 
        self.base_value = None 
        self.base_time = None
        self.indices = []

    def to_dict(self):
        d = {'type': self.type, 'indices': self.indices}
        if hasattr(self, 'value'): d['value'] = self.value
        if self.driver: d['driver'] = self.driver
        if self.base_value is not None: d['base_value'] = self.base_value
        if self.base_time is not None: d['base_time'] = self.base_time
        return d

class Coincident(Constraint):
    def __init__(self, entity_idx1, pt_idx1, entity_idx2, pt_idx2):
        super().__init__('COINCIDENT')
        self.indices = [(entity_idx1, pt_idx1), (entity_idx2, pt_idx2)]

class Collinear(Constraint):
    def __init__(self, pt_wall_idx, pt_idx, line_wall_idx):
        super().__init__('COLLINEAR')
        self.indices = [(pt_wall_idx, pt_idx), line_wall_idx]

class Midpoint(Constraint):
    def __init__(self, pt_wall_idx, pt_idx, line_wall_idx):
        super().__init__('MIDPOINT')
        self.indices = [(pt_wall_idx, pt_idx), line_wall_idx]

class Length(Constraint):
    def __init__(self, entity_idx, target_length):
        super().__init__('LENGTH')
        self.indices = [entity_idx]
        self.value = target_length


class Radius(Constraint):
    def __init__(self, entity_idx, target_radius):
        super().__init__('RADIUS')
        self.indices = [entity_idx]
        self.value = target_radius

class EqualLength(Constraint):
    def __init__(self, entity_idx1, entity_idx2):
        super().__init__('EQUAL')
        self.indices = [entity_idx1, entity_idx2]

class Angle(Constraint):
    def __init__(self, type_name, entity_idx1, entity_idx2=None):
        super().__init__(type_name) 
        if entity_idx2 is not None: self.indices = [entity_idx1, entity_idx2]
        else: self.indices = [entity_idx1]

class FixedAngle(Constraint):
    def __init__(self, w1_idx, w2_idx, angle_deg):
        super().__init__('ANGLE')
        self.indices = [w1_idx, w2_idx]
        self.value = angle_deg

def create_constraint(data):
    t = data['type']
    idx = data['indices']
    
    c = None
    if t == 'COINCIDENT': c = Coincident(idx[0][0], idx[0][1], idx[1][0], idx[1][1])
    elif t == 'COLLINEAR': c = Collinear(idx[0][0], idx[0][1], idx[1])
    elif t == 'MIDPOINT': c = Midpoint(idx[0][0], idx[0][1], idx[1])
    elif t == 'LENGTH': c = Length(idx[0], data['value'])
    elif t == 'RADIUS': c = Radius(idx[0], data['value'])
    elif t == 'EQUAL': c = EqualLength(idx[0], idx[1])
    elif t == 'ANGLE': c = FixedAngle(idx[0], idx[1], data['value'])
    elif t in ['HORIZONTAL', 'VERTICAL', 'PARALLEL', 'PERPENDICULAR']:
        if len(idx) == 2: c = Angle(t, idx[0], idx[1])
        else: c = Angle(t, idx[0])
        
    if c:
        if 'driver' in data: c.driver = data['driver']
        if 'base_value' in data: c.base_value = data['base_value']
        if 'base_time' in data: c.base_time = data['base_time']
    
    return c