import math
import numpy as np
from geometry import Line, Circle
from constraints import Length, EqualLength, Angle, Midpoint, Coincident, Collinear, FixedAngle

def get_l(s, w): 
    return np.linalg.norm(s.walls[w].end - s.walls[w].start)

def get_angle(s, w1, w2):
    l1 = s.walls[w1]; l2 = s.walls[w2]
    v1 = l1.end - l1.start
    v2 = l2.end - l2.start
    return math.degrees(math.atan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1]))

CONSTRAINT_DEFS = {
    'LENGTH':   [{'w':1, 'p':0, 't':Line, 'msg':"Select 1 Line", 'f': lambda s,w,p: Length(w[0], get_l(s, w[0]))}],
    'EQUAL':    [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: EqualLength(w[0], w[1])}],
    'PARALLEL': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PARALLEL', w[0], w[1])}],
    'PERPENDICULAR': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PERPENDICULAR', w[0], w[1])}],
    'HORIZONTAL': [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('HORIZONTAL', w[0]), 'multi':True}],
    'VERTICAL':   [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('VERTICAL', w[0]), 'multi':True}],
    'MIDPOINT':   [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Midpoint(p[0][0], p[0][1], w[0])}],
    'COLLINEAR':  [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Collinear(p[0][0], p[0][1], w[0])}],
    'ANGLE':      [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: FixedAngle(w[0], w[1], get_angle(s, w[0], w[1]))}],
    'COINCIDENT': [
        {'w':0, 'p':2, 't':None, 'msg':"Select 2 Points", 'f': lambda s,w,p: Coincident(p[0][0], p[0][1], p[1][0], p[1][1])},
        {'w':1, 'p':1, 't':(Line, Circle), 'msg':"Select Point & Entity", 'f': lambda s,w,p: Coincident(p[0][0], p[0][1], w[0], -1)}
    ]
}