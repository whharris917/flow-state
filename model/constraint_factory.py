"""
Constraint Factory - Domain Logic for Constraint Creation

This module contains the constraint definition rules (CONSTRAINT_DEFS) used
by the ConstraintBuilder to create constraints based on user selections.

Moved from core/definitions.py as part of CR-2026-001 to place domain logic
in the model package where it belongs.

Usage:
    from model.constraint_factory import CONSTRAINT_DEFS

    rules = CONSTRAINT_DEFS.get('LENGTH', [])
    for rule in rules:
        if matches_selection(rule, walls, points):
            constraint = rule['f'](sketch, walls, points)
"""

import math
import numpy as np
from model.geometry import Line, Circle
from model.constraints import (
    Length, EqualLength, Angle, Midpoint, Coincident, Collinear, FixedAngle, Radius
)


def get_l(s, w):
    """Get length of a line entity."""
    return np.linalg.norm(s.walls[w].end - s.walls[w].start)


def get_r(s, w):
    """Get radius of a circle entity."""
    return s.walls[w].radius


def get_angle(s, w1, w2):
    """Get angle between two line entities in degrees."""
    l1 = s.walls[w1]
    l2 = s.walls[w2]
    v1 = l1.end - l1.start
    v2 = l2.end - l2.start
    return math.degrees(math.atan2(
        v1[0] * v2[1] - v1[1] * v2[0],
        v1[0] * v2[0] + v1[1] * v2[1]
    ))


# Constraint definition rules - maps constraint types to selection requirements
# Each rule specifies:
#   w: number of wall (entity) selections required
#   p: number of point selections required
#   t: required entity type(s) or None
#   msg: user prompt message
#   f: factory function (sketch, walls, points) -> Constraint
#   multi: if True, can apply to multiple entities
#   binary: if True, uses master/follower pattern
CONSTRAINT_DEFS = {
    # LENGTH is context-aware: applies Length to Lines, Radius to Circles
    'LENGTH': [
        {
            'w': 1, 'p': 0, 't': Line,
            'msg': "Select Line(s) or Circle(s)",
            'f': lambda s, w, p: Length(w[0], get_l(s, w[0])),
            'multi': True
        },
        {
            'w': 1, 'p': 0, 't': Circle,
            'msg': "Select Line(s) or Circle(s)",
            'f': lambda s, w, p: Radius(w[0], get_r(s, w[0])),
            'multi': True
        },
    ],
    'RADIUS': [
        {
            'w': 1, 'p': 0, 't': Circle,
            'msg': "Select 1 Circle",
            'f': lambda s, w, p: Radius(w[0], get_r(s, w[0]))
        }
    ],
    # EQUAL and PARALLEL use master/follower pattern: first selected is master
    'EQUAL': [
        {
            'w': 2, 'p': 0, 't': Line,
            'msg': "Select 2+ Lines",
            'f': lambda s, w, p: EqualLength(w[0], w[1]),
            'multi': True, 'binary': True
        }
    ],
    'PARALLEL': [
        {
            'w': 2, 'p': 0, 't': Line,
            'msg': "Select 2+ Lines",
            'f': lambda s, w, p: Angle('PARALLEL', w[0], w[1]),
            'multi': True, 'binary': True
        }
    ],
    'PERPENDICULAR': [
        {
            'w': 2, 'p': 0, 't': Line,
            'msg': "Select 2 Lines",
            'f': lambda s, w, p: Angle('PERPENDICULAR', w[0], w[1])
        }
    ],
    'HORIZONTAL': [
        {
            'w': 1, 'p': 0, 't': Line,
            'msg': "Select Line(s)",
            'f': lambda s, w, p: Angle('HORIZONTAL', w[0]),
            'multi': True
        }
    ],
    'VERTICAL': [
        {
            'w': 1, 'p': 0, 't': Line,
            'msg': "Select Line(s)",
            'f': lambda s, w, p: Angle('VERTICAL', w[0]),
            'multi': True
        }
    ],
    'MIDPOINT': [
        {
            'w': 1, 'p': 1, 't': Line,
            'msg': "Select Line & Point",
            'f': lambda s, w, p: Midpoint(p[0][0], p[0][1], w[0])
        }
    ],
    'COLLINEAR': [
        {
            'w': 1, 'p': 1, 't': Line,
            'msg': "Select Line & Point",
            'f': lambda s, w, p: Collinear(p[0][0], p[0][1], w[0])
        }
    ],
    'ANGLE': [
        {
            'w': 2, 'p': 0, 't': Line,
            'msg': "Select 2 Lines",
            'f': lambda s, w, p: FixedAngle(w[0], w[1], get_angle(s, w[0], w[1]))
        }
    ],
    'COINCIDENT': [
        {
            'w': 0, 'p': 2, 't': None,
            'msg': "Select 2 Points",
            'f': lambda s, w, p: Coincident(p[0][0], p[0][1], p[1][0], p[1][1])
        },
        {
            'w': 1, 'p': 1, 't': (Line, Circle),
            'msg': "Select Point & Entity",
            'f': lambda s, w, p: Coincident(p[0][0], p[0][1], w[0], -1)
        }
    ]
}
