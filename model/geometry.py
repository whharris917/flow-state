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
- Rigid body dynamics (mass, velocity, forces) when dynamic=True
"""

import numpy as np
import core.config as config


class Entity:
    """
    Base class for all geometric entities.

    Rigid Body State (active when dynamic=True):
        mass: Total mass of the rigid body
        velocity: Linear velocity [vx, vy]
        angular_vel: Angular velocity (radians/sec)
        force_accum: Accumulated force this frame [fx, fy]
        torque_accum: Accumulated torque this frame
        damping: Linear velocity damping factor
        angular_damping: Angular velocity damping factor
    """

    def __init__(self, material_id="Default"):
        self.material_id = material_id
        self.anim = None  # Animation data (for drivers)
        self.physical = False  # If True, entity is atomized by compiler

        # --- Rigid Body Dynamics ---
        self.dynamic = False  # If True, entity responds to physics forces
        self.mass = 1.0  # Total mass (will be computed from atom count if auto)
        self.velocity = np.zeros(2, dtype=np.float64)  # Linear velocity [vx, vy]
        self.angular_vel = 0.0  # Angular velocity (radians/sec)
        self.force_accum = np.zeros(2, dtype=np.float64)  # Accumulated force [fx, fy]
        self.torque_accum = 0.0  # Accumulated torque
        self.damping = config.ENTITY_DAMPING  # Linear velocity damping (per frame)
        self.angular_damping = config.ENTITY_ANGULAR_DAMPING  # Angular velocity damping

    @property
    def inv_mass(self):
        """Inverse mass (0 if infinite/static)."""
        if self.mass <= 0 or not self.dynamic:
            return 0.0
        return 1.0 / self.mass

    @property
    def inertia(self):
        """Moment of inertia. Override in subclasses for accurate calculation."""
        return self.mass  # Default: treat as point mass

    @property
    def inv_inertia(self):
        """Inverse moment of inertia."""
        i = self.inertia
        if i <= 0 or not self.dynamic:
            return 0.0
        return 1.0 / i

    def get_center_of_mass(self):
        """Get center of mass. Override in subclasses."""
        raise NotImplementedError

    def apply_velocity(self, dt):
        """Apply current velocity to geometry. Override in subclasses."""
        raise NotImplementedError

    def clear_accumulators(self):
        """Reset force and torque accumulators for next frame."""
        self.force_accum[:] = 0
        self.torque_accum = 0

    def apply_force(self, fx, fy):
        """Add force to the accumulator."""
        self.force_accum[0] += fx
        self.force_accum[1] += fy

    def apply_torque(self, torque):
        """Add torque to the accumulator."""
        self.torque_accum += torque

    # Velocity clamp limits to prevent physics explosions
    MAX_LINEAR_VELOCITY = 1000.0
    MAX_ANGULAR_VELOCITY = 50.0

    def integrate(self, dt):
        """
        Integrate forces and torques into velocity, then apply velocity to geometry.

        This is a simple semi-implicit Euler integration:
        1. v += (F/m) * dt
        2. ω += (τ/I) * dt
        3. Clamp velocities to safe limits
        4. Apply velocity to geometry via apply_velocity()

        Only affects dynamic entities. Static entities ignore this call.
        """
        if not self.dynamic:
            return

        # Linear acceleration: a = F / m
        if self.inv_mass > 0:
            self.velocity[0] += self.force_accum[0] * self.inv_mass * dt
            self.velocity[1] += self.force_accum[1] * self.inv_mass * dt

        # Angular acceleration: α = τ / I
        if self.inv_inertia > 0:
            self.angular_vel += self.torque_accum * self.inv_inertia * dt

        # NaN safety check - reset if values exploded
        if (np.isnan(self.velocity[0]) or np.isnan(self.velocity[1]) or
            np.isnan(self.angular_vel)):
            print(f"WARNING: NaN detected in entity velocity, resetting to 0")
            self.velocity[:] = 0
            self.angular_vel = 0.0
            return

        # Clamp velocities to prevent physics explosions
        self.velocity[0] = np.clip(self.velocity[0],
                                   -self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY)
        self.velocity[1] = np.clip(self.velocity[1],
                                   -self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY)
        self.angular_vel = np.clip(self.angular_vel,
                                   -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

        # Apply velocity to geometry
        self.apply_velocity(dt)

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
        if self.anchored:
            return 0.0
        if self.dynamic:
            return self.inv_mass
        return 1.0

    def get_center_of_mass(self):
        return self.pos.copy()

    def apply_velocity(self, dt):
        """Apply velocity to position (simple point mass)."""
        if not self.anchored and self.dynamic:
            self.pos += self.velocity * dt
            # Apply damping
            self.velocity *= self.damping

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
        if self.physical:
            d['physical'] = self.physical
        if self.dynamic:
            d['dynamic'] = self.dynamic
            d['mass'] = self.mass
            d['velocity'] = self.velocity.tolist()
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
        p.physical = data.get('physical', False)
        p.dynamic = data.get('dynamic', False)
        if p.dynamic:
            p.mass = data.get('mass', 1.0)
            if 'velocity' in data:
                p.velocity = np.array(data['velocity'], dtype=np.float64)
        return p


class Line(Entity):
    """
    A line segment defined by start and end points.

    Attributes:
        start: numpy array [x, y] for start point
        end: numpy array [x, y] for end point
        ref: If True, this is a reference/construction line (not atomized)
        anchored: List [start_anchored, end_anchored]
        infinite: If True (and ref=True), line extends to world boundaries when rendered

    Rigid Body:
        When dynamic=True, the line behaves as a rigid rod with mass and inertia.
        Inertia is computed as (1/12) * mass * length^2 (uniform rod formula).
    """

    def __init__(self, start, end, is_ref=False, material_id="Default"):
        super().__init__(material_id)
        self.start = np.array(start, dtype=np.float64)
        self.end = np.array(end, dtype=np.float64)
        self.ref = is_ref
        self.anchored = [False, False]
        self.infinite = False

    def length(self):
        """Calculate the length of the line segment."""
        return np.linalg.norm(self.end - self.start)

    # Minimum inertia to prevent division by zero / instability
    MIN_INERTIA = 1.0

    @property
    def inertia(self):
        """Moment of inertia for a uniform rod about its center: I = (1/12) * m * L^2"""
        L = self.length()
        calculated = (1.0 / 12.0) * self.mass * L * L * config.INERTIA_STABILITY_FACTOR
        return max(calculated, self.MIN_INERTIA)

    def get_center_of_mass(self):
        """Center of mass is the midpoint of the line."""
        return (self.start + self.end) * 0.5

    def get_point(self, index):
        return self.start if index == 0 else self.end

    def set_point(self, index, pos):
        if index == 0:
            self.start[:] = pos
        else:
            self.end[:] = pos

    def get_inv_mass(self, index):
        if self.anchored[index]:
            return 0.0
        if self.dynamic:
            return self.inv_mass
        return 1.0

    def apply_velocity(self, dt):
        """
        Apply linear and angular velocity to line endpoints.

        Linear velocity moves the center of mass.
        Angular velocity rotates both endpoints around the COM.
        """
        if not self.dynamic:
            return

        com = self.get_center_of_mass()

        # Apply linear velocity to COM
        delta = self.velocity * dt

        # Apply angular velocity (rotation around COM)
        theta = self.angular_vel * dt
        if abs(theta) > 1e-9:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Rotate start around COM
            if not self.anchored[0]:
                r = self.start - com
                self.start[0] = com[0] + r[0] * cos_t - r[1] * sin_t + delta[0]
                self.start[1] = com[1] + r[0] * sin_t + r[1] * cos_t + delta[1]
            elif not self.anchored[1]:
                # If start is anchored, only apply delta to end
                pass

            # Rotate end around COM
            if not self.anchored[1]:
                r = self.end - com
                self.end[0] = com[0] + r[0] * cos_t - r[1] * sin_t + delta[0]
                self.end[1] = com[1] + r[0] * sin_t + r[1] * cos_t + delta[1]
        else:
            # No rotation, just translate
            if not self.anchored[0]:
                self.start += delta
            if not self.anchored[1]:
                self.end += delta

        # Apply damping
        self.velocity *= self.damping
        self.angular_vel *= self.angular_damping

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
        if self.physical:
            d['physical'] = self.physical
        if self.infinite:
            d['infinite'] = self.infinite
        if self.anim:
            d['anim'] = self.anim
        if self.dynamic:
            d['dynamic'] = self.dynamic
            d['mass'] = self.mass
            d['velocity'] = self.velocity.tolist()
            d['angular_vel'] = self.angular_vel
        return d

    @staticmethod
    def from_dict(data):
        mat_id = data.get('material_id', "Default")
        l = Line(data['start'], data['end'], data.get('ref', False), mat_id)
        l.anchored = data.get('anchored', [False, False])
        l.physical = data.get('physical', False)
        l.infinite = data.get('infinite', False)
        l.anim = data.get('anim', None)
        l.dynamic = data.get('dynamic', False)
        if l.dynamic:
            l.mass = data.get('mass', 1.0)
            if 'velocity' in data:
                l.velocity = np.array(data['velocity'], dtype=np.float64)
            l.angular_vel = data.get('angular_vel', 0.0)
        return l


class Circle(Entity):
    """
    A circle defined by center and radius.

    Attributes:
        center: numpy array [x, y]
        radius: float
        anchored: List [center_anchored] (single element for consistency)

    Rigid Body:
        When dynamic=True, the circle behaves as a disk/ring with mass and inertia.
        Inertia is computed as (1/2) * mass * radius^2 (solid disk formula).
    """

    # Minimum inertia to prevent division by zero / instability
    MIN_INERTIA = 1.0

    def __init__(self, center, radius, material_id="Default"):
        super().__init__(material_id)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.anchored = [False]

    @property
    def inertia(self):
        """Moment of inertia for a solid disk about its center: I = (1/2) * m * r^2"""
        calculated = 0.5 * self.mass * self.radius * self.radius * config.INERTIA_STABILITY_FACTOR
        return max(calculated, self.MIN_INERTIA)

    def get_center_of_mass(self):
        """Center of mass is the center of the circle."""
        return self.center.copy()

    def get_point(self, index):
        return self.center

    def set_point(self, index, pos):
        if index == 0:
            self.center[:] = pos

    def get_inv_mass(self, index):
        if self.anchored[0]:
            return 0.0
        if self.dynamic:
            return self.inv_mass
        return 1.0

    def apply_velocity(self, dt):
        """
        Apply linear velocity to circle center.
        Angular velocity affects tethered atom positions but not circle geometry.
        """
        if not self.dynamic:
            return

        if not self.anchored[0]:
            self.center += self.velocity * dt

        # Apply damping
        self.velocity *= self.damping
        self.angular_vel *= self.angular_damping

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
        if self.physical:
            d['physical'] = self.physical
        if self.anim:
            d['anim'] = self.anim
        if self.dynamic:
            d['dynamic'] = self.dynamic
            d['mass'] = self.mass
            d['velocity'] = self.velocity.tolist()
            d['angular_vel'] = self.angular_vel
        return d

    @staticmethod
    def from_dict(data):
        c = Circle(data['center'], data['radius'], data.get('material_id', "Default"))
        c.anchored = data.get('anchored', [False])
        c.physical = data.get('physical', False)
        c.anim = data.get('anim', None)
        c.dynamic = data.get('dynamic', False)
        if c.dynamic:
            c.mass = data.get('mass', 1.0)
            if 'velocity' in data:
                c.velocity = np.array(data['velocity'], dtype=np.float64)
            c.angular_vel = data.get('angular_vel', 0.0)
        return c
