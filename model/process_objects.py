"""
ProcessObject - Simulation Boundary Conditions

ProcessObjects represent active boundary conditions in the simulation:
- Source: Particle emitters
- Sink: Particle absorbers (future)
- Heater: Temperature control zones (future)
- Sensor: Measurement points (future)

Architecture:
- ProcessObjects sit ALONGSIDE Sketch, NOT inside it
- They expose "handles" (Points) to Sketch for constraint participation
- Handles are regular Points with is_handle=True flag
- Compiler skips handle Points during atomization
- Scene calls execute() on all ProcessObjects each frame

Handle System:
- Handles ARE Sketch entities (Points with is_handle=True)
- They participate in constraints like any other Point
- Deleting a handle deletes the entire owning ProcessObject
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import random
import math
import numpy as np

from model.geometry import Point

if TYPE_CHECKING:
    from model.sketch import Sketch
    from engine.simulation import Simulation


@dataclass
class SourceProperties:
    """
    Properties for a particle Source.
    
    Particle Properties:
        sigma: Particle size parameter (LJ sigma)
        epsilon: Interaction strength (LJ epsilon)
        mass: Particle mass
    
    Injection Behavior:
        rate: Target particles per second
        temperature: For Maxwell-Boltzmann velocity sampling
        injection_direction: Preferred direction in radians (0 = right)
        injection_spread: Angular spread (2*pi = isotropic)
    """
    # Particle properties
    sigma: float = 1.0
    epsilon: float = 1.0
    mass: float = 1.0
    
    # Injection behavior
    rate: float = 10.0              # particles per second (target rate)
    temperature: float = 1.0        # for velocity sampling
    injection_direction: float = 0  # angle in radians, 0 = no bias (right)
    injection_spread: float = 2 * math.pi  # angular spread (2π = isotropic)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'sigma': self.sigma,
            'epsilon': self.epsilon,
            'mass': self.mass,
            'rate': self.rate,
            'temperature': self.temperature,
            'injection_direction': self.injection_direction,
            'injection_spread': self.injection_spread,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SourceProperties':
        """Deserialize from dictionary."""
        return SourceProperties(
            sigma=data.get('sigma', 1.0),
            epsilon=data.get('epsilon', 1.0),
            mass=data.get('mass', 1.0),
            rate=data.get('rate', 10.0),
            temperature=data.get('temperature', 1.0),
            injection_direction=data.get('injection_direction', 0),
            injection_spread=data.get('injection_spread', 2 * math.pi),
        )


class ProcessObject(ABC):
    """
    Base class for simulation boundary conditions.
    
    ProcessObjects:
    - Sit alongside Sketch (not inside it)
    - Expose "handles" (Points) for constraint participation
    - Execute each frame to affect the simulation
    - Have their own non-handle geometry for rendering
    """
    
    def __init__(self):
        self.handles: Dict[str, Point] = {}  # name → Point
        self.enabled: bool = True
        self._owner_scene = None  # Set when added to scene
    
    @abstractmethod
    def execute(self, simulation: 'Simulation', dt: float) -> None:
        """
        Called each frame by Scene when simulation is running.
        
        Args:
            simulation: The Simulation instance to affect
            dt: Time step in seconds
        """
        pass
    
    @abstractmethod
    def get_geometry_for_rendering(self) -> List[dict]:
        """
        Returns non-handle geometry for rendering.
        
        Handles are rendered as normal Sketch Points.
        This returns additional geometry like dashed circles.
        
        Returns:
            List of geometry descriptors for the renderer
        """
        pass
    
    def register_handles(self, sketch: 'Sketch') -> None:
        """
        Add handle points to sketch for constraint participation.
        
        Args:
            sketch: The Sketch instance to register handles with
        """
        for name, point in self.handles.items():
            point.is_handle = True
            point._owner_process_object = self
            sketch.entities.append(point)
    
    def unregister_handles(self, sketch: 'Sketch') -> None:
        """
        Remove handle points and their constraints from sketch.
        
        Args:
            sketch: The Sketch instance to unregister handles from
        """
        for name, point in self.handles.items():
            if point in sketch.entities:
                idx = sketch.entities.index(point)
                sketch.remove_entity(idx)
    
    def get_handle_indices(self, sketch: 'Sketch') -> Dict[str, int]:
        """
        Get the entity indices of handles in the sketch.
        
        Args:
            sketch: The Sketch to search
            
        Returns:
            Dict mapping handle name to entity index
        """
        indices = {}
        for name, point in self.handles.items():
            if point in sketch.entities:
                indices[name] = sketch.entities.index(point)
        return indices
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to dictionary for save/load."""
        pass
    
    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> 'ProcessObject':
        """Deserialize from dictionary."""
        pass


class Source(ProcessObject):
    """
    Particle emitter - spawns particles within a circular region.
    
    The Source has:
    - A center handle (Point) that can be constrained
    - A radius defining the spawn region
    - Properties controlling particle spawning
    
    Spawn Algorithm:
    - Accumulator-based to maintain target rate
    - Rejection sampling for non-overlapping placement
    - Adaptive attempts when behind target rate
    - Hard cap of 50 particles/frame to prevent lag
    """
    
    MAX_SPAWNS_PER_FRAME = 50  # Hard cap to prevent frame drops
    BASE_ATTEMPTS = 5         # Base attempts per spawn
    ROLLING_WINDOW = 2.0      # Seconds for rolling average
    
    def __init__(self, center: tuple, radius: float, 
                 properties: SourceProperties = None):
        """
        Create a new Source.
        
        Args:
            center: (x, y) tuple for center position
            radius: Spawn region radius
            properties: SourceProperties or None for defaults
        """
        super().__init__()
        
        self.radius = float(radius)
        self.properties = properties or SourceProperties()
        
        # Spawn tracking
        self._spawn_accumulator = 0.0
        self._recent_spawns: List[tuple] = []  # [(timestamp, count), ...]
        
        # Create the center handle
        center_point = Point(center[0], center[1], anchored=False)
        center_point.is_handle = True
        self.handles['center'] = center_point
    
    @property
    def center(self) -> Point:
        """Get the center handle point."""
        return self.handles['center']
    
    @property
    def x(self) -> float:
        """Get center x coordinate."""
        return self.center.pos[0]
    
    @property
    def y(self) -> float:
        """Get center y coordinate."""
        return self.center.pos[1]
    
    def execute(self, simulation: 'Simulation', dt: float) -> None:
        """
        Spawn particles according to rate with adaptive algorithm.
        
        Args:
            simulation: The Simulation to add particles to
            dt: Time step in seconds
        """
        if not self.enabled:
            return
        
        import time
        current_time = time.time()
        
        # Accumulate spawn credit
        self._spawn_accumulator += self.properties.rate * dt
        
        # Calculate adaptive attempts based on how far behind we are
        catchup_factor = self._calculate_catchup_factor(current_time)
        max_attempts = int(self.BASE_ATTEMPTS * catchup_factor)
        
        spawned = 0
        attempts = 0
        
        while self._spawn_accumulator >= 1.0 and spawned < self.MAX_SPAWNS_PER_FRAME:
            if self._try_spawn_particle(simulation):
                self._spawn_accumulator -= 1.0
                spawned += 1
            attempts += 1
            if attempts >= max_attempts:
                break
        
        # Record spawn for rolling average
        self._record_spawn(current_time, spawned)
    
    def _calculate_catchup_factor(self, current_time: float) -> float:
        """
        Calculate how aggressively to try spawning based on recent performance.
        
        Returns:
            Multiplier for attempt count (1.0 = normal, higher = more aggressive)
        """
        # Clean old entries
        cutoff = current_time - self.ROLLING_WINDOW
        self._recent_spawns = [(t, c) for t, c in self._recent_spawns if t > cutoff]
        
        if not self._recent_spawns:
            return 1.0
        
        # Calculate actual rate vs target rate
        total_spawned = sum(c for _, c in self._recent_spawns)
        elapsed = current_time - self._recent_spawns[0][0] if self._recent_spawns else 1.0
        elapsed = max(elapsed, 0.01)  # Prevent division by zero
        
        actual_rate = total_spawned / elapsed
        target_rate = self.properties.rate
        
        if target_rate <= 0:
            return 1.0
        
        ratio = actual_rate / target_rate
        
        # If we're at 50% or less of target, double the attempts
        if ratio < 0.5:
            return 2.0
        elif ratio < 0.8:
            return 1.5
        else:
            return 1.0
    
    def _record_spawn(self, current_time: float, count: int) -> None:
        """Record a spawn event for rolling average tracking."""
        if count > 0:
            self._recent_spawns.append((current_time, count))
    
    def _try_spawn_particle(self, simulation: 'Simulation') -> bool:
        """
        Attempt to spawn a single particle using rejection sampling.
        
        Returns:
            True if particle was spawned, False if rejected
        """
        # Random position within radius (sqrt for uniform area distribution)
        angle = random.uniform(0, 2 * math.pi)
        r = self.radius * math.sqrt(random.uniform(0, 1))
        x = self.x + r * math.cos(angle)
        y = self.y + r * math.sin(angle)
        
        # Check bounds
        if x < 0 or x > simulation.world_size or y < 0 or y > simulation.world_size:
            return False
        
        # Overlap check using the simulation's method
        if simulation.has_particle_near(x, y, self.properties.sigma * 0.8):
            return False
        
        # Sample velocity from Maxwell-Boltzmann with optional direction bias
        vx, vy = self._sample_velocity()
        
        # Add the particle
        simulation._add_particle(
            x=x, y=y, vx=vx, vy=vy,
            is_static=0,
            sigma=self.properties.sigma,
            epsilon=self.properties.epsilon
        )
        
        return True
    
    def _sample_velocity(self) -> tuple:
        """
        Sample velocity from Maxwell-Boltzmann distribution with optional direction bias.
        
        Returns:
            (vx, vy) tuple
        """
        # Standard deviation for Maxwell-Boltzmann (sqrt(kT/m), using kT = temperature)
        std = math.sqrt(self.properties.temperature / self.properties.mass)
        
        # Sample random direction with optional bias
        if self.properties.injection_spread >= 2 * math.pi - 0.01:
            # Isotropic - uniform random direction
            angle = random.uniform(0, 2 * math.pi)
        else:
            # Biased direction with spread
            half_spread = self.properties.injection_spread / 2
            angle = self.properties.injection_direction + random.uniform(-half_spread, half_spread)
        
        # Sample speed from Maxwell distribution (Rayleigh for 2D)
        speed = std * math.sqrt(-2 * math.log(max(random.random(), 1e-10)))
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        return vx, vy
    
    def get_geometry_for_rendering(self) -> List[dict]:
        """
        Return the dashed circle geometry for rendering.
        
        The center handle is rendered as a normal Point by the Sketch renderer.
        This returns the dashed circumference.
        """
        return [{
            'type': 'dashed_circle',
            'center': (self.x, self.y),
            'radius': self.radius,
        }]
    
    def contains_point(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        """
        Check if a point is inside the Source's spawn region.
        
        Args:
            x, y: Point to test
            tolerance: Extra distance for edge hit testing
            
        Returns:
            True if point is inside (or near edge with tolerance)
        """
        dx = x - self.x
        dy = y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        return dist <= self.radius + tolerance
    
    def hit_test(self, x: float, y: float, tolerance: float = 5.0) -> bool:
        """
        Check if a point hits the Source (center or circumference).
        
        Args:
            x, y: Point to test (world coordinates)
            tolerance: Hit tolerance in world units
            
        Returns:
            True if point hits center or circumference
        """
        dx = x - self.x
        dy = y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Hit center?
        if dist <= tolerance:
            return True
        
        # Hit circumference?
        if abs(dist - self.radius) <= tolerance:
            return True
        
        return False
    
    def to_dict(self) -> dict:
        """Serialize Source to dictionary."""
        return {
            'type': 'source',
            'center': [float(self.x), float(self.y)],
            'radius': self.radius,
            'enabled': self.enabled,
            'properties': self.properties.to_dict(),
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Source':
        """Deserialize Source from dictionary."""
        props = SourceProperties.from_dict(data.get('properties', {}))
        source = Source(
            center=tuple(data['center']),
            radius=data['radius'],
            properties=props,
        )
        source.enabled = data.get('enabled', True)
        return source


# =============================================================================
# DashedCircle helper for rendering (used by renderer)
# =============================================================================

@dataclass
class DashedCircle:
    """Descriptor for a dashed circle to be rendered."""
    center: tuple
    radius: float
    dash_length: float = 5.0  # in pixels
    gap_length: float = 3.0   # in pixels
    color: tuple = (100, 180, 255)  # Light blue


# =============================================================================
# Factory for deserialization
# =============================================================================

def create_process_object(data: dict) -> Optional[ProcessObject]:
    """
    Factory function to create ProcessObject from serialized data.
    
    Args:
        data: Dictionary with 'type' key and type-specific data
        
    Returns:
        ProcessObject instance or None if unknown type
    """
    obj_type = data.get('type')
    
    if obj_type == 'source':
        return Source.from_dict(data)
    # Future: elif obj_type == 'sink': return Sink.from_dict(data)
    # Future: elif obj_type == 'heater': return Heater.from_dict(data)
    
    return None
