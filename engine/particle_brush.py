"""
Particle Brush - Brush-based Particle Creation/Deletion

This module contains the brush interaction logic that was previously
embedded in the Simulation class. The Simulation should only know
about particles, not about "brushes" which are a UI/tool concept.

The brush generates positions using hexagonal close packing, checks
for overlaps, and delegates to Simulation's primitive particle methods.

Usage:
    brush = ParticleBrush(simulation)
    brush.paint(x, y, radius)  # Add particles
    brush.erase(x, y, radius)  # Remove particles
"""

import math
import numpy as np


class ParticleBrush:
    """
    Handles brush-based particle creation and deletion.
    
    This class knows about:
    - Hexagonal packing geometry
    - Overlap detection
    - Brush radius and painting patterns
    
    It does NOT know about:
    - Physics (forces, integration)
    - Rendering
    - Input handling
    
    The Simulation provides primitive operations:
    - add_particle()
    - remove_particles_at()
    - check_overlap()
    """
    
    def __init__(self, simulation):
        """
        Initialize the brush with a reference to the simulation.
        
        Args:
            simulation: The Simulation instance to paint into
        """
        self.sim = simulation
    
    def paint(self, x: float, y: float, radius: float,
              sigma: float = None, epsilon: float = None,
              color: tuple = None) -> int:
        """
        Add particles in a circular brush pattern using hexagonal packing.

        Args:
            x: World X coordinate of brush center
            y: World Y coordinate of brush center
            radius: Brush radius in world units
            sigma: Particle size (uses sim.sigma if None)
            epsilon: Interaction strength (uses sim.epsilon if None)
            color: RGB tuple for particle color (uses default blue if None)

        Returns:
            Number of particles added
        """
        sim = self.sim
        sigma = sigma if sigma is not None else sim.sigma
        epsilon = epsilon if epsilon is not None else sim.epsilon
        color = color if color is not None else (50, 150, 255)

        # Optimal spacing for Lennard-Jones potential (2^(1/6) ≈ 1.12246)
        spacing = 1.12246 * sigma
        row_height = spacing * 0.866025  # sqrt(3)/2 for hex packing

        r_sq = radius * radius
        n_rows = int(radius / row_height) + 1
        n_cols = int(radius / spacing) + 1

        # Generate candidate positions
        positions = []
        for row in range(-n_rows, n_rows + 1):
            # Offset every other row for hex packing
            offset_x = 0.5 * spacing if (row % 2 != 0) else 0.0
            y_curr = y + row * row_height

            for col in range(-n_cols, n_cols + 1):
                x_curr = x + col * spacing + offset_x

                # Check if within brush circle
                dx = x_curr - x
                dy = y_curr - y
                if dx * dx + dy * dy <= r_sq:
                    # Check if within world bounds
                    if 0 < x_curr < sim.world_size and 0 < y_curr < sim.world_size:
                        positions.append((x_curr, y_curr))

        # Add particles that don't overlap
        added = 0
        overlap_threshold = 0.8 * sigma

        for px, py in positions:
            if not self._check_overlap(px, py, overlap_threshold):
                self._add_particle(px, py, sigma, epsilon, color)
                added += 1

        if added > 0:
            sim.rebuild_next = True

        return added
    
    def erase(self, x: float, y: float, radius: float) -> int:
        """
        Remove dynamic particles within a circular brush area.

        Static and tethered particles are preserved.
        
        Args:
            x: World X coordinate of brush center
            y: World Y coordinate of brush center
            radius: Brush radius in world units
        
        Returns:
            Number of particles removed
        """
        sim = self.sim
        r_sq = radius * radius
        
        # Find particles to remove (dynamic only)
        indices_to_remove = []
        for i in range(sim.count):
            # Skip non-dynamic particles (static=1, tethered=3)
            if sim.is_static[i] != 0:
                continue
            
            dx = sim.pos_x[i] - x
            dy = sim.pos_y[i] - y
            if dx * dx + dy * dy <= r_sq:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return 0
        
        # Build list of indices to keep
        remove_set = set(indices_to_remove)
        keep_indices = [i for i in range(sim.count) if i not in remove_set]
        
        # Compact arrays
        sim.compact_arrays(keep_indices)
        sim.rebuild_next = True
        
        return len(indices_to_remove)
    
    def _add_particle(self, x: float, y: float, sigma: float, epsilon: float,
                      color: tuple = (50, 150, 255)):
        """
        Add a single dynamic particle to the simulation.

        Args:
            x, y: World coordinates
            sigma: Particle size parameter
            epsilon: Interaction strength parameter
            color: RGB tuple for particle color
        """
        sim = self.sim

        # Ensure capacity
        if sim.count >= sim.capacity:
            sim._resize_arrays()

        idx = sim.count
        sim.pos_x[idx] = x
        sim.pos_y[idx] = y
        sim.vel_x[idx] = 0.0
        sim.vel_y[idx] = 0.0
        sim.is_static[idx] = 0  # Dynamic
        sim.atom_sigma[idx] = sigma
        sim.atom_eps_sqrt[idx] = math.sqrt(epsilon)
        sim.atom_color[idx] = color
        sim.count += 1
    
    def _check_overlap(self, x: float, y: float, threshold: float) -> bool:
        """
        Check if a position overlaps with existing particles.
        
        Args:
            x, y: World coordinates to check
            threshold: Minimum distance to existing particles
        
        Returns:
            True if overlapping, False if clear
        """
        sim = self.sim
        if sim.count == 0:
            return False
        
        threshold_sq = threshold * threshold
        
        # Vectorized distance check
        dx = sim.pos_x[:sim.count] - x
        dy = sim.pos_y[:sim.count] - y
        dist_sq = dx * dx + dy * dy
        
        return np.any(dist_sq < threshold_sq)
    
    # =========================================================================
    # Advanced Brush Operations
    # =========================================================================
    
    def spray(self, x: float, y: float, radius: float, density: float = 0.5,
              sigma: float = None, epsilon: float = None,
              color: tuple = None) -> int:
        """
        Add particles with random placement (spray pattern).

        Unlike paint(), this uses random positions instead of hex grid.
        Useful for more organic-looking distributions.

        Args:
            x, y: World coordinates of brush center
            radius: Brush radius in world units
            density: Particles per unit area (approximate)
            sigma: Particle size (uses sim.sigma if None)
            epsilon: Interaction strength (uses sim.epsilon if None)
            color: RGB tuple for particle color (uses default blue if None)

        Returns:
            Number of particles added
        """
        sim = self.sim
        sigma = sigma if sigma is not None else sim.sigma
        epsilon = epsilon if epsilon is not None else sim.epsilon
        color = color if color is not None else (50, 150, 255)

        # Calculate number of particles to attempt
        area = math.pi * radius * radius
        n_attempts = int(area * density)

        added = 0
        overlap_threshold = 0.8 * sigma

        for _ in range(n_attempts):
            # Random position within circle
            r = radius * math.sqrt(np.random.random())
            theta = 2 * math.pi * np.random.random()
            px = x + r * math.cos(theta)
            py = y + r * math.sin(theta)

            # Bounds check
            if not (0 < px < sim.world_size and 0 < py < sim.world_size):
                continue

            # Overlap check
            if not self._check_overlap(px, py, overlap_threshold):
                self._add_particle(px, py, sigma, epsilon, color)
                added += 1

        if added > 0:
            sim.rebuild_next = True

        return added
    
    def fill_rect(self, x1: float, y1: float, x2: float, y2: float,
                  sigma: float = None, epsilon: float = None,
                  color: tuple = None) -> int:
        """
        Fill a rectangular region with particles.

        Args:
            x1, y1: One corner of the rectangle
            x2, y2: Opposite corner of the rectangle
            sigma: Particle size (uses sim.sigma if None)
            epsilon: Interaction strength (uses sim.epsilon if None)
            color: RGB tuple for particle color (uses default blue if None)

        Returns:
            Number of particles added
        """
        sim = self.sim
        sigma = sigma if sigma is not None else sim.sigma
        epsilon = epsilon if epsilon is not None else sim.epsilon
        color = color if color is not None else (50, 150, 255)

        # Normalize rectangle
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Clamp to world bounds
        min_x = max(0, min_x)
        max_x = min(sim.world_size, max_x)
        min_y = max(0, min_y)
        max_y = min(sim.world_size, max_y)

        spacing = 1.12246 * sigma
        row_height = spacing * 0.866025
        overlap_threshold = 0.8 * sigma

        added = 0
        row = 0
        y_curr = min_y

        while y_curr <= max_y:
            offset_x = 0.5 * spacing if (row % 2 != 0) else 0.0
            x_curr = min_x + offset_x

            while x_curr <= max_x:
                if not self._check_overlap(x_curr, y_curr, overlap_threshold):
                    self._add_particle(x_curr, y_curr, sigma, epsilon, color)
                    added += 1
                x_curr += spacing
            
            y_curr += row_height
            row += 1
        
        if added > 0:
            sim.rebuild_next = True
        
        return added