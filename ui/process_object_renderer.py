"""
ProcessObject Renderer - Rendering Support for ProcessObjects

This module provides the rendering functions for ProcessObjects like Sources.
These functions should be called from the main Renderer class.

Integration:
Add these methods to ui/renderer.py or call them from there.

Usage in Renderer:
    def draw_process_objects(self, process_objects, session, layout):
        # Import and call the render function
        from ui.process_object_renderer import render_process_objects
        render_process_objects(self.screen, process_objects, session, layout)
"""

import pygame
import math

from core.utils import sim_to_screen


def render_process_objects(screen, process_objects, session, layout, world_size):
    """
    Render all ProcessObjects in the scene.
    
    The handles (Points) are rendered by normal Sketch entity rendering.
    This function renders the non-handle geometry like dashed circles.
    
    Args:
        screen: Pygame screen surface
        process_objects: List of ProcessObject instances
        session: Session instance (for camera transforms)
        layout: Layout dictionary
        world_size: World size for coordinate transforms
    """
    for obj in process_objects:
        geometries = obj.get_geometry_for_rendering()
        
        for geom in geometries:
            if geom['type'] == 'dashed_circle':
                _draw_dashed_circle(
                    screen, 
                    geom['center'],
                    geom['radius'],
                    session,
                    layout,
                    world_size,
                    enabled=obj.enabled
                )


def _draw_dashed_circle(screen, center, radius, session, layout, world_size, enabled=True):
    """
    Draw a dashed circle for a Source's spawn region.
    
    Args:
        screen: Pygame screen surface
        center: (x, y) world coordinates
        radius: Circle radius in world units
        session: Session instance
        layout: Layout dictionary
        world_size: World size for transforms
        enabled: If False, draw grayed out
    """
    # Transform center to screen coordinates
    cx, cy = sim_to_screen(
        center[0], center[1],
        session.camera.zoom,
        session.camera.pan_x,
        session.camera.pan_y,
        world_size,
        layout
    )
    
    # Calculate screen radius
    # Get two points at world_dist apart and measure screen distance
    p0 = sim_to_screen(0, 0, session.camera.zoom, session.camera.pan_x, 
                       session.camera.pan_y, world_size, layout)
    p1 = sim_to_screen(radius, 0, session.camera.zoom, session.camera.pan_x,
                       session.camera.pan_y, world_size, layout)
    screen_radius = abs(p1[0] - p0[0])
    
    if screen_radius < 2:
        return  # Too small to render
    
    # Choose color based on enabled state
    if enabled:
        color = (100, 180, 255)  # Light blue
    else:
        color = (80, 80, 100)    # Grayed out
    
    # Draw dashed circle
    circumference = 2 * math.pi * screen_radius
    dash_length = 6   # pixels
    gap_length = 4    # pixels
    segment_length = dash_length + gap_length
    num_segments = max(8, int(circumference / segment_length))
    
    for i in range(num_segments):
        # Draw only even segments (dashes)
        start_angle = (i / num_segments) * 2 * math.pi
        end_angle = ((i + 0.5) / num_segments) * 2 * math.pi
        
        # Draw arc as line segments
        steps = max(2, int((end_angle - start_angle) * screen_radius / 5))
        points = []
        
        for j in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * j / steps
            x = cx + screen_radius * math.cos(angle)
            y = cy + screen_radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, 2)


def render_source_preview(screen, center_screen, radius_screen, mouse_pos):
    """
    Render a Source preview during tool placement.
    
    Args:
        screen: Pygame screen surface
        center_screen: (x, y) screen coordinates of center
        radius_screen: Radius in screen pixels
        mouse_pos: Current mouse position (for radius line)
    """
    color = (100, 180, 255)
    
    # Draw center point
    pygame.draw.circle(screen, color, center_screen, 5)
    pygame.draw.circle(screen, (255, 255, 255), center_screen, 5, 1)
    
    if radius_screen > 5:
        # Draw dashed circle preview
        circumference = 2 * math.pi * radius_screen
        dash_length = 8
        gap_length = 4
        segment_length = dash_length + gap_length
        num_segments = max(8, int(circumference / segment_length))
        
        for i in range(num_segments):
            if i % 2 == 0:
                start_angle = (i / num_segments) * 2 * math.pi
                end_angle = ((i + 0.6) / num_segments) * 2 * math.pi
                
                steps = max(2, int((end_angle - start_angle) * radius_screen / 5))
                points = []
                
                for j in range(steps + 1):
                    angle = start_angle + (end_angle - start_angle) * j / steps
                    x = center_screen[0] + radius_screen * math.cos(angle)
                    y = center_screen[1] + radius_screen * math.sin(angle)
                    points.append((int(x), int(y)))
                
                if len(points) >= 2:
                    pygame.draw.lines(screen, color, False, points, 2)
        
        # Draw radius line
        pygame.draw.line(screen, color, center_screen, mouse_pos, 1)
