"""
SourceTool - Tool for Creating Particle Source Emitters

This tool follows the two-click pattern:
1. First click: Set center position (with snap support)
2. Second click: Set radius

The Source's center handle participates in constraints like any Point.
"""

import pygame
import math

import core.config as config
import core.utils as utils

from core.session import InteractionState
from model.process_objects import Source, SourceProperties


class SourceTool:
    """
    Tool for creating Source (particle emitter) ProcessObjects.
    
    Two-click workflow:
    1. Click to set center (snaps to existing geometry)
    2. Click to set radius
    
    During step 2, a preview circle is drawn following the mouse.
    """
    
    def __init__(self, app):
        self.app = app
        self.name = "Source"
        
        # State
        self.center = None          # (x, y) world coords after first click
        self.preview_radius = None  # Current radius preview
        self.center_snap = None     # Snap target for constraint creation
    
    def activate(self):
        """Called when tool becomes active."""
        self._reset()
    
    def deactivate(self):
        """Called when switching away from this tool."""
        self._reset()
        self.app.session.constraint_builder.snap_target = None
    
    def _reset(self):
        """Reset tool state."""
        self.center = None
        self.preview_radius = None
        self.center_snap = None
    
    def cancel(self):
        """Cancel current operation."""
        if self.center is not None:
            self._reset()
            self.app.session.status.set("Source cancelled")
        self.app.session.constraint_builder.snap_target = None
    
    def handle_event(self, event, layout):
        """Handle pygame events. Returns True if event was consumed."""
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            
            # Only handle clicks in the viewport
            if not (layout['LEFT_X'] < mx < layout['RIGHT_X']):
                return False
            
            # Get snapped world position
            wx, wy, snap = self._get_snapped(mx, my, layout)
            
            if self.center is None:
                # First click: set center
                self.center = (wx, wy)
                self.center_snap = snap
                self.app.session.status.set("Click to set radius")
                return True
            else:
                # Second click: set radius and create Source
                radius = math.hypot(wx - self.center[0], wy - self.center[1])
                radius = max(0.5, radius)  # Minimum radius
                
                self._create_source(radius)
                self._reset()
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            
            if self.center is not None:
                # Update preview radius
                wx, wy = self._get_world_pos(mx, my, layout)
                self.preview_radius = math.hypot(wx - self.center[0], wy - self.center[1])
                self.preview_radius = max(0.5, self.preview_radius)
            else:
                # Update snap indicator
                self._update_hover_snap(mx, my, layout)
            
            return False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
                return True
        
        return False
    
    def update(self, dt, layout):
        """Called every frame while tool is active."""
        pass
    
    def draw_overlay(self, screen, renderer, layout):
        """Draw tool-specific overlay graphics."""
        mx, my = pygame.mouse.get_pos()
        
        # Only draw if mouse is in viewport
        if not (layout['MID_X'] < mx < layout['RIGHT_X']):
            return
        
        if self.center is not None:
            # Draw center point
            cx_screen, cy_screen = self._world_to_screen(
                self.center[0], self.center[1], layout
            )
            pygame.draw.circle(screen, (100, 180, 255), (cx_screen, cy_screen), 5)
            pygame.draw.circle(screen, (255, 255, 255), (cx_screen, cy_screen), 5, 1)
            
            # Draw preview circle if we have a radius
            if self.preview_radius is not None:
                screen_radius = self._world_to_screen_distance(self.preview_radius, layout)
                self._draw_dashed_circle(
                    screen, 
                    (cx_screen, cy_screen), 
                    int(screen_radius),
                    (100, 180, 255, 128)
                )
                
                # Draw radius line
                pygame.draw.line(
                    screen, (100, 180, 255),
                    (cx_screen, cy_screen), (mx, my), 1
                )
        else:
            # Draw crosshair at mouse position
            pygame.draw.circle(screen, (100, 180, 255), (mx, my), 4)
    
    def _create_source(self, radius):
        """Create the Source and add to scene."""
        from model.process_objects import Source, SourceProperties
        from model.constraints import Coincident
        from core.commands import AddConstraintCommand
        
        # Create Source with default properties
        source = Source(
            center=self.center,
            radius=radius,
            properties=SourceProperties()
        )
        
        # Add to scene (this registers handles with Sketch)
        self.app.scene.add_process_object(source)
        
        # If we snapped to a point, create a coincident constraint
        if self.center_snap and pygame.key.get_mods() & pygame.KMOD_CTRL:
            # Get the handle index in the sketch
            handle_indices = source.get_handle_indices(self.app.scene.sketch)
            if 'center' in handle_indices:
                center_idx = handle_indices['center']
                snap_entity, snap_pt = self.center_snap
                
                # Create coincident constraint between Source center and snap target
                constraint = Coincident(center_idx, 0, snap_entity, snap_pt)
                self.app.scene.execute(AddConstraintCommand(
                    self.app.scene.sketch, constraint
                ))
        
        self.app.session.status.set(f"Source created (r={radius:.1f})")
    
    def _get_world_pos(self, mx, my, layout):
        """Convert screen coordinates to world coordinates."""
        return utils.screen_to_sim(
            mx, my,
            self.app.session.camera.zoom,
            self.app.session.camera.pan_x,
            self.app.session.camera.pan_y,
            self.app.sim.world_size,
            layout
        )
    
    def _get_snapped(self, mx, my, layout):
        """Get snapped world position and snap target."""
        mods = pygame.key.get_mods()
        snap_to_points = bool(mods & pygame.KMOD_CTRL)
        constrain_to_axis = bool(mods & pygame.KMOD_SHIFT)
        return utils.get_snapped_pos(
            mx, my,
            self.app.scene.sketch.entities,
            self.app.session.camera.zoom,
            self.app.session.camera.pan_x,
            self.app.session.camera.pan_y,
            self.app.sim.world_size,
            layout,
            snap_to_points=snap_to_points,
            constrain_to_axis=constrain_to_axis
        )
    
    def _update_hover_snap(self, mx, my, layout):
        """Update snap indicator when not placing."""
        _, _, snap = self._get_snapped(mx, my, layout)
        self.app.session.constraint_builder.snap_target = snap
    
    def _world_to_screen(self, wx, wy, layout):
        """Convert world coordinates to screen coordinates."""
        return utils.sim_to_screen(
            wx, wy,
            self.app.session.camera.zoom,
            self.app.session.camera.pan_x,
            self.app.session.camera.pan_y,
            self.app.sim.world_size,
            layout
        )
    
    def _world_to_screen_distance(self, world_dist, layout):
        """Convert a world distance to screen pixels."""
        # Get scale factor
        zoom = self.app.session.camera.zoom
        world_size = self.app.sim.world_size
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * zoom
        return world_dist * final_scale
    
    def _draw_dashed_circle(self, screen, center, radius, color):
        """Draw a dashed circle preview."""
        if radius < 5:
            pygame.draw.circle(screen, color[:3], center, radius, 1)
            return
        
        # Calculate dash parameters
        circumference = 2 * math.pi * radius
        dash_length = 8  # pixels
        gap_length = 4   # pixels
        segment_length = dash_length + gap_length
        num_segments = max(8, int(circumference / segment_length))
        
        # Draw dashes
        for i in range(num_segments):
            if i % 2 == 0:  # Draw only even segments (dashes)
                start_angle = (i / num_segments) * 2 * math.pi
                end_angle = ((i + 0.6) / num_segments) * 2 * math.pi
                
                # Draw arc as line segments
                steps = max(2, int((end_angle - start_angle) * radius / 5))
                points = []
                for j in range(steps + 1):
                    angle = start_angle + (end_angle - start_angle) * j / steps
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((int(x), int(y)))
                
                if len(points) >= 2:
                    pygame.draw.lines(screen, color[:3], False, points, 2)
