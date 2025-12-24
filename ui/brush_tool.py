"""
Refactored BrushTool - Uses ParticleBrush for particle operations

This is a drop-in replacement for the BrushTool in ui/tools.py.
The key change is that brush logic (hex packing, overlap detection)
is now in ParticleBrush, keeping the tool focused on input handling.

Replace the BrushTool class in ui/tools.py with this version.
"""

import pygame
import core.config as config
import core.utils as utils

from core.session import InteractionState
from engine.particle_brush import ParticleBrush


class BrushTool:
    """
    Brush tool for painting/erasing particles.
    
    This tool handles INPUT only:
    - Mouse down/up detection
    - Coordinate transforms
    - Delegating to ParticleBrush for actual particle operations
    
    The ParticleBrush handles BRUSH LOGIC:
    - Hexagonal packing
    - Overlap detection
    - Particle creation/deletion
    """
    
    def __init__(self, app):
        self.app = app
        self.name = "Brush"
        self.brush_radius = 5.0
        
        # Lazy-initialized brush (created on first use)
        self._brush = None
    
    @property
    def brush(self):
        """Get the ParticleBrush, creating it if needed."""
        if self._brush is None:
            self._brush = ParticleBrush(self.app.sim)
        return self._brush
    
    def activate(self):
        """Called when tool becomes active."""
        pass
    
    def deactivate(self):
        """Called when switching away from this tool."""
        pass
    
    def cancel(self):
        """Cancel current operation."""
        if self.app.session.state == InteractionState.PAINTING:
            self.app.session.state = InteractionState.IDLE
    
    def update(self, dt, layout):
        """
        Called every frame while tool is active.
        
        Handles continuous painting while mouse is held down.
        """
        if self.app.session.state != InteractionState.PAINTING:
            return
        
        mx, my = pygame.mouse.get_pos()
        
        # Check if mouse is in the viewport
        if not (layout['MID_X'] < mx < layout['RIGHT_X'] and 
                config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
            return
        
        # Convert to world coordinates
        sim = self.app.sim
        wx, wy = utils.screen_to_sim(
            mx, my,
            self.app.session.zoom,
            self.app.session.pan_x,
            self.app.session.pan_y,
            sim.world_size,
            layout
        )
        
        # Paint or erase based on which button is held
        if pygame.mouse.get_pressed()[0]:  # Left button - paint
            self.brush.paint(wx, wy, self.brush_radius)
        elif pygame.mouse.get_pressed()[2]:  # Right button - erase
            self.brush.erase(wx, wy, self.brush_radius)
    
    def handle_event(self, event, layout):
        """
        Handle pygame events.
        
        Returns True if event was consumed.
        """
        # Only active in SIM mode
        if self.app.session.mode != config.MODE_SIM:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            
            # Check if click is in viewport
            if not (layout['MID_X'] < mx < layout['RIGHT_X'] and 
                    config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
                return False
            
            if event.button in (1, 3):  # Left or right click
                # Enter painting state
                self.app.session.state = InteractionState.PAINTING
                
                # Snapshot for undo (physics domain)
                self.app.sim.snapshot()
                
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.app.session.state == InteractionState.PAINTING:
                self.app.session.state = InteractionState.IDLE
                return True
        
        return False
    
    def draw_overlay(self, screen, renderer, layout):
        """Draw brush cursor overlay."""
        mx, my = pygame.mouse.get_pos()
        
        # Only draw if mouse is in viewport
        if not (layout['MID_X'] < mx < layout['RIGHT_X'] and 
                config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
            return
        
        # Convert brush radius to screen space
        zoom = self.app.session.zoom
        world_size = self.app.sim.world_size
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * zoom
        screen_radius = self.brush_radius * final_scale
        
        renderer.draw_tool_brush(mx, my, screen_radius)