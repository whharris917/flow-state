"""
CameraController - View Transform Management

Owns:
- Zoom level and pan offset
- Stored view states per mode (sim vs editor)
- Coordinate transform methods

This extracts camera concerns from the Session god object.
"""

import core.config as config


class CameraController:
    """
    Manages camera state and coordinate transformations.
    
    The camera defines how world coordinates map to screen coordinates.
    Each mode (SIM, EDITOR) has its own stored view state.
    """
    
    def __init__(self):
        # Active camera state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Stored view states per mode
        self._stored_views = {
            config.MODE_SIM: {'zoom': 1.0, 'pan_x': 0.0, 'pan_y': 0.0},
            config.MODE_EDITOR: {'zoom': 1.5, 'pan_x': 0.0, 'pan_y': 0.0},
        }
        
        # For pan gesture tracking
        self.last_mouse_pos = (0, 0)
    
    # =========================================================================
    # View State Management
    # =========================================================================
    
    def store_view(self, mode):
        """Store current view state for a mode."""
        self._stored_views[mode] = {
            'zoom': self.zoom,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y,
        }
    
    def restore_view(self, mode):
        """Restore view state for a mode."""
        if mode in self._stored_views:
            state = self._stored_views[mode]
            self.zoom = state['zoom']
            self.pan_x = state['pan_x']
            self.pan_y = state['pan_y']
    
    def get_view_state(self):
        """Get current view state as a dict (for serialization)."""
        return {
            'zoom': self.zoom,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y,
        }
    
    def set_view_state(self, state):
        """Set view state from a dict (for deserialization)."""
        if state:
            self.zoom = state.get('zoom', self.zoom)
            self.pan_x = state.get('pan_x', self.pan_x)
            self.pan_y = state.get('pan_y', self.pan_y)
    
    # =========================================================================
    # Coordinate Transforms
    # =========================================================================
    
    def world_to_screen(self, wx, wy, world_size, layout):
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            wx, wy: World coordinates
            world_size: Size of the simulation world
            layout: Screen layout dict with MID_X, MID_W, MID_H keys
        
        Returns:
            (sx, sy): Screen coordinates as integers
        """
        cx_world = world_size / 2.0
        cy_world = world_size / 2.0
        cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
        cy_screen = config.TOP_MENU_H + (layout['MID_H'] / 2.0)
        
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * self.zoom
        
        sx = cx_screen + (wx - cx_world) * final_scale + self.pan_x
        sy = cy_screen + (wy - cy_world) * final_scale + self.pan_y
        return int(sx), int(sy)
    
    def screen_to_world(self, sx, sy, world_size, layout):
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            sx, sy: Screen coordinates
            world_size: Size of the simulation world
            layout: Screen layout dict with MID_X, MID_W, MID_H keys
        
        Returns:
            (wx, wy): World coordinates as floats
        """
        cx_world = world_size / 2.0
        cy_world = world_size / 2.0
        cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
        cy_screen = config.TOP_MENU_H + (layout['MID_H'] / 2.0)
        
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * self.zoom
        
        wx = (sx - self.pan_x - cx_screen) / final_scale + cx_world
        wy = (sy - self.pan_y - cy_screen) / final_scale + cy_world
        return wx, wy
    
    def get_world_radius(self, screen_radius, world_size, layout):
        """
        Convert a screen-space radius to world-space radius.
        
        Args:
            screen_radius: Radius in pixels
            world_size: Size of the simulation world
            layout: Screen layout dict
        
        Returns:
            World-space radius
        """
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * self.zoom
        return screen_radius / final_scale
    
    def get_screen_radius(self, world_radius, world_size, layout):
        """
        Convert a world-space radius to screen-space radius.
        
        Args:
            world_radius: Radius in world units
            world_size: Size of the simulation world
            layout: Screen layout dict
        
        Returns:
            Screen-space radius in pixels
        """
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * self.zoom
        return world_radius * final_scale
    
    # =========================================================================
    # Camera Controls
    # =========================================================================
    
    def apply_zoom(self, delta, min_zoom=0.1, max_zoom=50.0):
        """
        Apply zoom change.
        
        Args:
            delta: Positive for zoom in, negative for zoom out
            min_zoom: Minimum allowed zoom level
            max_zoom: Maximum allowed zoom level
        """
        factor = 1.1 if delta > 0 else (1.0 / 1.1)
        self.zoom = max(min_zoom, min(self.zoom * factor, max_zoom))
    
    def apply_pan(self, dx, dy):
        """
        Apply pan offset.
        
        Args:
            dx, dy: Pan delta in screen pixels
        """
        self.pan_x += dx
        self.pan_y += dy
    
    def reset(self):
        """Reset camera to default state."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0