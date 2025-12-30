"""
Tools - Input Handling Tools for the Editor

Each tool handles mouse events for a specific operation:
- BrushTool: Paint/erase particles (physics domain)
- LineTool: Draw lines (via commands with supersede)
- RectTool: Draw rectangles (via commands with supersede)
- CircleTool: Draw circles (via commands with supersede)
- PointTool: Create points (via commands)
- SelectTool: Select and move entities (via commands with historize)

Architecture:
- Tools receive a reference to the application (self.app)
- ALL geometry mutations go through Commands via scene.execute()
- Creation tools use supersede=True pattern for live preview
- Editing tools use historize=False pattern for drag feedback
- Orchestrator (Scene.update) handles rebuild/solve automatically
- BrushTool operates on physics (not CAD), uses sim directly

Tool Contract - Allowed Access via self.app:

    READ-ONLY STATE:
    - session.camera.zoom, pan_x, pan_y  — View transforms
    - session.mode                       — SIM vs EDITOR mode
    - session.state                      — Interaction state enum
    - session.selection.walls/points     — Current selection
    - sim.world_size                     — For coordinate transforms
    - layout                             — Screen regions
    - sketch.entities                    — Geometry for hit testing/snapping
    
    WRITE STATE:
    - session.state                              — Set interaction state
    - session.constraint_builder.snap_target     — Update snap indicator
    - session.selection.walls/points             — Modify selection
    - session.status.set()                       — Status bar messages
    
    COMMANDS (geometry changes):
    - scene.execute(cmd)          — Execute command (rebuild/solve automatic)
    - scene.discard()             — For cancel operations (removes, not redoable)
    
    PHYSICS (BrushTool only):
    - sim.snapshot()              — Physics undo
    - ParticleBrush.paint()       — Create particles
    - ParticleBrush.erase()       — Erase particles
"""

import pygame
import numpy as np
import math
import time

import core.config as config
import core.utils as utils

from model.geometry import Line, Circle
from model.constraints import Coincident
from core.session import InteractionState
from engine.particle_brush import ParticleBrush

# Commands for proper undo/redo
from core.commands import (
    AddLineCommand, AddCircleCommand, RemoveEntityCommand,
    MoveEntityCommand, MoveMultipleCommand, AddConstraintCommand,
    AddRectangleCommand, SetPointCommand, SetCircleRadiusCommand,
    SetEntityGeometryCommand
)


# =============================================================================
# Base Tool Class
# =============================================================================

class Tool:
    """
    Base class for all editor tools.
    
    See module docstring for the allowed access contract via self.app.
    """
    def __init__(self, app, name="Tool"):
        self.app = app
        self.name = name

    def activate(self):
        """Called when tool becomes active."""
        pass

    def deactivate(self):
        """Called when switching away from this tool."""
        self.app.session.constraint_builder.snap_target = None

    def handle_event(self, event, layout):
        """Handle a pygame event. Return True if event was consumed."""
        return False

    def update(self, dt, layout):
        """Called every frame while tool is active."""
        pass

    def draw_overlay(self, screen, renderer, layout):
        """Draw tool-specific overlay graphics."""
        pass
    
    def cancel(self):
        """Cancel the current operation."""
        pass


# =============================================================================
# Brush Tool (Physics Domain)
# =============================================================================

"""
Refactored BrushTool - Uses ParticleBrush for particle operations

This is a drop-in replacement for the BrushTool in ui/tools.py.
The key change is that brush logic (hex packing, overlap detection)
is now in ParticleBrush, keeping the tool focused on input handling.
"""

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
            self.app.session.camera.zoom,
            self.app.session.camera.pan_x,
            self.app.session.camera.pan_y,
            sim.world_size,
            layout
        )
        
        # Paint or erase based on which button is held
        if pygame.mouse.get_pressed()[0]:  # Left button - paint
            mat = self.app.session.active_material
            self.brush.paint(wx, wy, self.brush_radius,
                           sigma=mat.sigma, epsilon=mat.epsilon,
                           color=mat.color)
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
        zoom = self.app.session.camera.zoom
        world_size = self.app.sim.world_size
        base_scale = (layout['MID_W'] - 50) / world_size
        final_scale = base_scale * zoom
        screen_radius = self.brush_radius * final_scale
        
        renderer.draw_tool_brush(mx, my, screen_radius)


# =============================================================================
# Geometry Tool Base Class
# =============================================================================

class GeometryTool(Tool):
    """
    Base class for geometry creation tools.
    
    Uses Commands with supersede pattern:
    - MOUSEDOWN: Create initial geometry (historize=True)
    - MOUSEMOVE: Supersede with updated geometry (supersede=True)
    - MOUSEUP: Final supersede, stays in undo stack
    - CANCEL: scene.undo() removes the preview
    """
    def __init__(self, app, name):
        super().__init__(app, name)
        self.dragging = False
        self.start_pos = None

    @property
    def sketch(self):
        return self.app.scene.sketch
    
    @property
    def scene(self):
        return self.app.scene

    def get_world_pos(self, mx, my, layout):
        """Convert screen coordinates to world coordinates."""
        return utils.screen_to_sim(
            mx, my, 
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y, 
            self.app.sim.world_size, layout
        )

    def get_snapped(self, mx, my, layout, anchor=None, exclude_idx=-1):
        """Get snapped world position and snap target info."""
        mods = pygame.key.get_mods()
        snap_to_points = bool(mods & pygame.KMOD_CTRL)
        constrain_to_axis = bool(mods & pygame.KMOD_SHIFT)
        return utils.get_snapped_pos(
            mx, my,
            self.sketch.entities,
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
            self.app.sim.world_size, layout, anchor, exclude_idx,
            snap_to_points=snap_to_points, constrain_to_axis=constrain_to_axis
        )

    def cancel(self):
        """Cancel current operation - discard the preview geometry."""
        if self.dragging:
            self.scene.discard()  # Remove preview permanently (not redoable)
            self.dragging = False
            self.start_pos = None
            self.app.session.state = InteractionState.IDLE
            self.app.session.status.set("Cancelled")
        self.app.session.constraint_builder.snap_target = None

    def _update_hover_snap(self, mx, my, layout):
        """Update snap indicator when not dragging."""
        if not self.dragging:
            _, _, snap = self.get_snapped(mx, my, layout, anchor=None)
            self.app.session.constraint_builder.snap_target = snap


# =============================================================================
# Line Tool
# =============================================================================

class LineTool(GeometryTool):
    """
    Line drawing tool using supersede pattern.

    Supports two interaction modes:
    - Drag mode: Click, hold, drag to endpoint, release
    - Click-click mode: Quick click sets start, move mouse, click again for end

    If user releases within 0.1s of clicking, enters click-click mode.
    """
    QUICK_CLICK_THRESHOLD = 0.1  # seconds

    def __init__(self, app):
        super().__init__(app, "Line")
        self.start_snap = None
        self.click_time = 0
        self.click_click_mode = False

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not (layout['LEFT_X'] < mx < layout['RIGHT_X']):
                return False

            # If in click-click mode, this click finalizes the line
            if self.click_click_mode and self.dragging:
                return self._finalize_line(mx, my, layout)

            # Start a new line
            sx, sy, snap = self.get_snapped(mx, my, layout)
            self.start_pos = (sx, sy)
            self.start_snap = snap
            self.click_time = time.time()
            self.click_click_mode = False

            # Create initial degenerate line (zero length)
            is_ref = (self.name == "Ref Line")
            cmd = AddLineCommand(self.sketch, (sx, sy), (sx, sy), is_ref=is_ref)
            self.scene.execute(cmd)

            self.dragging = True
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                anchor = self.start_pos
                # Exclude the line being created from snapping (it's the last entity)
                current_line_idx = len(self.sketch.entities) - 1
                sx, sy, snap = self.get_snapped(mx, my, layout, anchor, exclude_idx=current_line_idx)
                self.app.session.constraint_builder.snap_target = snap

                # Supersede with updated line
                is_ref = (self.name == "Ref Line")
                cmd = AddLineCommand(
                    self.sketch, self.start_pos, (sx, sy),
                    is_ref=is_ref, supersede=True
                )
                self.scene.execute(cmd)

                # Show length in status bar
                if self.app.session.mode == config.MODE_EDITOR:
                    length = math.hypot(sx - self.start_pos[0], sy - self.start_pos[1])
                    self.app.session.status.set(f"Length: {length:.2f}")
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            # Check if this was a quick click (< threshold)
            elapsed = time.time() - self.click_time
            if elapsed < self.QUICK_CLICK_THRESHOLD and not self.click_click_mode:
                # Enter click-click mode - don't finalize yet
                self.click_click_mode = True
                self.app.session.status.set("Click to set endpoint")
                return True

            # Normal drag release - finalize the line
            mx, my = event.pos
            return self._finalize_line(mx, my, layout)

        return False

    def _finalize_line(self, mx, my, layout):
        """Finalize line creation with snap constraints."""
        anchor = self.start_pos

        # Exclude the line being created from snapping (it's the last entity)
        current_line_idx = len(self.sketch.entities) - 1
        sx, sy, snap = self.get_snapped(mx, my, layout, anchor, exclude_idx=current_line_idx)

        # Final supersede - this one stays in the undo stack
        is_ref = (self.name == "Ref Line")
        cmd = AddLineCommand(
            self.sketch, self.start_pos, (sx, sy),
            is_ref=is_ref, supersede=True
        )
        self.scene.execute(cmd)
        wall_idx = cmd.created_index

        # Add snap constraints if Ctrl held
        if pygame.key.get_mods() & pygame.KMOD_CTRL:
            if self.start_snap:
                c = Coincident(wall_idx, 0, self.start_snap[0], self.start_snap[1])
                self.scene.execute(AddConstraintCommand(self.sketch, c))
            if snap:
                c = Coincident(wall_idx, 1, snap[0], snap[1])
                self.scene.execute(AddConstraintCommand(self.sketch, c))

        self._finish_drag()
        return True

    def _finish_drag(self):
        """Clean up drag state."""
        self.dragging = False
        self.start_pos = None
        self.start_snap = None
        self.click_click_mode = False
        self.app.session.state = InteractionState.IDLE
        self.app.session.constraint_builder.snap_target = None

    def cancel(self):
        """Cancel current operation - also resets click-click mode."""
        self.click_click_mode = False
        super().cancel()

    def draw_overlay(self, screen, renderer, layout):
        # Show snap point when hovering (not dragging)
        if not self.dragging:
            mx, my = pygame.mouse.get_pos()
            if layout['MID_X'] < mx < layout['RIGHT_X']:
                renderer.draw_tool_point(mx, my)


# =============================================================================
# Rectangle Tool
# =============================================================================

class RectTool(GeometryTool):
    """
    Rectangle drawing tool using supersede pattern.
    
    Creates 4 lines with corner coincident constraints and H/V constraints.
    """
    def __init__(self, app):
        super().__init__(app, "Rectangle")

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, _ = self.get_snapped(mx, my, layout)
                self.start_pos = (sx, sy)
                
                # Create initial degenerate rectangle (zero size)
                cmd = AddRectangleCommand(self.sketch, sx, sy, sx, sy)
                self.scene.execute(cmd)
                
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cx, cy = self.get_world_pos(mx, my, layout)
                sx, sy = self.start_pos
                
                # Supersede with updated rectangle
                cmd = AddRectangleCommand(
                    self.sketch, sx, sy, cx, cy,
                    supersede=True
                )
                self.scene.execute(cmd)
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            mx, my = event.pos
            cx, cy = self.get_world_pos(mx, my, layout)
            sx, sy = self.start_pos
            
            # Final supersede - stays in undo stack
            cmd = AddRectangleCommand(
                self.sketch, sx, sy, cx, cy,
                supersede=True
            )
            self.scene.execute(cmd)
            
            self._finish_drag()
            return True
        
        return False

    def _finish_drag(self):
        """Clean up drag state."""
        self.dragging = False
        self.start_pos = None
        self.app.session.state = InteractionState.IDLE


# =============================================================================
# Circle Tool
# =============================================================================

class CircleTool(GeometryTool):
    """
    Circle drawing tool using supersede pattern.

    Supports two interaction modes:
    - Drag mode: Click, hold, drag to set radius, release
    - Click-click mode: Quick click sets center, move mouse, click again for radius

    If user releases within 0.1s of clicking, enters click-click mode.
    """
    QUICK_CLICK_THRESHOLD = 0.1  # seconds

    def __init__(self, app):
        super().__init__(app, "Circle")
        self.center_snap = None
        self.click_time = 0
        self.click_click_mode = False

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not (layout['LEFT_X'] < mx < layout['RIGHT_X']):
                return False

            # If in click-click mode, this click finalizes the circle
            if self.click_click_mode and self.dragging:
                return self._finalize_circle(mx, my, layout)

            # Start a new circle
            sx, sy, snap = self.get_snapped(mx, my, layout)
            self.start_pos = (sx, sy)
            self.center_snap = snap
            self.click_time = time.time()
            self.click_click_mode = False

            # Create initial circle with minimal radius
            cmd = AddCircleCommand(self.sketch, (sx, sy), 0.1)
            self.scene.execute(cmd)

            self.dragging = True
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cx, cy = self.get_world_pos(mx, my, layout)
                center = self.start_pos
                radius = max(0.1, math.hypot(cx - center[0], cy - center[1]))

                # Supersede with updated circle
                cmd = AddCircleCommand(
                    self.sketch, center, radius,
                    supersede=True
                )
                self.scene.execute(cmd)

                # Show radius in status bar
                if self.app.session.mode == config.MODE_EDITOR:
                    self.app.session.status.set(f"Radius: {radius:.2f}")
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            # Check if this was a quick click (< threshold)
            elapsed = time.time() - self.click_time
            if elapsed < self.QUICK_CLICK_THRESHOLD and not self.click_click_mode:
                # Enter click-click mode - don't finalize yet
                self.click_click_mode = True
                self.app.session.status.set("Click to set radius")
                return True

            # Normal drag release - finalize the circle
            mx, my = event.pos
            return self._finalize_circle(mx, my, layout)

        return False

    def _finalize_circle(self, mx, my, layout):
        """Finalize circle creation with snap constraints."""
        cx, cy = self.get_world_pos(mx, my, layout)
        center = self.start_pos
        radius = max(0.1, math.hypot(cx - center[0], cy - center[1]))

        # Final supersede - stays in undo stack
        cmd = AddCircleCommand(
            self.sketch, center, radius,
            supersede=True
        )
        self.scene.execute(cmd)
        circle_idx = cmd.created_index

        # Add snap constraint if Ctrl held
        if pygame.key.get_mods() & pygame.KMOD_CTRL and self.center_snap:
            c = Coincident(circle_idx, 0, self.center_snap[0], self.center_snap[1])
            self.scene.execute(AddConstraintCommand(self.sketch, c))

        self._finish_drag()
        return True

    def _finish_drag(self):
        """Clean up drag state."""
        self.dragging = False
        self.start_pos = None
        self.center_snap = None
        self.click_click_mode = False
        self.app.session.state = InteractionState.IDLE
        self.app.session.constraint_builder.snap_target = None

    def cancel(self):
        """Cancel current operation - also resets click-click mode."""
        self.click_click_mode = False
        super().cancel()

    def draw_overlay(self, screen, renderer, layout):
        if not self.dragging:
            mx, my = pygame.mouse.get_pos()
            if layout['MID_X'] < mx < layout['RIGHT_X']:
                renderer.draw_tool_point(mx, my)


# =============================================================================
# Point Tool
# =============================================================================

class PointTool(GeometryTool):
    """
    Point creation tool.
    
    Creates a degenerate line (start == end) via single click.
    No drag behavior - immediate creation.
    """
    def __init__(self, app):
        super().__init__(app, "Point")

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._update_hover_snap(mx, my, layout)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                
                # Create point as degenerate line via command
                cmd = AddLineCommand(self.sketch, (sx, sy), (sx, sy), is_ref=False)
                self.scene.execute(cmd)
                wall_idx = cmd.created_index
                
                # Add snap constraint if Ctrl held
                if snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    c = Coincident(wall_idx, 0, snap[0], snap[1])
                    self.scene.execute(AddConstraintCommand(self.sketch, c))
                return True
        return False
    
    def draw_overlay(self, screen, renderer, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['MID_X'] < mx < layout['RIGHT_X']:
            renderer.draw_tool_point(mx, my)


# =============================================================================
# Select Tool
# =============================================================================

class SelectTool(Tool):
    """
    Selection and manipulation tool.
    
    Uses historize pattern (not supersede) for editing:
    - During drag: Commands with historize=False for visual feedback
    - On release: Command with historize=True for undoable commit
    
    Modes:
    - EDIT: Drag individual points
    - RESIZE_CIRCLE: Drag circle edge to resize
    - MOVE_WALL: Drag entire entity
    - MOVE_GROUP: Drag multiple selected entities
    """
    def __init__(self, app):
        super().__init__(app, "Select")
        self._reset_drag_state()

    def _reset_drag_state(self):
        """Reset all drag-related state."""
        self.mode = None
        self.target_idx = -1
        self.target_pt = -1
        self.group_indices = []
        self.drag_start_mouse = None
        self.total_dx = 0.0
        self.total_dy = 0.0
        # For edit/resize commit
        self.original_position = None
        self.original_radius = None
        # For interaction constraint (User Servo)
        self.handle_t = None  # Parametric grab position on line (0.0-1.0)
        # For geometry undo (stores all point positions at drag start)
        self.start_positions = None

    @property
    def sketch(self):
        return self.app.scene.sketch

    @property
    def scene(self):
        return self.app.scene

    def deactivate(self):
        """Called when switching away from SelectTool."""
        super().deactivate()
        # Clear selection when leaving SelectTool
        self.app.session.selection.clear()
        self._reset_drag_state()

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_click(event.pos, layout)
        
        elif event.type == pygame.MOUSEMOTION:
            if self.app.session.state == InteractionState.DRAGGING_GEOMETRY:
                return self._handle_drag(event.pos, layout)
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.app.session.state == InteractionState.DRAGGING_GEOMETRY:
                return self._handle_release(event.pos, layout)
        
        return False

    def cancel(self):
        """Cancel drag operation - restore original state."""
        if self.app.session.state != InteractionState.DRAGGING_GEOMETRY:
            return
        
        # Restore original state based on mode
        if self.mode == 'EDIT' and self.original_position is not None:
            entity = self.sketch.entities[self.target_idx]
            entity.set_point(self.target_pt, np.array(self.original_position))
        
        elif self.mode == 'RESIZE_CIRCLE' and self.original_radius is not None:
            entity = self.sketch.entities[self.target_idx]
            entity.radius = self.original_radius
        
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            # Reverse the accumulated movement
            if self.mode == 'MOVE_GROUP' and self.group_indices:
                for idx in self.group_indices:
                    if 0 <= idx < len(self.sketch.entities):
                        self.sketch.entities[idx].move(-self.total_dx, -self.total_dy)
            elif self.target_idx >= 0:
                if 0 <= self.target_idx < len(self.sketch.entities):
                    self.sketch.entities[self.target_idx].move(-self.total_dx, -self.total_dy)
        
        # Rebuild atoms to match restored geometry
        self.scene.rebuild()

        # Clear interaction data (User Servo cancelled)
        self.sketch.interaction_data = None

        self._reset_drag_state()
        self.app.session.state = InteractionState.IDLE
        self.app.session.status.set("Cancelled")

    def _handle_click(self, mouse_pos, layout):
        """Handle mouse down - determine what was clicked and start drag."""
        mx, my = mouse_pos
        if not (layout['MID_X'] < mx < layout['RIGHT_X']):
            return False

        session = self.app.session
        entities = self.sketch.entities
        builder = session.constraint_builder

        # Build point map for hit testing
        point_map = self._build_point_map(entities, layout)

        # =====================================================================
        # Pending Constraint Mode - intercept clicks for constraint building
        # =====================================================================
        if builder.is_pending:
            # Check for point hit first (for COINCIDENT, MIDPOINT, etc.)
            hit_pt = self._hit_test_points(mx, my, point_map)
            if hit_pt:
                wall_idx, pt_idx = hit_pt
                builder.add_wall(wall_idx)
                builder.add_point(wall_idx, pt_idx)
                self._try_finalize_constraint(builder, session)
                return True

            # Check for entity body hit
            sim_x, sim_y = utils.screen_to_sim(
                mx, my, session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                self.app.sim.world_size, layout
            )
            hit_idx = self.sketch.find_entity_at(sim_x, sim_y, 0.5 / session.camera.zoom)

            if hit_idx >= 0:
                builder.add_wall(hit_idx)
                self._try_finalize_constraint(builder, session)
                return True

            # Clicked empty space - cancel pending constraint
            self._cancel_pending_constraint(builder, session)
            return True

        # =====================================================================
        # Normal Selection Mode
        # =====================================================================

        # 1. Check for point hit (highest priority)
        hit_pt = self._hit_test_points(mx, my, point_map)
        if hit_pt:
            wall_idx, pt_idx = hit_pt
            shift_held = pygame.key.get_mods() & pygame.KMOD_SHIFT

            if shift_held:
                # Shift+Click: toggle point selection
                session.selection.toggle_point(wall_idx, pt_idx)
            else:
                # Regular click: clear other selections, select this point
                session.selection.walls.clear()
                session.selection.points.clear()
                session.selection.select_point(wall_idx, pt_idx)

            self._start_edit_drag(wall_idx, pt_idx, mouse_pos, layout)
            return True

        # 2. Check for circle resize handle
        resize_hit = self._hit_test_circle_resize(mx, my, entities, layout)
        if resize_hit is not None:
            self._start_resize_drag(resize_hit, mouse_pos)
            return True

        # 3. Check for entity body hit
        sim_x, sim_y = utils.screen_to_sim(
            mx, my, session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
            self.app.sim.world_size, layout
        )
        hit_idx = self.sketch.find_entity_at(sim_x, sim_y, 0.5 / session.camera.zoom)

        if hit_idx >= 0:
            shift_held = pygame.key.get_mods() & pygame.KMOD_SHIFT

            # Clear point selection when selecting entities
            if not shift_held:
                session.selection.points.clear()

            if shift_held:
                # Shift+Click: toggle membership in selection (takes priority)
                session.selection.toggle_entity(hit_idx)
                # Only start drag if entity is now selected
                if session.selection.is_entity_selected(hit_idx):
                    if len(session.selection.walls) > 1:
                        self._start_group_move(mouse_pos, layout)
                    else:
                        self._start_entity_move(hit_idx, mouse_pos, layout)
            elif hit_idx in session.selection.walls and len(session.selection.walls) > 1:
                # Click on already-selected entity in group: start group move
                self._start_group_move(mouse_pos, layout)
            else:
                # Regular click: replace selection and move
                session.selection.walls.clear()
                session.selection.points.clear()
                session.selection.walls.add(hit_idx)
                self._start_entity_move(hit_idx, mouse_pos, layout)
            return True
        
        # 4. Clicked empty - deselect
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            session.selection.walls.clear()
            session.selection.points.clear()
        return True

    def _handle_drag(self, mouse_pos, layout):
        """Handle mouse move during drag."""
        mx, my = mouse_pos
        curr_sim = utils.screen_to_sim(
            mx, my,
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
            self.app.sim.world_size, layout
        )
        prev_sim = utils.screen_to_sim(
            self.drag_start_mouse[0], self.drag_start_mouse[1],
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
            self.app.sim.world_size, layout
        )

        dx = curr_sim[0] - prev_sim[0]
        dy = curr_sim[1] - prev_sim[1]
        self.drag_start_mouse = (mx, my)

        # Update interaction data target (User Servo tracks mouse)
        if self.sketch.interaction_data is not None:
            self.sketch.interaction_data['target'] = curr_sim

        if self.mode == 'EDIT':
            return self._handle_edit_drag(mx, my, layout)
        elif self.mode == 'RESIZE_CIRCLE':
            return self._handle_resize_drag(curr_sim)
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            return self._handle_move_drag(dx, dy)

        return False

    def _handle_release(self, mouse_pos, layout):
        """Handle mouse up - commit the operation."""
        if self.mode == 'EDIT':
            self._commit_edit()
        elif self.mode == 'RESIZE_CIRCLE':
            self._commit_resize()
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            self._commit_move()

        # Clear interaction data (User Servo released)
        self.sketch.interaction_data = None

        self._reset_drag_state()
        self.app.session.state = InteractionState.IDLE
        return True

    # -------------------------------------------------------------------------
    # Drag Start Methods
    # -------------------------------------------------------------------------

    def _start_edit_drag(self, wall_idx, pt_idx, mouse_pos, layout):
        """Start editing a specific point."""
        self.mode = 'EDIT'
        self.target_idx = wall_idx
        self.target_pt = pt_idx
        self.drag_start_mouse = mouse_pos

        # Capture original position for cancel/commit
        entity = self.sketch.entities[wall_idx]
        pt = entity.get_point(pt_idx)
        self.original_position = (float(pt[0]), float(pt[1]))

        # Set interaction data for point editing (User Servo)
        sim_pos = utils.screen_to_sim(
            mouse_pos[0], mouse_pos[1],
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
            self.app.sim.world_size, layout
        )
        self.sketch.interaction_data = {
            'entity_idx': wall_idx,
            'point_idx': pt_idx,
            'handle_t': None,
            'target': sim_pos
        }

        self.app.session.state = InteractionState.DRAGGING_GEOMETRY

    def _start_resize_drag(self, circle_idx, mouse_pos):
        """Start resizing a circle."""
        self.mode = 'RESIZE_CIRCLE'
        self.target_idx = circle_idx
        self.drag_start_mouse = mouse_pos

        # Select the circle being resized so it highlights yellow
        self.app.session.selection.walls.clear()
        self.app.session.selection.points.clear()
        self.app.session.selection.walls.add(circle_idx)

        # Capture original radius for cancel/commit
        entity = self.sketch.entities[circle_idx]
        self.original_radius = float(entity.radius)

        self.app.session.state = InteractionState.DRAGGING_GEOMETRY

    def _start_entity_move(self, entity_idx, mouse_pos, layout):
        """Start moving a single entity."""
        self.mode = 'MOVE_WALL'
        self.target_idx = entity_idx
        self.drag_start_mouse = mouse_pos
        self.total_dx = 0.0
        self.total_dy = 0.0

        # Calculate handle_t for lines (parametric grab position)
        entity = self.sketch.entities[entity_idx]
        sim_pos = utils.screen_to_sim(
            mouse_pos[0], mouse_pos[1],
            self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
            self.app.sim.world_size, layout
        )

        # Capture start positions for undo (before any movement)
        self.start_positions = self._capture_entity_positions(entity)

        if isinstance(entity, Line):
            # Calculate t parameter (0.0 = start, 1.0 = end)
            A = entity.start
            B = entity.end
            AB = B - A
            len_sq = np.dot(AB, AB)
            if len_sq > 1e-8:
                AP = np.array(sim_pos) - A
                self.handle_t = np.clip(np.dot(AP, AB) / len_sq, 0.0, 1.0)
            else:
                self.handle_t = 0.5
        else:
            self.handle_t = None

        # Set initial interaction data (User Servo)
        self.sketch.interaction_data = {
            'entity_idx': entity_idx,
            'point_idx': None,
            'handle_t': self.handle_t,
            'target': sim_pos
        }

        self.app.session.state = InteractionState.DRAGGING_GEOMETRY

    def _capture_entity_positions(self, entity):
        """Capture all point positions of an entity for undo."""
        positions = []
        if isinstance(entity, Line):
            positions.append(tuple(entity.start))
            positions.append(tuple(entity.end))
        elif isinstance(entity, Circle):
            positions.append(tuple(entity.center))
        elif hasattr(entity, 'pos'):
            # Point entity
            positions.append(tuple(entity.pos))
        return positions

    def _start_group_move(self, mouse_pos, layout):
        """Start moving multiple selected entities."""
        self.mode = 'MOVE_GROUP'
        self.group_indices = list(self.app.session.selection.walls)
        self.drag_start_mouse = mouse_pos
        self.total_dx = 0.0
        self.total_dy = 0.0

        self.app.session.state = InteractionState.DRAGGING_GEOMETRY

    # -------------------------------------------------------------------------
    # Drag Handling Methods
    # -------------------------------------------------------------------------

    def _handle_edit_drag(self, mx, my, layout):
        """Handle point editing drag using commands."""
        entities = self.sketch.entities
        if self.target_idx >= len(entities):
            return False
        w = entities[self.target_idx]
        
        if isinstance(w, Line):
            anchor = w.end if self.target_pt == 0 else w.start
            mods = pygame.key.get_mods()
            snap_to_points = bool(mods & pygame.KMOD_CTRL)
            constrain_to_axis = bool(mods & pygame.KMOD_SHIFT)
            dest_x, dest_y, snap = utils.get_snapped_pos(
                mx, my, entities,
                self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
                self.app.sim.world_size, self.app.layout, anchor, self.target_idx,
                snap_to_points=snap_to_points, constrain_to_axis=constrain_to_axis
            )
            self.app.session.constraint_builder.snap_target = snap
            
            # Use command with historize=False for preview
            cmd = SetPointCommand(
                self.sketch, self.target_idx, self.target_pt,
                (dest_x, dest_y),
                historize=False
            )
            self.scene.execute(cmd)
        
        elif isinstance(w, Circle):
            mods = pygame.key.get_mods()
            snap_to_points = bool(mods & pygame.KMOD_CTRL)
            constrain_to_axis = bool(mods & pygame.KMOD_SHIFT)
            dest_x, dest_y, snap = utils.get_snapped_pos(
                mx, my, entities,
                self.app.session.camera.zoom, self.app.session.camera.pan_x, self.app.session.camera.pan_y,
                self.app.sim.world_size, self.app.layout, None, self.target_idx,
                snap_to_points=snap_to_points, constrain_to_axis=constrain_to_axis
            )
            self.app.session.constraint_builder.snap_target = snap
            
            # Use command with historize=False for preview
            cmd = SetPointCommand(
                self.sketch, self.target_idx, 0,
                (dest_x, dest_y),
                historize=False
            )
            self.scene.execute(cmd)
        
        return True

    def _handle_resize_drag(self, curr_sim):
        """Handle circle resize drag using commands."""
        w = self.sketch.entities[self.target_idx]
        if isinstance(w, Circle):
            new_r = math.hypot(curr_sim[0] - w.center[0], curr_sim[1] - w.center[1])
            new_r = max(0.1, new_r)
            
            # Use command with historize=False for preview
            cmd = SetCircleRadiusCommand(
                self.sketch, self.target_idx, new_r,
                historize=False
            )
            self.scene.execute(cmd)
        return True

    def _handle_move_drag(self, dx, dy):
        """Handle entity/group move drag using commands."""
        self.total_dx += dx
        self.total_dy += dy

        if self.mode == 'MOVE_GROUP':
            # Group move: use uniform translation (no parametric drag)
            cmd = MoveMultipleCommand(
                self.sketch, self.group_indices, dx, dy,
                historize=False
            )
            self.scene.execute(cmd)
        # For MOVE_WALL mode: interaction constraint handles movement via solver
        # Don't apply MoveEntityCommand here - let the User Servo do the work
        # with proper weighted distribution based on handle_t

        return True

    # -------------------------------------------------------------------------
    # Commit Methods
    # -------------------------------------------------------------------------

    def _commit_edit(self):
        """Commit point edit with historized command."""
        if self.target_idx >= len(self.sketch.entities):
            return
        
        entity = self.sketch.entities[self.target_idx]
        current_pt = entity.get_point(self.target_pt)
        final_position = (float(current_pt[0]), float(current_pt[1]))
        
        # Only commit if position actually changed
        if self.original_position and self.original_position != final_position:
            cmd = SetPointCommand(
                self.sketch, self.target_idx, self.target_pt,
                final_position,
                old_position=self.original_position,
                historize=True
            )
            # Add directly to queue (already at final position)
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()
        
        # Handle snap connection
        if self.app.session.constraint_builder.snap_target:
            snap = self.app.session.constraint_builder.snap_target
            if not (self.target_idx == snap[0] and self.target_pt == snap[1]):
                c = Coincident(self.target_idx, self.target_pt, snap[0], snap[1])
                self.scene.execute(AddConstraintCommand(self.sketch, c))
                self.app.session.status.set("Snapped & Connected")

    def _commit_resize(self):
        """Commit circle resize with historized command."""
        if self.target_idx >= len(self.sketch.entities):
            return
        
        entity = self.sketch.entities[self.target_idx]
        if not hasattr(entity, 'radius'):
            return
        
        final_radius = float(entity.radius)
        
        # Only commit if radius actually changed
        if self.original_radius is not None and abs(self.original_radius - final_radius) > 1e-6:
            cmd = SetCircleRadiusCommand(
                self.sketch, self.target_idx,
                final_radius,
                old_radius=self.original_radius,
                historize=True
            )
            # Add directly to queue (already at final radius)
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()

    def _commit_move(self):
        """Commit entity/group move with historized command."""
        if self.mode == 'MOVE_GROUP' and self.group_indices:
            # Group move: still uses delta-based command (uniform translation)
            if abs(self.total_dx) < 1e-6 and abs(self.total_dy) < 1e-6:
                return
            cmd = MoveMultipleCommand(
                self.sketch, self.group_indices,
                self.total_dx, self.total_dy,
                historize=True
            )
            # Add directly to queue (already moved)
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()

        elif self.mode == 'MOVE_WALL' and self.target_idx >= 0:
            # Single entity: use absolute geometry command (handles rotation)
            entity = self.sketch.entities[self.target_idx]
            current_positions = self._capture_entity_positions(entity)

            # Only commit if positions actually changed
            if self.start_positions and current_positions != self.start_positions:
                cmd = SetEntityGeometryCommand(
                    self.sketch, self.target_idx,
                    self.start_positions, current_positions,
                    historize=True
                )
                # Add directly to queue (geometry already at final state)
                self.scene.commands.undo_stack.append(cmd)
                self.scene.commands.redo_stack.clear()

    # -------------------------------------------------------------------------
    # Hit Testing Helpers
    # -------------------------------------------------------------------------

    def _build_point_map(self, entities, layout):
        """Build a map of screen positions to (entity_idx, point_idx) pairs."""
        point_map = {}
        session = self.app.session
        
        for i, w in enumerate(entities):
            if isinstance(w, Line):
                for pt_idx in [0, 1]:
                    pt = w.get_point(pt_idx)
                    sx, sy = utils.sim_to_screen(
                        pt[0], pt[1],
                        session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                        self.app.sim.world_size, layout
                    )
                    key = (int(sx), int(sy))
                    if key not in point_map:
                        point_map[key] = []
                    point_map[key].append((i, pt_idx))
            elif isinstance(w, Circle):
                sx, sy = utils.sim_to_screen(
                    w.center[0], w.center[1],
                    session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                    self.app.sim.world_size, layout
                )
                key = (int(sx), int(sy))
                if key not in point_map:
                    point_map[key] = []
                point_map[key].append((i, 0))
        
        return point_map

    def _hit_test_points(self, mx, my, point_map):
        """Test if mouse hit a point. Returns (entity_idx, point_idx) or None."""
        base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            dist = math.hypot(mx - center_pos[0], my - center_pos[1])
            max_r = base_r + step_r * (len(items) - 1)
            if dist <= max_r:
                # Return first item (could enhance to pick closest)
                return items[0]
        return None

    def _hit_test_circle_resize(self, mx, my, entities, layout):
        """Test if mouse hit a circle's resize handle (edge). Returns entity index or None."""
        session = self.app.session
        
        for i, w in enumerate(entities):
            if isinstance(w, Circle):
                sx, sy = utils.sim_to_screen(
                    w.center[0], w.center[1],
                    session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                    self.app.sim.world_size, layout
                )
                
                # Calculate screen radius
                p0 = utils.sim_to_screen(0, 0, session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                                         self.app.sim.world_size, layout)
                pr = utils.sim_to_screen(w.radius, 0, session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
                                         self.app.sim.world_size, layout)
                screen_r = abs(pr[0] - p0[0])
                
                # Check if near edge (not center)
                dist_from_center = math.hypot(mx - sx, my - sy)
                if abs(dist_from_center - screen_r) < 8:  # 8 pixel tolerance
                    return i

        return None

    # -------------------------------------------------------------------------
    # Constraint Building Helpers
    # -------------------------------------------------------------------------

    def _try_finalize_constraint(self, builder, session):
        """
        Try to finalize the pending constraint.

        Uses ConstraintBuilder.try_build_command() to create the command,
        executes it via Scene, and cleans up UI state on success.
        """
        cmd = builder.try_build_command(self.sketch)
        if cmd:
            # Execute the command and rebuild geometry
            self.scene.execute(cmd)

            # Clean up UI state
            ctype = builder.pending_type
            self._clear_constraint_ui(builder, session)
            session.status.set(f"Applied {ctype}")
            self.app.sound_manager.play_sound('click')
        else:
            # Not ready yet - update status with progress
            session.status.set(builder.get_status_message())
            self.app.sound_manager.play_sound('click')

    def _cancel_pending_constraint(self, builder, session):
        """Cancel the pending constraint and clean up UI state."""
        self._clear_constraint_ui(builder, session)
        session.status.set("Constraint cancelled")

    def _clear_constraint_ui(self, builder, session):
        """Clear all constraint-related UI state."""
        builder.reset()
        session.selection.walls.clear()
        session.selection.points.clear()
        for btn in self.app.input_handler.constraint_btn_map.keys():
            btn.active = False

    def draw_overlay(self, screen, renderer, layout):
        """Draw selection indicators and drag feedback."""
        pass  # Selection highlighting is handled by renderer
