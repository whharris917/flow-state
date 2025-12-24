"""
Tools - Input Handling Tools for the Editor

Each tool handles mouse events for a specific operation:
- BrushTool: Paint/erase particles (physics domain)
- LineTool: Draw lines (via commands)
- RectTool: Draw rectangles (via commands)
- CircleTool: Draw circles (via commands)
- PointTool: Create points (via commands)
- SelectTool: Select and move entities (via commands)

Architecture:
- ALL geometry mutations go through Commands
- Commands with historize=False are used during drag for visual feedback
- Commands with historize=True are used on release for undoable commits
- Tools call scene.rebuild() after geometry changes
- Tools call sketch.solve() for constraint solving during drags
- BrushTool operates on physics (not CAD), uses sim directly
"""

import pygame
import numpy as np
import math
import time

import core.config as config
import core.utils as utils

from model.geometry import Line, Circle
from model.constraints import Coincident, Length, Angle
from core.session import InteractionState

# Commands for proper undo/redo
from core.commands import (
    AddLineCommand, AddCircleCommand, RemoveEntityCommand,
    MoveEntityCommand, MoveMultipleCommand, AddConstraintCommand,
    AddRectangleCommand, CompositeCommand,
    SetPointCommand, SetCircleRadiusCommand
)


class Tool:
    def __init__(self, app, name="Tool"):
        self.app = app
        self.name = name

    def activate(self):
        pass

    def deactivate(self):
        self.app.session.current_snap_target = None

    def handle_event(self, event, layout):
        return False

    def update(self, dt, layout):
        pass

    def draw_overlay(self, screen, renderer, layout):
        pass
    
    def cancel(self):
        pass


class BrushTool(Tool):
    """
    Brush tool for painting/erasing particles.
    
    Note: This operates on PHYSICS (particles), not CAD (geometry).
    Particle operations use Simulation's snapshot system, not Commands.
    """
    def __init__(self, app):
        super().__init__(app, "Brush")
        self.brush_radius = 5.0 

    def update(self, dt, layout):
        if self.app.session.state == InteractionState.PAINTING:
            mx, my = pygame.mouse.get_pos()
            if layout['LEFT_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT:
                sim = self.app.sim
                sx, sy = utils.screen_to_sim(
                    mx, my, 
                    self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
                    sim.world_size, layout
                )
                
                radius = self.brush_radius
                
                if pygame.mouse.get_pressed()[0]: 
                    sim.add_particles_brush(sx, sy, radius)
                elif pygame.mouse.get_pressed()[2]: 
                    sim.delete_particles_brush(sx, sy, radius)

    def handle_event(self, event, layout):
        if self.app.session.mode != config.MODE_SIM:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if not (layout['MID_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT):
                return False

            if event.button == 1:
                self.app.session.state = InteractionState.PAINTING
                self.app.sim.snapshot()  # Physics snapshot (not CAD)
                return True
            elif event.button == 3:
                self.app.session.state = InteractionState.PAINTING
                self.app.sim.snapshot()
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.app.session.state == InteractionState.PAINTING:
                self.app.session.state = InteractionState.IDLE
                return True
        return False

    def draw_overlay(self, screen, renderer, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['MID_X'] < mx < layout['RIGHT_X'] and config.TOP_MENU_H < my < config.WINDOW_HEIGHT:
            zoom = self.app.session.zoom
            world_size = self.app.sim.world_size
            base_scale = (layout['MID_W'] - 50) / world_size
            final_scale = base_scale * zoom
            screen_r = self.brush_radius * final_scale
            
            renderer.draw_tool_brush(mx, my, screen_r)


class GeometryTool(Tool):
    """
    Base class for geometry creation tools.
    Uses Commands for undoable CAD operations.
    """
    def __init__(self, app, name):
        super().__init__(app, name)
        self.dragging = False
        self.start_pos = None
        self.created_indices = []
        self.pending_commands = []

    @property
    def sketch(self):
        return self.app.scene.sketch
    
    @property
    def scene(self):
        return self.app.scene

    def get_world_pos(self, mx, my, layout):
        return utils.screen_to_sim(
            mx, my, 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout
        )

    def get_snapped(self, mx, my, layout, anchor=None, exclude_idx=-1):
        return utils.get_snapped_pos(
            mx, my, 
            self.sketch.entities, 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout, anchor, exclude_idx
        )

    def cancel(self):
        """Cancel current operation - undo any pending geometry."""
        if self.dragging:
            # Remove any geometry created during this drag
            for idx in sorted(self.created_indices, reverse=True):
                if idx < len(self.sketch.entities):
                    self.sketch.entities.pop(idx)
            self.dragging = False
            self.created_indices = []
            self.pending_commands = []
            self.scene.rebuild()
            self.app.session.set_status("Cancelled")
        self.app.session.current_snap_target = None

    def _update_hover_snap(self, mx, my, layout):
        if not self.dragging:
            _, _, snap = self.get_snapped(mx, my, layout, anchor=None)
            self.app.session.current_snap_target = snap


class LineTool(GeometryTool):
    """
    Line drawing tool.
    Uses AddLineCommand for undoable line creation.
    """
    def __init__(self, app):
        super().__init__(app, "Line")
        self.current_wall_idx = -1
        self.start_snap = None

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                self.start_snap = snap
                self.start_pos = (sx, sy)
                
                # Create preview line for visual feedback during drag
                is_ref = (self.name == "Ref Line")
                self.sketch.add_line((sx, sy), (sx, sy), is_ref=is_ref)
                self.current_wall_idx = len(self.sketch.entities) - 1
                self.created_indices = [self.current_wall_idx]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                entities = self.sketch.entities
                if self.current_wall_idx < len(entities):
                    w = entities[self.current_wall_idx]
                    sx, sy, snap = self.get_snapped(mx, my, layout, w.start, self.current_wall_idx)
                    
                    w.end[:] = (sx, sy)
                    self.app.session.current_snap_target = snap
                    self.scene.rebuild()
                    
                    if self.app.session.mode == config.MODE_EDITOR:
                        self.sketch.solve()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            mx, my = event.pos
            entities = self.sketch.entities
            
            if self.current_wall_idx < len(entities):
                w = entities[self.current_wall_idx]
                
                # Check if line is too short (degenerate)
                if np.linalg.norm(w.end - w.start) < 1e-4:
                    self.sketch.entities.pop(self.current_wall_idx)
                    self.dragging = False
                    self.created_indices = []
                    self.app.session.state = InteractionState.IDLE
                    self.scene.rebuild()
                    return True

                # Line is valid - finalize with command
                sx, sy, end_snap = self.get_snapped(mx, my, layout, w.start, self.current_wall_idx)
                
                # Remove the preview line (we'll recreate via command)
                start = tuple(w.start)
                end = tuple(w.end)
                is_ref = w.ref
                material_id = w.material_id
                self.sketch.entities.pop(self.current_wall_idx)
                
                # Create line via command
                line_cmd = AddLineCommand(self.sketch, start, end, is_ref=is_ref, material_id=material_id)
                self.scene.execute(line_cmd)
                line_idx = line_cmd.created_index
                
                # Add snap constraints if Ctrl was held
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if self.start_snap:
                        c = Coincident(line_idx, 0, self.start_snap[0], self.start_snap[1])
                        self.scene.execute(AddConstraintCommand(self.sketch, c))
                    
                    if end_snap:
                        c = Coincident(line_idx, 1, end_snap[0], end_snap[1])
                        self.scene.execute(AddConstraintCommand(self.sketch, c))
                
                # Add orientation constraints if Shift was held
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    dx = abs(start[0] - end[0])
                    dy = abs(start[1] - end[1])
                    if dy < 0.001:
                        self.scene.execute(AddConstraintCommand(self.sketch, Angle('HORIZONTAL', line_idx)))
                    elif dx < 0.001:
                        self.scene.execute(AddConstraintCommand(self.sketch, Angle('VERTICAL', line_idx)))

            self.dragging = False
            self.created_indices = []
            self.start_snap = None
            self.app.session.current_snap_target = None
            self.app.session.state = InteractionState.IDLE
            self.sketch.solve()
            self.scene.rebuild()
            return True
        return False


class RectTool(GeometryTool):
    """
    Rectangle drawing tool.
    Uses AddRectangleCommand for undoable rectangle creation.
    """
    def __init__(self, app):
        super().__init__(app, "Rectangle")
        self.base_idx = -1
        self.rect_start_pos = None

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, _ = self.get_snapped(mx, my, layout)
                self.rect_start_pos = (sx, sy)
                
                # Create 4 preview lines
                self.base_idx = len(self.sketch.entities)
                for _ in range(4):
                    self.sketch.add_line((sx, sy), (sx, sy))
                self.created_indices = [self.base_idx + i for i in range(4)]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cur_x, cur_y = self.get_world_pos(mx, my, layout)
                sx, sy = self.rect_start_pos
                
                entities = self.sketch.entities
                if self.base_idx + 3 < len(entities):
                    entities[self.base_idx].start[:] = (sx, sy)
                    entities[self.base_idx].end[:] = (cur_x, sy)
                    entities[self.base_idx+1].start[:] = (cur_x, sy)
                    entities[self.base_idx+1].end[:] = (cur_x, cur_y)
                    entities[self.base_idx+2].start[:] = (cur_x, cur_y)
                    entities[self.base_idx+2].end[:] = (sx, cur_y)
                    entities[self.base_idx+3].start[:] = (sx, cur_y)
                    entities[self.base_idx+3].end[:] = (sx, sy)
                    self.scene.rebuild()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            mx, my = event.pos
            cur_x, cur_y = self.get_world_pos(mx, my, layout)
            sx, sy = self.rect_start_pos
            
            # Remove preview lines
            for _ in range(4):
                if self.base_idx < len(self.sketch.entities):
                    self.sketch.entities.pop(self.base_idx)
            
            # Create rectangle via command (includes constraints)
            cmd = AddRectangleCommand(self.sketch, sx, sy, cur_x, cur_y)
            self.scene.execute(cmd)
            
            self.dragging = False
            self.created_indices = []
            self.app.session.state = InteractionState.IDLE
            return True
        return False


class CircleTool(GeometryTool):
    """
    Circle drawing tool.
    Uses AddCircleCommand for undoable circle creation.
    """
    def __init__(self, app):
        super().__init__(app, "Circle")
        self.circle_idx = -1
        self.center_snap = None

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sx, sy, snap = self.get_snapped(mx, my, layout)
                self.center_snap = snap
                self.start_pos = (sx, sy)
                
                # Create preview circle
                self.sketch.add_circle((sx, sy), 0.1)
                self.circle_idx = len(self.sketch.entities) - 1
                self.created_indices = [self.circle_idx]
                self.dragging = True
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.dragging:
                cur_x, cur_y = self.get_world_pos(mx, my, layout)
                entities = self.sketch.entities
                if self.circle_idx < len(entities):
                    c = entities[self.circle_idx]
                    c.radius = max(0.1, math.hypot(cur_x - c.center[0], cur_y - c.center[1]))
                    self.scene.rebuild()
                return True
            else:
                self._update_hover_snap(mx, my, layout)
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            entities = self.sketch.entities
            if self.circle_idx < len(entities):
                c = entities[self.circle_idx]
                center = tuple(c.center)
                radius = c.radius
                material_id = c.material_id
                
                # Remove preview
                self.sketch.entities.pop(self.circle_idx)
                
                # Create via command
                cmd = AddCircleCommand(self.sketch, center, radius, material_id=material_id)
                self.scene.execute(cmd)
                circle_idx = cmd.created_index
                
                # Add snap constraint if Ctrl held
                if pygame.key.get_mods() & pygame.KMOD_CTRL and self.center_snap:
                    c = Coincident(circle_idx, 0, self.center_snap[0], self.center_snap[1])
                    self.scene.execute(AddConstraintCommand(self.sketch, c))

            self.dragging = False
            self.created_indices = []
            self.center_snap = None
            self.app.session.state = InteractionState.IDLE
            self.app.session.current_snap_target = None
            return True
        return False


class PointTool(GeometryTool):
    """
    Point creation tool.
    Creates a degenerate line (start == end) via command.
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


class SelectTool(Tool):
    """
    Selection and manipulation tool.
    
    ALL geometry mutations go through Commands:
    - During drag: Commands with historize=False for visual feedback
    - On release: Command with historize=True for undoable commit
    
    This ensures all mutations are tracked and the final state is undoable.
    """
    def __init__(self, app):
        super().__init__(app, "Select")
        self.mode = None 
        self.target_idx = -1
        self.target_pt = -1
        self.group_indices = []
        self.drag_start_mouse = (0, 0)
        self.drag_start_sim = (0, 0)
        self.drag_rect = None
        self.last_click_time = 0.0
        self.last_click_pos = (0, 0)
        self.DOUBLE_CLICK_TIME = 0.3
        self.DOUBLE_CLICK_DIST = 5
        
        # === DRAG STATE FOR PROPER UNDO ===
        # Store original state at drag start for proper undo
        self.original_position = None      # For EDIT mode (point position)
        self.original_radius = None        # For RESIZE_CIRCLE mode
        self.original_positions = {}       # For MOVE_WALL/MOVE_GROUP mode: {entity_idx: [(pt_idx, pos), ...]}
        
        # Track accumulated movement for commit command
        self.total_dx = 0.0
        self.total_dy = 0.0

    @property
    def sketch(self):
        return self.app.scene.sketch
    
    @property
    def scene(self):
        return self.app.scene

    def handle_event(self, event, layout):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                if event.button == 1: 
                    return self._handle_left_click(mx, my, layout)
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mode:
                return self._handle_drag(event.pos, layout)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.mode:
                return self._handle_release(event.pos, layout)
                
        return False

    def cancel(self):
        """Cancel current drag and restore original state."""
        if self.mode == 'EDIT' and self.original_position is not None:
            # Restore original point position
            if self.target_idx < len(self.sketch.entities):
                entity = self.sketch.entities[self.target_idx]
                entity.set_point(self.target_pt, np.array(self.original_position))
                self.scene.rebuild()
                
        elif self.mode == 'RESIZE_CIRCLE' and self.original_radius is not None:
            # Restore original radius
            if self.target_idx < len(self.sketch.entities):
                entity = self.sketch.entities[self.target_idx]
                if hasattr(entity, 'radius'):
                    entity.radius = self.original_radius
                self.scene.rebuild()
                
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP'] and self.original_positions:
            # Restore original positions
            for idx, points in self.original_positions.items():
                if idx < len(self.sketch.entities):
                    entity = self.sketch.entities[idx]
                    for pt_idx, pos in points:
                        entity.set_point(pt_idx, np.array(pos))
            self.scene.rebuild()
        
        self._reset_drag_state()
        self.app.session.state = InteractionState.IDLE
        self.app.session.set_status("Cancelled")

    def _reset_drag_state(self):
        """Reset all drag-related state."""
        self.mode = None
        self.group_indices = []
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.original_position = None
        self.original_radius = None
        self.original_positions = {}
        self.app.session.current_snap_target = None
        
        # Remove temp constraint
        if self.app.session.temp_constraint_active:
            self.sketch.constraints = [c for c in self.sketch.constraints if not c.temp]
            self.app.session.temp_constraint_active = False

    def _capture_entity_positions(self, entity_indices):
        """Capture all point positions for a set of entities."""
        positions = {}
        for idx in entity_indices:
            if idx < len(self.sketch.entities):
                entity = self.sketch.entities[idx]
                points = []
                if isinstance(entity, Line):
                    points.append((0, tuple(entity.start)))
                    points.append((1, tuple(entity.end)))
                elif isinstance(entity, Circle):
                    points.append((0, tuple(entity.center)))
                elif hasattr(entity, 'pos'):
                    points.append((0, tuple(entity.pos)))
                positions[idx] = points
        return positions

    def _handle_left_click(self, mx, my, layout):
        now = time.time()
        is_double = False
        dist = math.hypot(mx - self.last_click_pos[0], my - self.last_click_pos[1])
        if (now - self.last_click_time < self.DOUBLE_CLICK_TIME) and (dist < self.DOUBLE_CLICK_DIST):
            is_double = True
        
        self.last_click_time = now
        self.last_click_pos = (mx, my)

        # 1. Check Points
        point_map = utils.get_grouped_points(
            self.sketch.entities, 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout
        )
        hit_pt = self._hit_test_points(mx, my, point_map)
        
        if hit_pt:
            if hit_pt in self.app.session.selected_points:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._select_point(hit_pt) 
            else:
                self._select_point(hit_pt)

            self.mode = 'EDIT'
            self.target_idx = hit_pt[0]
            self.target_pt = hit_pt[1]
            self.drag_start_mouse = (mx, my)
            self.drag_start_sim = utils.screen_to_sim(
                mx, my, 
                self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
                self.app.sim.world_size, layout
            )
            
            # === CAPTURE ORIGINAL POSITION ===
            if self.target_idx < len(self.sketch.entities):
                entity = self.sketch.entities[self.target_idx]
                pt = entity.get_point(self.target_pt)
                self.original_position = (float(pt[0]), float(pt[1]))
            
            self.total_dx = 0.0
            self.total_dy = 0.0
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True
            
        # 2. Check Walls
        hit_wall = self._hit_test_walls(mx, my, layout)
        if hit_wall != -1:
            w = self.sketch.entities[hit_wall]
            
            if is_double:
                group = utils.get_connected_group(self.sketch.constraints, hit_wall)
                self.app.session.selected_walls.update(group)
                self.app.session.set_status("Group Selected")
                return True

            if hit_wall in self.app.session.selected_walls:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._select_wall(hit_wall)
            else:
                if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.app.session.selected_walls.clear()
                    self.app.session.selected_points.clear()
                self._select_wall(hit_wall)
            
            if isinstance(w, Circle):
                self.mode = 'RESIZE_CIRCLE'
                self.target_idx = hit_wall
                self.drag_start_mouse = (mx, my)
                
                # === CAPTURE ORIGINAL RADIUS ===
                self.original_radius = float(w.radius)
                
                self.app.session.state = InteractionState.DRAGGING_GEOMETRY
                return True
            
            is_part_of_selection = hit_wall in self.app.session.selected_walls
            if is_part_of_selection and len(self.app.session.selected_walls) > 1:
                self.mode = 'MOVE_GROUP'
                self.group_indices = list(self.app.session.selected_walls)
                
                # === CAPTURE ORIGINAL POSITIONS FOR ALL GROUP ENTITIES ===
                self.original_positions = self._capture_entity_positions(self.group_indices)
            else:
                self.mode = 'MOVE_WALL'
                self.target_idx = hit_wall
                
                # === CAPTURE ORIGINAL POSITIONS ===
                self.original_positions = self._capture_entity_positions([hit_wall])
                
                # Add temp length constraint for editor mode
                if isinstance(w, Line) and self.app.session.mode == config.MODE_EDITOR:
                    l = np.linalg.norm(w.end - w.start)
                    c = Length(hit_wall, l)
                    c.temp = True
                    self.sketch.constraints.append(c)
                    self.app.session.temp_constraint_active = True
            
            self.drag_start_mouse = (mx, my)
            self.drag_start_sim = utils.screen_to_sim(
                mx, my, 
                self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
                self.app.sim.world_size, layout
            )
            self.total_dx = 0.0
            self.total_dy = 0.0
            self.app.session.state = InteractionState.DRAGGING_GEOMETRY
            return True
            
        # 3. Clicked Empty -> Deselect
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.app.session.selected_walls.clear()
            self.app.session.selected_points.clear()
        return True

    def _handle_drag(self, mouse_pos, layout):
        mx, my = mouse_pos
        curr_sim = utils.screen_to_sim(
            mx, my, 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout
        )
        prev_sim = utils.screen_to_sim(
            self.drag_start_mouse[0], self.drag_start_mouse[1], 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout
        )
        
        dx = curr_sim[0] - prev_sim[0]
        dy = curr_sim[1] - prev_sim[1]
        self.drag_start_mouse = (mx, my)
        
        if self.mode == 'EDIT':
            return self._handle_edit_drag(mx, my, dx, dy, layout)
        elif self.mode == 'RESIZE_CIRCLE':
            return self._handle_resize_drag(mx, my, curr_sim, layout)
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            return self._handle_move_drag(dx, dy, layout)
            
        return False

    def _handle_edit_drag(self, mx, my, dx, dy, layout):
        """Handle point editing drag using commands."""
        entities = self.sketch.entities
        if self.target_idx >= len(entities):
            return False
        w = entities[self.target_idx]
        
        if isinstance(w, Line):
            anchor = w.end if self.target_pt == 0 else w.start
            dest_x, dest_y, snap = utils.get_snapped_pos(
                mx, my, entities, 
                self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
                self.app.sim.world_size, self.app.layout, anchor, self.target_idx
            )
            
            self.app.session.current_snap_target = snap
            
            # Use command for mutation (historize=False for drag preview)
            cmd = SetPointCommand(
                self.sketch, self.target_idx, self.target_pt,
                (dest_x, dest_y),
                historize=False
            )
            self.scene.execute(cmd)
        
        elif isinstance(w, Circle):
            dest_x, dest_y, snap = utils.get_snapped_pos(
                mx, my, entities, 
                self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
                self.app.sim.world_size, self.app.layout, None, self.target_idx
            )
            self.app.session.current_snap_target = snap
            
            # Use command for mutation (historize=False for drag preview)
            cmd = SetPointCommand(
                self.sketch, self.target_idx, 0,
                (dest_x, dest_y),
                historize=False
            )
            self.scene.execute(cmd)
        
        self.scene.rebuild()
        if self.app.session.mode == config.MODE_EDITOR:
            self.sketch.solve()
        return True

    def _handle_resize_drag(self, mx, my, curr_sim, layout):
        """Handle circle resize drag using commands."""
        w = self.sketch.entities[self.target_idx]
        if isinstance(w, Circle):
            new_r = math.hypot(curr_sim[0] - w.center[0], curr_sim[1] - w.center[1])
            new_r = max(0.1, new_r)
            
            # Use command for mutation (historize=False for drag preview)
            cmd = SetCircleRadiusCommand(
                self.sketch, self.target_idx, new_r,
                historize=False
            )
            self.scene.execute(cmd)
            self.scene.rebuild()
        return True

    def _handle_move_drag(self, dx, dy, layout):
        """Handle entity/group move drag using commands."""
        self.total_dx += dx
        self.total_dy += dy
        
        if self.mode == 'MOVE_GROUP':
            # Use command for mutation (historize=False for drag preview)
            cmd = MoveMultipleCommand(
                self.sketch, self.group_indices, dx, dy,
                historize=False
            )
            self.scene.execute(cmd)
        else:
            # Single entity move
            cmd = MoveEntityCommand(
                self.sketch, self.target_idx, dx, dy,
                historize=False
            )
            self.scene.execute(cmd)
        
        self.scene.rebuild()
        if self.app.session.mode == config.MODE_EDITOR:
            self.sketch.solve()
        return True

    def _handle_release(self, mouse_pos, layout):
        mx, my = mouse_pos
        
        if self.mode == 'EDIT':
            self._commit_edit(mx, my, layout)
        elif self.mode == 'RESIZE_CIRCLE':
            self._commit_resize()
        elif self.mode in ['MOVE_WALL', 'MOVE_GROUP']:
            self._commit_move()
        
        self._reset_drag_state()
        self.app.session.state = InteractionState.IDLE
        self.sketch.solve()
        self.scene.rebuild()
        return True

    def _commit_edit(self, mx, my, layout):
        """Commit point edit with historized command."""
        if self.target_idx >= len(self.sketch.entities):
            return
        
        entity = self.sketch.entities[self.target_idx]
        current_pt = entity.get_point(self.target_pt)
        final_position = (float(current_pt[0]), float(current_pt[1]))
        
        # Only commit if position actually changed
        if self.original_position and self.original_position != final_position:
            # Create historized command with explicit original position
            cmd = SetPointCommand(
                self.sketch, self.target_idx, self.target_pt,
                final_position,
                old_position=self.original_position,
                historize=True
            )
            # Execute directly on queue (not through scene.execute to avoid double-execution)
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()
        
        # Handle snap connection
        if self.app.session.current_snap_target:
            snap = self.app.session.current_snap_target
            if not (self.target_idx == snap[0] and self.target_pt == snap[1]):
                c = Coincident(
                    self.target_idx, self.target_pt,
                    snap[0], snap[1]
                )
                self.scene.execute(AddConstraintCommand(self.sketch, c))
                self.app.session.set_status("Snapped & Connected")

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
            # Add directly to queue
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()

    def _commit_move(self):
        """Commit entity/group move with historized command."""
        # Only commit if there was actual movement
        if abs(self.total_dx) < 1e-6 and abs(self.total_dy) < 1e-6:
            return
        
        if self.mode == 'MOVE_GROUP' and self.group_indices:
            # For group move, we need to store the reverse delta
            cmd = MoveMultipleCommand(
                self.sketch, self.group_indices,
                self.total_dx, self.total_dy,
                historize=True
            )
            # The entities are already moved, so we just record the command
            # Set dx/dy so undo will reverse the full movement
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()
            
        elif self.mode == 'MOVE_WALL' and self.target_idx >= 0:
            cmd = MoveEntityCommand(
                self.sketch, self.target_idx,
                self.total_dx, self.total_dy,
                historize=True
            )
            self.scene.commands.undo_stack.append(cmd)
            self.scene.commands.redo_stack.clear()

    def _hit_test_points(self, mx, my, point_map):
        base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            dist = math.hypot(mx - center_pos[0], my - center_pos[1])
            max_r = base_r + (len(items) - 1) * step_r
            if dist <= max_r:
                if dist < base_r:
                    return items[0]
                k = int((dist - base_r) / step_r) + 1
                if k < len(items):
                    return items[k]
                return items[-1]
        return None

    def _hit_test_walls(self, mx, my, layout):
        sim_x, sim_y = utils.screen_to_sim(
            mx, my, 
            self.app.session.zoom, self.app.session.pan_x, self.app.session.pan_y, 
            self.app.sim.world_size, layout
        )
        rad_sim = 5.0 / (((layout['MID_W'] - 50) / self.app.sim.world_size) * self.app.session.zoom)
        
        entities = self.sketch.entities
        for i, w in enumerate(entities):
            if isinstance(w, Line):
                if np.array_equal(w.start, w.end):
                    continue
                p1, p2 = w.start, w.end
                p3 = np.array([sim_x, sim_y])
                d_vec = p2 - p1
                len_sq = np.dot(d_vec, d_vec)
                if len_sq == 0:
                    dist = np.linalg.norm(p3 - p1)
                else:
                    t = max(0, min(1, np.dot(p3 - p1, d_vec) / len_sq))
                    proj = p1 + t * d_vec
                    dist = np.linalg.norm(p3 - proj)
                if dist < rad_sim:
                    return i
            elif isinstance(w, Circle):
                d = math.hypot(sim_x - w.center[0], sim_y - w.center[1])
                if abs(d - w.radius) < rad_sim:
                    return i
        return -1

    def _select_point(self, pt_tuple):
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            if pt_tuple not in self.app.session.selected_points:
                self.app.session.selected_walls.clear()
                self.app.session.selected_points.clear()
        
        if pt_tuple in self.app.session.selected_points:
            self.app.session.selected_points.remove(pt_tuple)
        else:
            self.app.session.selected_points.add(pt_tuple)

    def _select_wall(self, wall_idx):
        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            if wall_idx not in self.app.session.selected_walls:
                self.app.session.selected_walls.clear()
                self.app.session.selected_points.clear()
        
        if wall_idx in self.app.session.selected_walls:
            self.app.session.selected_walls.remove(wall_idx)
        else:
            self.app.session.selected_walls.add(wall_idx)