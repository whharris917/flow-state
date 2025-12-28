"""
Renderer - Main Drawing Module

Handles all rendering for the application including:
- Viewport and grid (Clipped to SceneViewport)
- Particles
- Geometry (Lines, Circles, Points)
- Constraints
- ProcessObjects (Sources, etc.)
- UI overlays
"""

import pygame
import math
import numpy as np
import core.config as config

from core.utils import sim_to_screen, screen_to_sim, get_grouped_points, calculate_current_temp
from model.geometry import Line, Circle, Point


class Renderer:
    def __init__(self, screen, font, big_font):
        self.screen = screen
        self.font = font
        self.big_font = big_font

    def draw_app(self, app, layout, ui_list):
        """Main draw method - renders the entire application state."""
        session = app.session 
        sim = app.sim
        sketch = app.sketch
        
        self.screen.fill(config.BACKGROUND_COLOR)
        
        # 1. Determine Viewport Rect from UI Tree
        viewport_rect = None
        if hasattr(app.ui, 'scene_viewport'):
            viewport_rect = app.ui.scene_viewport.rect
        
        # 2. Draw Scene Content
        self._draw_viewport(session, sim, layout, viewport_rect)
        
        if session.mode == config.MODE_SIM:
            self._draw_particles(session, sim, layout)
        else:
            self._draw_editor_guides(session, sim, layout)
        
        if hasattr(app, 'scene') and hasattr(app.scene, 'process_objects'):
            self._draw_process_objects(app.scene.process_objects, session, layout, sim.world_size)
            
        self._draw_geometry(session, sketch, layout, world_size=sim.world_size)
        
        if session.current_tool:
            session.current_tool.draw_overlay(self.screen, self, layout)
        
        if session.show_constraints:
            self._draw_constraints(session, sketch, layout, world_size=sim.world_size)
            self._draw_editor_overlays(session, sketch, layout, world_size=sim.world_size)
            
        if session.placing_geo_data:
            self._draw_placement_preview(session, sim, layout)

        self._draw_snap_indicator(session, sim, sketch, layout)

        # 3. Unclip and Draw UI
        self.screen.set_clip(None)
        
        if session.mode == config.MODE_SIM:
            self._draw_stats(session, sim, layout)
        
        # Draw UI Tree (Panels, Buttons, Menus, Status Bar)
        for el in ui_list:
            el.draw(self.screen, self.font)

    # =========================================================================
    # ProcessObject Rendering
    # =========================================================================

    def _draw_process_objects(self, process_objects, session, layout, world_size):
        """Render all ProcessObjects (Sources, Sinks, etc.)."""
        for pobj in process_objects:
            geometries = pobj.get_geometry_for_rendering()
            
            for geom in geometries:
                if geom['type'] == 'dashed_circle':
                    self._draw_dashed_circle_entity(
                        geom['center'],
                        geom['radius'],
                        geom.get('color', config.COLOR_SOURCE),
                        pobj.enabled,
                        session,
                        layout,
                        world_size
                    )

    def _draw_dashed_circle_entity(self, center, radius, color, enabled, session, layout, world_size):
        """Draw a dashed circle for ProcessObjects like Sources."""
        transform = lambda x, y: sim_to_screen(
            x, y, session.camera.zoom, session.camera.pan_x, 
            session.camera.pan_y, world_size, layout
        )
        
        sx, sy = transform(center[0], center[1])
        p0 = transform(0, 0)
        pr = transform(radius, 0)
        screen_radius = abs(pr[0] - p0[0])
        
        draw_color = color if enabled else config.COLOR_SOURCE_DISABLED
        
        num_dashes = max(8, int(screen_radius / 6))
        dash_angle = math.pi / num_dashes
        
        for i in range(num_dashes):
            start_angle = i * 2 * dash_angle
            end_angle = start_angle + dash_angle
            
            x1 = sx + screen_radius * math.cos(start_angle)
            y1 = sy + screen_radius * math.sin(start_angle)
            x2 = sx + screen_radius * math.cos(end_angle)
            y2 = sy + screen_radius * math.sin(end_angle)
            
            pygame.draw.line(self.screen, draw_color, (int(x1), int(y1)), (int(x2), int(y2)), 2)

    def draw_source_preview(self, center_screen, radius_screen, mouse_pos):
        """Draw preview during Source tool placement."""
        num_dashes = max(8, int(radius_screen / 6))
        dash_angle = math.pi / num_dashes
        color = config.COLOR_SOURCE
        
        cx, cy = center_screen
        
        for i in range(num_dashes):
            start_angle = i * 2 * dash_angle
            end_angle = start_angle + dash_angle
            
            x1 = cx + radius_screen * math.cos(start_angle)
            y1 = cy + radius_screen * math.sin(start_angle)
            x2 = cx + radius_screen * math.cos(end_angle)
            y2 = cy + radius_screen * math.sin(end_angle)
            
            pygame.draw.line(self.screen, color, (int(x1), int(y1)), (int(x2), int(y2)), 2)
        
        pygame.draw.line(self.screen, color, center_screen, mouse_pos, 1)
        pygame.draw.circle(self.screen, color, center_screen, 4)

    # =========================================================================
    # Snap Indicator
    # =========================================================================

    def _draw_snap_indicator(self, session, sim, sketch, layout):
        if session.constraint_builder.snap_target:
            w_idx, pt_idx = session.constraint_builder.snap_target
            walls = sketch.entities
            if w_idx < len(walls):
                ent = walls[w_idx]
                pt_pos = ent.get_point(pt_idx)
                sx, sy = sim_to_screen(
                    pt_pos[0], pt_pos[1], 
                    session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
                    sim.world_size, layout
                )
                pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 6)
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 8, 1)

    # =========================================================================
    # Viewport & Grid
    # =========================================================================

    def _draw_viewport(self, session, sim, layout, viewport_rect=None):
        """Draw the simulation viewport background and border."""
        if viewport_rect:
            # Clip to the actual SceneViewport from UI Tree
            self.screen.set_clip(viewport_rect)
            
            # Draw Viewport Background/Grid border if needed
            g_col = config.GRID_COLOR if session.mode == config.MODE_SIM else (50, 60, 50)
            pygame.draw.rect(self.screen, g_col, viewport_rect, 2)
            
        else:
            # Fallback (Legacy)
            sim_rect = pygame.Rect(layout['MID_X'], config.TOP_MENU_H, layout['MID_W'], layout['MID_H'])
            self.screen.set_clip(sim_rect)
            g_col = config.GRID_COLOR if session.mode == config.MODE_SIM else (50, 60, 50)
            pygame.draw.rect(self.screen, g_col, sim_rect, 2)
        
        # Note: Grid lines (if any) or content are drawn by subsequent methods,
        # which are now clipped to this rect.

    # =========================================================================
    # Particles
    # =========================================================================

    def _draw_particles(self, session, sim, layout):
        for i in range(sim.count):
            if sim.is_static[i] and not getattr(session, 'show_wall_atoms', True):
                continue 
            
            sx, sy = sim_to_screen(
                sim.pos_x[i], sim.pos_y[i], 
                session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
                sim.world_size, layout
            )
            
            # Basic culling based on viewport approximation
            if layout['MID_X'] < sx < layout['RIGHT_X'] and config.TOP_MENU_H < sy < config.WINDOW_HEIGHT:
                is_stat = sim.is_static[i]
                col = config.COLOR_STATIC if is_stat else config.COLOR_DYNAMIC
                atom_sig = sim.atom_sigma[i]
                rad = max(2, int(atom_sig * config.PARTICLE_RADIUS_SCALE * ((layout['MID_W']-50)/sim.world_size) * session.camera.zoom))
                pygame.draw.circle(self.screen, col, (sx, sy), rad)

    # =========================================================================
    # Editor Guides
    # =========================================================================

    def _draw_editor_guides(self, session, sim, layout):
        cx, cy = sim_to_screen(
            sim.world_size/2, sim.world_size/2, 
            session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
            sim.world_size, layout
        )
        pygame.draw.line(self.screen, (50, 50, 50), (cx-10, cy), (cx+10, cy))
        pygame.draw.line(self.screen, (50, 50, 50), (cx, cy-10), (cx, cy+10))

    # =========================================================================
    # Geometry Drawing
    # =========================================================================

    def _draw_geometry(self, session, sketch, layout, world_size=50.0):
        transform = lambda x, y: sim_to_screen(
            x, y, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
            world_size, layout
        )
        
        walls = sketch.entities
        for i, w in enumerate(walls):
            mat = sketch.materials.get(w.material_id)
            base_color = mat.color if mat else (200, 200, 200)

            is_sel = (i in session.selection.walls)
            is_pend = (session.constraint_builder.pending_type and i in session.constraint_builder.target_walls)

            if isinstance(w, Line):
                self._draw_line_entity(w, transform, is_sel, is_pend, base_color, world_size)
            elif isinstance(w, Circle):
                self._draw_circle_entity(w, transform, is_sel, is_pend, base_color)
            elif isinstance(w, Point):
                self._draw_point_entity(w, transform, is_sel, is_pend, base_color)

    def _draw_line_entity(self, line, transform, is_selected, is_pending, color, world_size=50.0):
        draw_col = color if color else (255, 255, 255)
        width = 1

        if is_selected:
            draw_col = (255, 200, 50)
            width = 3
        elif is_pending:
            draw_col = (100, 255, 100)
            width = 3

        # Calculate screen positions
        if line.ref and line.infinite:
            # Extend line to world boundaries
            ext_start, ext_end = self._extend_line_to_bounds(
                line.start, line.end, world_size
            )
            s1 = transform(ext_start[0], ext_start[1])
            s2 = transform(ext_end[0], ext_end[1])
        else:
            s1 = transform(line.start[0], line.start[1])
            s2 = transform(line.end[0], line.end[1])

        # Draw the line (dashed for ref lines, solid otherwise)
        if line.ref:
            self._draw_dashed_line(draw_col, s1, s2, width)
        else:
            pygame.draw.line(self.screen, draw_col, s1, s2, width)

        # Draw anchor indicators at original positions (not for infinite lines)
        if not (line.ref and line.infinite):
            if line.anchored[0]:
                pygame.draw.circle(self.screen, (255, 50, 50), s1, 3)
            if line.anchored[1]:
                pygame.draw.circle(self.screen, (255, 50, 50), s2, 3)

    def _draw_circle_entity(self, circle, transform, is_selected, is_pending, color):
        sx, sy = transform(circle.center[0], circle.center[1])
        p0 = transform(0, 0)
        pr = transform(circle.radius, 0)
        s_radius = abs(pr[0] - p0[0])
        
        draw_col = color if color else (255, 255, 255)
        width = 1

        if is_selected:
            draw_col = (255, 200, 50)
            width = 3
        elif is_pending:
            draw_col = (100, 255, 100)
            width = 3
        
        if circle.anchored[0]:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(sx), int(sy)), 4)
        pygame.draw.circle(self.screen, draw_col, (int(sx), int(sy)), int(s_radius), width)

    def _draw_point_entity(self, pt, transform, is_selected, is_pending, color):
        sx, sy = transform(pt.pos[0], pt.pos[1])
        draw_col = color if color else (255, 255, 255)
        
        if is_selected:
            draw_col = (0, 255, 255)
        elif is_pending:
            draw_col = (100, 255, 100)
        
        if getattr(pt, 'is_handle', False):
            pygame.draw.circle(self.screen, config.COLOR_SOURCE, (int(sx), int(sy)), 6, 2)
        else:
            if pt.anchored:
                pygame.draw.circle(self.screen, (255, 50, 50), (int(sx), int(sy)), 6)
            pygame.draw.circle(self.screen, draw_col, (int(sx), int(sy)), 4)

    def _extend_line_to_bounds(self, p1, p2, world_size):
        """
        Extend a line segment to the world boundaries.

        Args:
            p1: Start point as numpy array [x, y]
            p2: End point as numpy array [x, y]
            world_size: Size of the world (0 to world_size box)

        Returns:
            Tuple of (extended_start, extended_end) as numpy arrays
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Handle degenerate case (point)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return p1, p2

        # Find t values for intersections with world boundaries
        t_values = []

        # Left boundary (x = 0)
        if abs(dx) > 1e-9:
            t = -p1[0] / dx
            y = p1[1] + t * dy
            if 0 <= y <= world_size:
                t_values.append(t)

        # Right boundary (x = world_size)
        if abs(dx) > 1e-9:
            t = (world_size - p1[0]) / dx
            y = p1[1] + t * dy
            if 0 <= y <= world_size:
                t_values.append(t)

        # Bottom boundary (y = 0)
        if abs(dy) > 1e-9:
            t = -p1[1] / dy
            x = p1[0] + t * dx
            if 0 <= x <= world_size:
                t_values.append(t)

        # Top boundary (y = world_size)
        if abs(dy) > 1e-9:
            t = (world_size - p1[1]) / dy
            x = p1[0] + t * dx
            if 0 <= x <= world_size:
                t_values.append(t)

        if len(t_values) < 2:
            return p1, p2

        # Get the min and max t values for the extended line
        t_min = min(t_values)
        t_max = max(t_values)

        ext_start = np.array([p1[0] + t_min * dx, p1[1] + t_min * dy])
        ext_end = np.array([p1[0] + t_max * dx, p1[1] + t_max * dy])

        return ext_start, ext_end

    def _draw_dashed_line(self, color, start_pos, end_pos, width=1, dash_length=10):
        x1, y1 = start_pos
        x2, y2 = end_pos
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        dash_amount = int(length / dash_length)
        if dash_amount == 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        for i in range(0, dash_amount, 2):
            s = (x1 + dx * i * dash_length, y1 + dy * i * dash_length)
            e = (x1 + dx * (i + 1) * dash_length, y1 + dy * (i + 1) * dash_length)
            pygame.draw.line(self.screen, color, s, e, width)

    # =========================================================================
    # Constraints
    # =========================================================================

    def _draw_constraints(self, session, sketch, layout, world_size=50.0):
        layout_data = self._calculate_constraint_layout(
            sketch.constraints, sketch.entities,
            session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
            world_size, layout
        )

        transform = lambda wx, wy: sim_to_screen(
            wx, wy, session.camera.zoom, session.camera.pan_x, session.camera.pan_y,
            world_size, layout
        )

        # First pass: draw all connectors (behind badges)
        for item in layout_data:
            idx = item['const_idx']
            x, y = item['x'], item['y']
            c = sketch.constraints[idx]
            if c.type in ['EQUAL', 'PARALLEL', 'PERPENDICULAR', 'ANGLE']:
                c1 = self._get_entity_center_screen(c.indices[0], sketch.entities, transform)
                c2 = self._get_entity_center_screen(c.indices[1], sketch.entities, transform)
                self._draw_connector((x, y), c1)
                self._draw_connector((x, y), c2)

        # Second pass: draw all badges (on top of connectors)
        for item in layout_data:
            self._draw_constraint_badge(item, session, sketch.entities, sketch.constraints, layout, world_size)

    def _calculate_constraint_layout(self, constraints, entities, zoom, pan_x, pan_y, world_size, layout):
        transform = lambda x, y: sim_to_screen(x, y, zoom, pan_x, pan_y, world_size, layout)
        
        grouped = {}
        threshold = 20
        layout_data = [] 
        
        for i, c in enumerate(constraints):
            cx, cy = self._get_constraint_raw_pos(c, entities, transform)
            found_group = None
            for key in grouped:
                if math.hypot(key[0]-cx, key[1]-cy) < threshold:
                    found_group = key
                    break
            
            if found_group:
                grouped[found_group].append(i)
            else:
                grouped[(cx, cy)] = [i]
        
        for (gx, gy), indices in grouped.items():
            for k, idx in enumerate(indices):
                offset_x = (k % 3) * 25
                offset_y = (k // 3) * 18
                layout_data.append({
                    'const_idx': idx,
                    'x': gx + offset_x,
                    'y': gy + offset_y,
                    'group_size': len(indices)
                })
        
        return layout_data

    def _get_constraint_raw_pos(self, c, entities, transform):
        t = c.type
        idx = c.indices

        if t == 'COINCIDENT':
            # Offset badge so it doesn't overlap with the point
            px, py = self._get_point_screen(idx[0][0], idx[0][1], entities, transform)
            return (px + 15, py - 15)
        elif t in ['COLLINEAR', 'MIDPOINT']:
            # Offset badge so it doesn't overlap with the point
            px, py = self._get_point_screen(idx[0][0], idx[0][1], entities, transform)
            return (px + 15, py - 15)
        elif t == 'LENGTH':
            return self._get_entity_center_screen(idx[0], entities, transform)
        elif t in ['EQUAL', 'PARALLEL', 'PERPENDICULAR', 'ANGLE']:
            c1 = self._get_entity_center_screen(idx[0], entities, transform)
            c2 = self._get_entity_center_screen(idx[1], entities, transform)
            return ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
        elif t in ['HORIZONTAL', 'VERTICAL']:
            if len(idx) == 1:
                return self._get_entity_center_screen(idx[0], entities, transform)
            else:
                c1 = self._get_entity_center_screen(idx[0], entities, transform)
                c2 = self._get_entity_center_screen(idx[1], entities, transform)
                return ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
        
        return (0, 0)

    def _draw_constraint_badge(self, item, session, entities, constraints, layout, world_size):
        """Draw a single constraint badge (connectors drawn separately in first pass)."""
        idx = item['const_idx']
        x, y = item['x'], item['y']
        c = constraints[idx]

        bg_col = (50, 50, 60)
        border_col = (80, 80, 100)
        text_col = (200, 200, 220)

        is_selected = (idx in session.selection.constraints if hasattr(session.selection, 'constraints') else False)
        if is_selected:
            border_col = (255, 200, 50)

        badge_text = self._get_constraint_badge_text(c)

        text_surf = self.font.render(badge_text, True, text_col)
        tw, th = text_surf.get_size()

        badge_rect = pygame.Rect(x - tw//2 - 4, y - th//2 - 2, tw + 8, th + 4)
        pygame.draw.rect(self.screen, bg_col, badge_rect, border_radius=3)
        pygame.draw.rect(self.screen, border_col, badge_rect, 1, border_radius=3)
        self.screen.blit(text_surf, (x - tw//2, y - th//2))

    def _get_constraint_badge_text(self, c):
        t = c.type
        if t == 'COINCIDENT':
            return 'Co'
        elif t == 'COLLINEAR':
            return '//'
        elif t == 'MIDPOINT':
            return 'Mid'
        elif t == 'LENGTH':
            val = getattr(c, 'value', 0)
            return f'L:{val:.1f}'
        elif t == 'EQUAL':
            return '='
        elif t == 'PARALLEL':
            return '||'
        elif t == 'PERPENDICULAR':
            return '_|_'
        elif t == 'HORIZONTAL':
            return 'H'
        elif t == 'VERTICAL':
            return 'V'
        elif t == 'ANGLE':
            val = getattr(c, 'value', 0)
            return f'<{val:.0f}'
        return '?'

    def _get_entity_center_screen(self, entity_idx, entities, transform):
        if entity_idx < 0 or entity_idx >= len(entities):
            return (0, 0)
        e = entities[entity_idx]
        if hasattr(e, 'start'):  # Line
            p1 = transform(e.start[0], e.start[1])
            p2 = transform(e.end[0], e.end[1])
            return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        elif hasattr(e, 'center'):  # Circle
            return transform(e.center[0], e.center[1])
        elif hasattr(e, 'pos'):  # Point
            return transform(e.pos[0], e.pos[1])
        return (0, 0)

    def _get_point_screen(self, ent_idx, pt_idx, entities, transform):
        if ent_idx < 0 or ent_idx >= len(entities):
            return (0, 0)
        e = entities[ent_idx]
        pt = e.get_point(pt_idx)
        return transform(pt[0], pt[1])

    def _draw_connector(self, p1, p2, color=(100, 100, 100)):
        pygame.draw.line(self.screen, color, p1, p2, 1)

    # =========================================================================
    # Editor Overlays (Point Grouping)
    # =========================================================================

    def _draw_editor_overlays(self, session, sketch, layout, world_size=50.0):
        point_map = get_grouped_points(
            sketch.entities, 
            session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
            world_size, layout
        )
        anchored_points_draw_list = []
        base_r, step_r = 5, 4
        
        for center_pos, items in point_map.items():
            cx, cy = center_pos
            count = len(items)
            for k in range(count - 1, -1, -1):
                w_idx, pt_idx = items[k]
                radius = base_r + (k * step_r)
                
                color = (200, 200, 200)
                if (w_idx, pt_idx) in session.selection.points:
                    color = (255, 200, 50)  # Yellow like selected lines
                elif session.constraint_builder.pending_type and (w_idx, pt_idx) in session.constraint_builder.target_points:
                    color = (100, 255, 100)
                
                pygame.draw.circle(self.screen, (30, 30, 30), (cx, cy), radius)
                pygame.draw.circle(self.screen, color, (cx, cy), radius, 2)
                
                w = sketch.entities[w_idx]
                is_anchored = False
                if isinstance(w, Line):
                    is_anchored = w.anchored[pt_idx]
                elif isinstance(w, Circle):
                    is_anchored = w.anchored[0]
                elif isinstance(w, Point):
                    is_anchored = w.anchored
                
                if is_anchored:
                    anchored_points_draw_list.append((cx, cy))
        
        for pt in anchored_points_draw_list: 
            pygame.draw.circle(self.screen, (255, 50, 50), pt, 3)

    # =========================================================================
    # Placement Preview
    # =========================================================================

    def _draw_placement_preview(self, session, sim, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['LEFT_X'] < mx < layout['RIGHT_X']:
            sim_mx, sim_my = screen_to_sim(
                mx, my, 
                session.camera.zoom, session.camera.pan_x, session.camera.pan_y, 
                sim.world_size, layout
            )
            zoom = session.camera.zoom
            pan_x, pan_y = session.camera.pan_x, session.camera.pan_y
            
            for wd in session.placing_geo_data.get('walls', []):
                if wd['type'] == 'line':
                    ws = wd['start']
                    we = wd['end']
                    s1 = sim_to_screen(sim_mx + ws[0], sim_my + ws[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    s2 = sim_to_screen(sim_mx + we[0], sim_my + we[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    pygame.draw.line(self.screen, (100, 255, 100), s1, s2, 2)
                elif wd['type'] == 'circle':
                    wc = wd['center']
                    wr = wd['radius']
                    sc = sim_to_screen(sim_mx + wc[0], sim_my + wc[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    pr = sim_to_screen(sim_mx + wc[0] + wr, sim_my + wc[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    sr = abs(pr[0] - sc[0])
                    pygame.draw.circle(self.screen, (100, 255, 100), sc, int(sr), 2)

    # =========================================================================
    # Stats & Status
    # =========================================================================

    def _draw_stats(self, session, sim, layout):
        metric_x = layout['LEFT_X'] + 15
        stats_y = config.WINDOW_HEIGHT - 80
        curr_t = calculate_current_temp(sim.vel_x, sim.vel_y, sim.count, config.ATOM_MASS)
        
        self.screen.blit(self.big_font.render(f"Particles: {sim.count}", True, (255, 255, 255)), (metric_x, stats_y))
        self.screen.blit(self.font.render(f"Pairs: {sim.pair_count} | T: {curr_t:.3f}", True, (180, 180, 180)), (metric_x, stats_y + 30))
        self.screen.blit(self.font.render(f"SPS: {int(sim.sps)}", True, (100, 255, 100)), (metric_x, stats_y + 50))

    # =========================================================================
    # Tool Overlay Helpers
    # =========================================================================

    def draw_tool_selection_rect(self, rect):
        pygame.draw.rect(self.screen, (0, 255, 255), rect, 1)

    def draw_tool_brush(self, cx, cy, radius):
        pygame.draw.circle(self.screen, (100, 255, 100), (cx, cy), max(1, int(radius)), 1)

    def draw_tool_line(self, start, end, is_ref=False):
        col = (100, 255, 100) if not is_ref else (200, 200, 200)
        pygame.draw.line(self.screen, col, start, end, 2)

    def draw_tool_rect(self, rect_tuple):
        pygame.draw.rect(self.screen, (100, 255, 100), rect_tuple, 2)

    def draw_tool_circle(self, center, radius, mouse_pos):
        pygame.draw.circle(self.screen, (100, 255, 100), center, int(radius), 2)
        pygame.draw.line(self.screen, (100, 255, 100), center, mouse_pos, 1)

    def draw_tool_point(self, cx, cy):
        pygame.draw.circle(self.screen, (100, 255, 100), (cx, cy), 4)