import pygame
import math
import core.config as config

from core.utils import sim_to_screen, screen_to_sim, get_grouped_points, calculate_current_temp
from model.geometry import Line, Circle, Point

class Renderer:
    def __init__(self, screen, font, big_font):
        self.screen = screen
        self.font = font
        self.big_font = big_font

    def draw_app(self, app, layout, ui_list):
        # Unpack commonly used references for readability
        session = app.session 
        sim = app.sim
        sketch = app.sketch
        
        self.screen.fill(config.BACKGROUND_COLOR)
        
        self._draw_viewport(session, sim, layout)
        
        if session.mode == config.MODE_SIM:
            self._draw_particles(session, sim, layout)
        else:
            self._draw_editor_guides(session, sim, layout)
            
        self._draw_geometry(session, sketch, layout, world_size=sim.world_size)
        
        if session.current_tool:
            # PASSING LAYOUT to the tool so it can calculate screen coords
            session.current_tool.draw_overlay(self.screen, self, layout)
        
        if session.show_constraints:
            self._draw_constraints(session, sketch, layout, world_size=sim.world_size)
            self._draw_editor_overlays(session, sketch, layout, world_size=sim.world_size)
            
        if session.placing_geo_data:
            self._draw_placement_preview(session, sim, layout)

        self._draw_snap_indicator(session, sim, sketch, layout)

        self.screen.set_clip(None)
        self._draw_panels(layout)
        
        if session.mode == config.MODE_SIM:
            self._draw_stats(session, sim, layout)
        
        for el in ui_list:
            el.draw(self.screen, self.font)
            
        self._draw_status(session, layout)

    def _draw_snap_indicator(self, session, sim, sketch, layout):
        if session.constraint_builder.snap_target:
            w_idx, pt_idx = session.constraint_builder.snap_target
            walls = sketch.entities
            if w_idx < len(walls):
                ent = walls[w_idx]
                pt_pos = ent.get_point(pt_idx)
                sx, sy = sim_to_screen(pt_pos[0], pt_pos[1], session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
                pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 6)
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 8, 1)

    def _draw_viewport(self, session, sim, layout):
        sim_rect = pygame.Rect(layout['MID_X'], config.TOP_MENU_H, layout['MID_W'], layout['MID_H'])
        self.screen.set_clip(sim_rect)
        
        tl = sim_to_screen(0, 0, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
        br = sim_to_screen(sim.world_size, sim.world_size, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
        
        g_col = config.GRID_COLOR if session.mode == config.MODE_SIM else (50, 60, 50)
        pygame.draw.rect(self.screen, g_col, (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]), 2)

    def _draw_particles(self, session, sim, layout):
        for i in range(sim.count):
            if sim.is_static[i] and not getattr(session, 'show_wall_atoms', True): continue 
            
            sx, sy = sim_to_screen(sim.pos_x[i], sim.pos_y[i], session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
            if layout['MID_X'] < sx < layout['RIGHT_X'] and config.TOP_MENU_H < sy < config.WINDOW_HEIGHT:
                is_stat = sim.is_static[i]
                col = config.COLOR_STATIC if is_stat else config.COLOR_DYNAMIC
                atom_sig = sim.atom_sigma[i]
                rad = max(2, int(atom_sig * config.PARTICLE_RADIUS_SCALE * ((layout['MID_W']-50)/sim.world_size) * session.camera.zoom))
                pygame.draw.circle(self.screen, col, (sx, sy), rad)

    def _draw_editor_guides(self, session, sim, layout):
        cx, cy = sim_to_screen(sim.world_size/2, sim.world_size/2, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
        pygame.draw.line(self.screen, (50, 50, 50), (cx-10, cy), (cx+10, cy))
        pygame.draw.line(self.screen, (50, 50, 50), (cx, cy-10), (cx, cy+10))

    def _draw_geometry(self, session, sketch, layout, world_size=50.0):
        transform = lambda x, y: sim_to_screen(x, y, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, world_size, layout)
        
        walls = sketch.entities
        for i, w in enumerate(walls):
            mat = sketch.materials.get(w.material_id)
            base_color = mat.color if mat else (200, 200, 200)
            
            is_sel = (i in session.selection.walls)
            is_pend = (session.constraint_builder.pending_type and i in session.constraint_builder.target_walls)
            
            if isinstance(w, Line):
                self._draw_line_entity(w, transform, is_sel, is_pend, base_color)
            elif isinstance(w, Circle):
                self._draw_circle_entity(w, transform, is_sel, is_pend, base_color)
            elif isinstance(w, Point):
                self._draw_point_entity(w, transform, is_sel, is_pend, base_color)

    def _draw_line_entity(self, line, transform, is_selected, is_pending, color):
        s1 = transform(line.start[0], line.start[1])
        s2 = transform(line.end[0], line.end[1])
        
        draw_col = color if color else (255, 255, 255)
        width = 1
        
        if is_selected: 
            draw_col = (255, 200, 50)
            width = 3
        elif is_pending:
            draw_col = (100, 255, 100)
            width = 3
        
        if line.ref: 
            self._draw_dashed_line(draw_col, s1, s2, width)
        else: 
            pygame.draw.line(self.screen, draw_col, s1, s2, width)

        if line.anchored[0]: pygame.draw.circle(self.screen, (255, 50, 50), s1, 3)
        if line.anchored[1]: pygame.draw.circle(self.screen, (255, 50, 50), s2, 3)

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
        
        if circle.anchored[0]: pygame.draw.circle(self.screen, (255, 50, 50), (int(sx), int(sy)), 4)
        pygame.draw.circle(self.screen, draw_col, (int(sx), int(sy)), int(s_radius), width)

    def _draw_point_entity(self, pt, transform, is_selected, is_pending, color):
        sx, sy = transform(pt.pos[0], pt.pos[1])
        draw_col = color if color else (255, 255, 255)
        if is_selected: draw_col = (0, 255, 255)
        elif is_pending: draw_col = (100, 255, 100)
        
        if pt.anchored:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(sx), int(sy)), 6)
        
        pygame.draw.circle(self.screen, draw_col, (int(sx), int(sy)), 4)

    def _draw_dashed_line(self, color, start_pos, end_pos, width=1, dash_length=10):
        x1, y1 = start_pos; x2, y2 = end_pos
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0: return
        dash_amount = int(length / dash_length)
        if dash_amount == 0: return
        dx = (x2 - x1) / length; dy = (y2 - y1) / length
        for i in range(0, dash_amount, 2):
            s = (x1 + dx * i * dash_length, y1 + dy * i * dash_length)
            e = (x1 + dx * (i + 1) * dash_length, y1 + dy * (i + 1) * dash_length)
            pygame.draw.line(self.screen, color, s, e, width)

    def _draw_constraints(self, session, sketch, layout, world_size=50.0):
        # 1. Calculate Layout internally (SoC Fix)
        layout_data = self._calculate_constraint_layout(sketch.constraints, sketch.entities, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, world_size, layout)
        
        # 2. Helper transform for drawing connectors
        transform = lambda x, y: sim_to_screen(x, y, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, world_size, layout)

        for item in layout_data:
            c = item['constraint']
            rect = item['rect']
            symbol = item['symbol']
            
            # Draw Connectors first
            self._draw_constraint_connectors(c, transform, sketch.entities)
            
            # Draw Icon Background
            s_rect = rect.copy(); s_rect.x += 2; s_rect.y += 2
            pygame.draw.rect(self.screen, (0,0,0), s_rect, border_radius=4)
            pygame.draw.rect(self.screen, item['bg_color'], rect, border_radius=4)
            pygame.draw.rect(self.screen, (100, 100, 100), rect, 1, border_radius=4)
            
            # Draw Text
            text = self.font.render(symbol, True, item['text_color'])
            tx = rect.centerx - text.get_width() // 2
            ty = rect.centery - text.get_height() // 2
            self.screen.blit(text, (tx, ty))

    def _draw_constraint_connectors(self, c, transform, entities):
        ctype = c.type
        indices = c.indices
        
        if ctype == 'COINCIDENT':
            idx1, idx2 = indices[0][1], indices[1][1]
            if idx1 == -1 or idx2 == -1: # Pt-Entity
                ent_idx = indices[0][0] if idx1 == -1 else indices[1][0]
                pt_ref = indices[1] if idx1 == -1 else indices[0]
                c_pos = self._get_entity_center_screen(ent_idx, entities, transform)
                p_pos = self._get_point_screen(pt_ref[0], pt_ref[1], entities, transform)
                self._draw_connector(c_pos, p_pos)
        
        elif ctype in ['COLLINEAR', 'MIDPOINT']:
            pt_ref = indices[0]
            line_idx = indices[1]
            p_pos = self._get_point_screen(pt_ref[0], pt_ref[1], entities, transform)
            l_pos = self._get_entity_center_screen(line_idx, entities, transform)
            col = (100, 150, 100) if ctype == 'COLLINEAR' else (100, 100, 150)
            self._draw_connector(p_pos, l_pos, col)
            
        elif ctype in ['EQUAL', 'ANGLE', 'PARALLEL', 'PERPENDICULAR']:
             c1 = self._get_entity_center_screen(indices[0], entities, transform)
             c2 = self._get_entity_center_screen(indices[1], entities, transform)
             self._draw_connector(c1, c2)

    def _get_entity_center_screen(self, entity_idx, entities, transform):
        if entity_idx < 0 or entity_idx >= len(entities): return (0,0)
        e = entities[entity_idx]
        if hasattr(e, 'start'): # Line
            p1 = transform(e.start[0], e.start[1])
            p2 = transform(e.end[0], e.end[1])
            return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        elif hasattr(e, 'center'): # Circle
            return transform(e.center[0], e.center[1])
        elif hasattr(e, 'pos'): # Point
             return transform(e.pos[0], e.pos[1])
        return (0,0)

    def _get_point_screen(self, ent_idx, pt_idx, entities, transform):
        if ent_idx < 0 or ent_idx >= len(entities): return (0,0)
        e = entities[ent_idx]
        pt = e.get_point(pt_idx)
        return transform(pt[0], pt[1])

    def _draw_connector(self, p1, p2, color=(100, 100, 100)):
        pygame.draw.line(self.screen, color, p1, p2, 1)

    def _draw_editor_overlays(self, session, sketch, layout, world_size=50.0):
        point_map = get_grouped_points(sketch.entities, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, world_size, layout)
        anchored_points_draw_list = []
        base_r, step_r = 5, 4
        
        for center_pos, items in point_map.items():
            cx, cy = center_pos
            count = len(items)
            for k in range(count - 1, -1, -1):
                w_idx, pt_idx = items[k]
                radius = base_r + (k * step_r)
                
                color = (200, 200, 200) 
                if (w_idx, pt_idx) in session.selection.points: color = (0, 255, 255)
                elif session.constraint_builder.pending_type and (w_idx, pt_idx) in session.constraint_builder.target_points: color = (100, 255, 100)
                
                pygame.draw.circle(self.screen, (30,30,30), (cx, cy), radius)
                pygame.draw.circle(self.screen, color, (cx, cy), radius, 2)
                
                w = sketch.entities[w_idx]
                is_anchored = False
                if isinstance(w, Line): is_anchored = w.anchored[pt_idx]
                elif isinstance(w, Circle): is_anchored = w.anchored[0]
                elif isinstance(w, Point): is_anchored = w.anchored
                if is_anchored: anchored_points_draw_list.append((cx, cy))
        
        for pt in anchored_points_draw_list: 
            pygame.draw.circle(self.screen, (255, 50, 50), pt, 3)

    def _draw_placement_preview(self, session, sim, layout):
        mx, my = pygame.mouse.get_pos()
        if layout['LEFT_X'] < mx < layout['RIGHT_X']:
            sim_mx, sim_my = screen_to_sim(mx, my, session.camera.zoom, session.camera.pan_x, session.camera.pan_y, sim.world_size, layout)
            zoom = session.camera.zoom
            pan_x, pan_y = session.camera.pan_x, session.camera.pan_y
            
            for wd in session.placing_geo_data.get('walls', []):
                if wd['type'] == 'line':
                    ws = wd['start']; we = wd['end']
                    s1 = sim_to_screen(sim_mx + ws[0], sim_my + ws[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    s2 = sim_to_screen(sim_mx + we[0], sim_my + we[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    pygame.draw.line(self.screen, (100, 255, 100), s1, s2, 2)
                elif wd['type'] == 'circle':
                        wc = wd['center']; wr = wd['radius']
                        sc = sim_to_screen(sim_mx + wc[0], sim_my + wc[1], zoom, pan_x, pan_y, sim.world_size, layout)
                        pr = sim_to_screen(sim_mx + wc[0] + wr, sim_my + wc[1], zoom, pan_x, pan_y, sim.world_size, layout)
                        sr = abs(pr[0] - sc[0])
                        pygame.draw.circle(self.screen, (100, 255, 100), sc, int(sr), 2)

    def _draw_panels(self, layout):
        pygame.draw.rect(self.screen, config.PANEL_BG_COLOR, (0, config.TOP_MENU_H, layout['LEFT_W'], layout['H']))
        pygame.draw.line(self.screen, config.PANEL_BORDER_COLOR, (layout['LEFT_W'], config.TOP_MENU_H), (layout['LEFT_W'], layout['H']))
        pygame.draw.rect(self.screen, config.PANEL_BG_COLOR, (layout['RIGHT_X'], config.TOP_MENU_H, layout['RIGHT_W'], layout['H']))
        pygame.draw.line(self.screen, config.PANEL_BORDER_COLOR, (layout['RIGHT_X'], config.TOP_MENU_H), (layout['RIGHT_X'], layout['H']))

    def _draw_stats(self, session, sim, layout):
        metric_x = layout['LEFT_X'] + 15
        stats_y = config.WINDOW_HEIGHT - 80
        curr_t = calculate_current_temp(sim.vel_x, sim.vel_y, sim.count, config.ATOM_MASS)
        self.screen.blit(self.big_font.render(f"Particles: {sim.count}", True, (255, 255, 255)), (metric_x, stats_y))
        self.screen.blit(self.font.render(f"Pairs: {sim.pair_count} | T: {curr_t:.3f}", True, (180, 180, 180)), (metric_x, stats_y + 30))
        self.screen.blit(self.font.render(f"SPS: {int(sim.sps)}", True, (100, 255, 100)), (metric_x, stats_y + 50))

    def _draw_status(self, session, layout):
        if session.status.is_visible:
            status_surf = self.font.render(session.status.message, True, (100, 255, 100))
            pygame.draw.rect(self.screen, (30,30,30), (layout['MID_X'] + 15, config.TOP_MENU_H + 10, status_surf.get_width()+10, 25), border_radius=5)
            self.screen.blit(status_surf, (layout['MID_X'] + 20, config.TOP_MENU_H + 15))

    # --- Tool Overlay Helpers ---
    def draw_tool_selection_rect(self, rect):
        pygame.draw.rect(self.screen, (0, 255, 255), rect, 1)

    def draw_tool_brush(self, cx, cy, radius):
        pygame.draw.circle(self.screen, (100, 255, 100), (cx, cy), max(1, int(radius)), 1)

    def draw_tool_line(self, start, end, is_ref=False):
        col = (100, 255, 100) if not is_ref else (200, 200, 200)
        pygame.draw.line(self.screen, col, start, end, 2)
        if is_ref:
            pass

    def draw_tool_rect(self, rect_tuple):
         pygame.draw.rect(self.screen, (100, 255, 100), rect_tuple, 2)

    def draw_tool_circle(self, center, radius, mouse_pos):
        pygame.draw.circle(self.screen, (100, 255, 100), center, int(radius), 2)
        pygame.draw.line(self.screen, (100, 255, 100), center, mouse_pos, 1)

    def draw_tool_point(self, cx, cy):
        pygame.draw.circle(self.screen, (100, 255, 100), (cx, cy), 4)

    # --- New Internal Layout Logic (Moved from simulation_geometry.py) ---

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
                    found_group = key; break
            if found_group:
                grouped[found_group].append((i, c)); layout_data.append((i, c, found_group))
            else:
                grouped[(cx, cy)] = [(i, c)]; layout_data.append((i, c, (cx, cy)))

        results = []
        group_counts = {k: 0 for k in grouped}
        
        for i, c, key in layout_data:
            idx = group_counts[key]; group_counts[key] += 1
            total_in_group = len(grouped[key]); spacing = 30
            
            start_x = -((total_in_group - 1) * spacing) / 2.0
            final_offset_x = start_x + idx * spacing
            
            cx, cy = self._get_constraint_raw_pos(c, entities, transform)
            final_x, final_y = cx + final_offset_x, cy
            
            symbol = self._get_constraint_symbol(c)
            
            w = len(symbol) * 9 + 12 
            h = 20
            rect = pygame.Rect(final_x - w//2, final_y - h//2, w, h)
            
            # Colors
            bg_color = (50, 50, 50)
            text_color = (255, 255, 255)
            if c.type == 'COINCIDENT': text_color = (100, 255, 255)
            elif c.type == 'COLLINEAR': text_color = (150, 255, 150)
            elif c.type == 'MIDPOINT': text_color = (150, 150, 255)
            elif c.type == 'LENGTH': text_color = (255, 200, 100)
            elif c.type == 'ANGLE' and hasattr(c, 'value'): text_color = (255, 200, 200)

            results.append({
                'index': i, 'constraint': c, 'center': (final_x, final_y),
                'rect': rect, 'symbol': symbol,
                'text_color': text_color, 'bg_color': bg_color
            })
        return results

    def _get_constraint_symbol(self, c):
        if c.type == 'COINCIDENT': return "C"
        elif c.type == 'COLLINEAR': return "CL"
        elif c.type == 'MIDPOINT': return "M"
        elif c.type == 'LENGTH': return f"{c.value:.1f}"
        elif c.type == 'EQUAL': return "="
        elif c.type == 'ANGLE':
            if hasattr(c, 'value'): return f"{c.value:.0f}°"
            return "//" if c.type == 'PARALLEL' else "T"
        elif c.type == 'PARALLEL': return "//"
        elif c.type == 'PERPENDICULAR': return "T"
        elif c.type == 'HORIZONTAL': return "H"
        elif c.type == 'VERTICAL': return "V"
        return "?"

    def _get_constraint_raw_pos(self, c, entities, transform):
        ctype = c.type
        indices = c.indices
        
        if ctype == 'COINCIDENT':
            idx1, idx2 = indices[0][1], indices[1][1]
            if idx1 == -1 or idx2 == -1: # Pt-Entity
                ent_idx = indices[0][0] if idx1 == -1 else indices[1][0]
                pt_ref = indices[1] if idx1 == -1 else indices[0]
                c_pos = self._get_entity_center_screen(ent_idx, entities, transform)
                p_pos = self._get_point_screen(pt_ref[0], pt_ref[1], entities, transform)
                return ((c_pos[0] + p_pos[0]) // 2, (c_pos[1] + p_pos[1]) // 2)
            else:
                p_pos = self._get_point_screen(indices[0][0], idx1, entities, transform)
                return (p_pos[0] + 15, p_pos[1] - 15)
                
        elif ctype in ['COLLINEAR', 'MIDPOINT']:
            pt_ref = indices[0]
            line_idx = indices[1]
            p_pos = self._get_point_screen(pt_ref[0], pt_ref[1], entities, transform)
            l_pos = self._get_entity_center_screen(line_idx, entities, transform)
            return ((p_pos[0] + l_pos[0]) // 2, (p_pos[1] + l_pos[1]) // 2)
            
        elif ctype in ['LENGTH', 'HORIZONTAL', 'VERTICAL']:
            c_pos = self._get_entity_center_screen(indices[0], entities, transform)
            if ctype == 'LENGTH': return (c_pos[0], c_pos[1] + 15)
            return (c_pos[0], c_pos[1] - 15)
            
        elif ctype in ['EQUAL', 'ANGLE', 'PARALLEL', 'PERPENDICULAR']:
                c1 = self._get_entity_center_screen(indices[0], entities, transform)
                c2 = self._get_entity_center_screen(indices[1], entities, transform)
                return ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
        return (0,0)
