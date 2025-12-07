import pygame
import numpy as np
import config
import math
import time 
from tkinter import filedialog, Tk, simpledialog

# Modules
from simulation_state import Simulation
from ui_widgets import SmartSlider, Button, InputField, ContextMenu, PropertiesDialog, MenuBar, RotationDialog
from geometry import Line, Point, Circle
from constraints import create_constraint, Coincident, Length, EqualLength, Angle, Midpoint
import utils
import file_io
from renderer import Renderer

class AppState:
    """Holds the application state (UI, Selection, Camera, Modes)"""
    def __init__(self):
        self.mode = config.MODE_SIM
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.current_tool = config.TOOL_BRUSH
        
        # Interaction State
        self.is_panning = False
        self.last_mouse_pos = (0, 0)
        self.is_painting = False
        self.is_erasing = False
        
        # Editor State
        self.wall_mode = None
        self.wall_idx = -1
        self.wall_pt = -1
        self.rect_start = None
        self.selected_walls = set()
        self.selected_points = set()
        self.current_group_indices = []
        
        # Constraints
        self.pending_constraint = None
        self.pending_targets_walls = []
        self.pending_targets_points = []
        self.temp_constraint_active = False
        
        # Snapping Helpers
        self.current_snap_target = None
        self.new_wall_start_snap = None
        
        # IO / Undo
        self.placing_geo_data = None
        self.sim_backup_state = None
        self.current_filepath = None
        
        # Status
        self.status_msg = ""
        self.status_time = 0
        
        # UI references (Set in main)
        self.input_world = None

    def set_status(self, msg):
        self.status_msg = msg
        self.status_time = time.time()

def main():
    try: root_tk = Tk(); root_tk.withdraw()
    except: root_tk = None

    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fast MD - Geometry Editor")
    
    font = pygame.font.SysFont("segoeui", 15)
    big_font = pygame.font.SysFont("segoeui", 22)
    renderer = Renderer(screen, font, big_font)
    clock = pygame.time.Clock()
    
    sim = Simulation()
    app = AppState()
    
    # Layout Dictionary
    layout = {
        'W': config.WINDOW_WIDTH, 'H': config.WINDOW_HEIGHT,
        'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
        'RIGHT_W': config.PANEL_RIGHT_WIDTH, 'RIGHT_X': config.WINDOW_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_X': config.PANEL_LEFT_WIDTH, 'MID_W': config.WINDOW_WIDTH - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_H': config.WINDOW_HEIGHT - config.TOP_MENU_H
    }

    # --- UI Setup ---
    menu_bar = MenuBar(layout['W'], config.TOP_MENU_H)
    menu_bar.items["File"] = ["New Simulation", "Open...", "Save", "Save As...", "---", "Create New Geometry", "Add Existing Geometry"] 
    
    # Left Panel
    lp_curr_y = config.TOP_MENU_H + 20; lp_margin = 10
    btn_play = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Play/Pause", active=False, color_active=(60, 120, 60), color_inactive=(180, 60, 60)); lp_curr_y += 50
    btn_clear = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Clear", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_curr_y += 50
    btn_reset = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Reset", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_curr_y += 50
    btn_undo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Undo", active=False, toggle=False); lp_curr_y += 45
    btn_redo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Redo", active=False, toggle=False)
    
    # Right Panel
    rp_start_x = layout['RIGHT_X'] + 15; rp_width = layout['RIGHT_W'] - 30; rp_curr_y = config.TOP_MENU_H + 120
    slider_gravity = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.0, 50.0, config.DEFAULT_GRAVITY, "Gravity", hard_min=0.0); rp_curr_y+=60
    slider_temp = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.0, 5.0, 0.5, "Temperature", hard_min=0.0); rp_curr_y+=60
    slider_damping = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.90, 1.0, config.DEFAULT_DAMPING, "Damping", hard_min=0.0, hard_max=1.0); rp_curr_y+=60
    slider_dt = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.0001, 0.01, config.DEFAULT_DT, "Time Step (dt)", hard_min=0.00001); rp_curr_y+=60
    slider_sigma = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.5, 2.0, config.ATOM_SIGMA, "Sigma (Size)", hard_min=0.1); rp_curr_y+=60
    slider_epsilon = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.1, 5.0, config.ATOM_EPSILON, "Epsilon (Strength)", hard_min=0.0); rp_curr_y+=60
    slider_M = SmartSlider(rp_start_x, rp_curr_y, rp_width, 1.0, 100.0, float(config.DEFAULT_DRAW_M), "Speed (Steps/Frame)", hard_min=1.0); rp_curr_y+=60
    slider_skin = SmartSlider(rp_start_x, rp_curr_y, rp_width, 0.1, 2.0, config.DEFAULT_SKIN_DISTANCE, "Skin Distance", hard_min=0.05); rp_curr_y+=60
    
    btn_w = (rp_width - 10) // 2
    btn_thermostat = Button(rp_start_x, rp_curr_y, btn_w, 30, "Thermostat", active=False)
    btn_boundaries = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Bounds", active=False); rp_curr_y += 40
    
    # Tools
    btn_tool_brush = Button(rp_start_x, rp_curr_y, btn_w, 30, "Brush", active=True, toggle=False)
    btn_tool_select = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Select", active=False, toggle=False); rp_curr_y += 40
    btn_tool_line = Button(rp_start_x, rp_curr_y, btn_w, 30, "Line", active=False, toggle=False)
    btn_tool_rect = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Rectangle", active=False, toggle=False); rp_curr_y += 40
    btn_tool_circle = Button(rp_start_x, rp_curr_y, btn_w, 30, "Circle", active=False, toggle=False)
    btn_tool_point = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Point", active=False, toggle=False); rp_curr_y += 40
    btn_tool_ref = Button(rp_start_x, rp_curr_y, btn_w, 30, "Ref Line", active=False, toggle=False); rp_curr_y += 40

    slider_brush_size = SmartSlider(rp_start_x, rp_curr_y, rp_width, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5); rp_curr_y+=60
    app.input_world = InputField(rp_start_x + 80, rp_curr_y, 60, 25, str(config.DEFAULT_WORLD_SIZE))
    btn_resize = Button(rp_start_x + 150, rp_curr_y, rp_width - 150, 25, "Resize & Restart", active=False, toggle=False)
    
    # Editor Buttons
    ae_curr_y = config.TOP_MENU_H + 40
    btn_ae_save = Button(rp_start_x, ae_curr_y, rp_width, 40, "Save Geometry", active=False, toggle=False, color_inactive=(50, 120, 50)); ae_curr_y+=50
    btn_ae_discard = Button(rp_start_x, ae_curr_y, rp_width, 40, "Discard & Exit", active=False, toggle=False, color_inactive=(150, 50, 50)); ae_curr_y+=50
    
    c_btn_w = (rp_width - 10) // 2
    btn_const_coincident = Button(rp_start_x, ae_curr_y, rp_width, 30, "Coincident (Pt-Pt)", toggle=False); ae_curr_y+=35
    btn_const_midpoint = Button(rp_start_x, ae_curr_y, rp_width, 30, "Midpoint (Pt-Ln)", toggle=False); ae_curr_y+=35
    btn_const_length = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Fix Length", toggle=False)
    btn_const_equal = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Equal Len", toggle=False); ae_curr_y+=35
    btn_const_parallel = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Parallel", toggle=False)
    btn_const_perp = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Perpendic", toggle=False); ae_curr_y+=35
    btn_const_horiz = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Horizontal", toggle=False)
    btn_const_vert = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Vertical", toggle=False); ae_curr_y+=35

    # Group UI Elements - Updated to include tool buttons
    ui_sim_elements = [
        btn_play, btn_clear, btn_reset, btn_undo, btn_redo, 
        slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, 
        btn_thermostat, btn_boundaries, 
        btn_tool_brush, btn_tool_line, 
        slider_brush_size, btn_resize
    ]
    
    ui_editor_elements = [
        btn_ae_save, btn_ae_discard, btn_undo, btn_redo, btn_clear, 
        btn_tool_select, btn_tool_line, btn_tool_rect, btn_tool_circle, btn_tool_point, btn_tool_ref,
        btn_const_coincident, btn_const_midpoint, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert
    ]
    
    # Right panel specifically for layout updates
    right_panel_elements = [
        slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, 
        btn_thermostat, btn_boundaries, 
        btn_tool_brush, btn_tool_select, btn_tool_line, btn_tool_rect, btn_tool_circle, btn_tool_point, btn_tool_ref, 
        slider_brush_size, app.input_world, btn_resize, 
        btn_ae_save, btn_ae_discard, 
        btn_const_coincident, btn_const_midpoint, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert
    ]

    # --- Helper Functions (Logic) ---
    
    def change_tool(tool_id):
        """Updates app state and button visuals"""
        app.current_tool = tool_id
        btn_tool_brush.active = (tool_id == config.TOOL_BRUSH)
        btn_tool_select.active = (tool_id == config.TOOL_SELECT)
        btn_tool_line.active = (tool_id == config.TOOL_LINE)
        btn_tool_rect.active = (tool_id == config.TOOL_RECT)
        btn_tool_circle.active = (tool_id == config.TOOL_CIRCLE)
        btn_tool_point.active = (tool_id == config.TOOL_POINT)
        btn_tool_ref.active = (tool_id == config.TOOL_REF)

    def enter_geometry_mode():
        sim.snapshot()
        app.sim_backup_state = sim.undo_stack.pop()
        sim.clear_particles(snapshot=False)
        sim.world_size = 30.0; sim.walls = []; sim.constraints = []
        app.mode = config.MODE_EDITOR
        change_tool(config.TOOL_SELECT)
        app.zoom = 1.5; app.pan_x = 0; app.pan_y = 0
        app.set_status("Entered Geometry Editor")

    def exit_editor_mode(restore_state):
        sim.clear_particles(snapshot=False)
        if restore_state: sim.restore_state(restore_state)
        else: sim.world_size = config.DEFAULT_WORLD_SIZE
        app.mode = config.MODE_SIM
        app.zoom = 1.0; app.pan_x = 0; app.pan_y = 0
        app.set_status("Returned to Simulation")

    def cancel_operation():
        if app.wall_mode in ['NEW', 'RECT', 'CIRCLE']:
            if app.wall_idx < len(sim.walls):
                # Rect cleanup
                if app.wall_mode == 'RECT':
                    added_count = len(sim.walls) - app.wall_idx
                    for _ in range(added_count):
                        if sim.walls: sim.walls.pop()
                else:
                    if sim.walls: sim.walls.pop()
            app.wall_mode = None
            app.wall_idx = -1
            app.set_status("Operation Cancelled")
            return True
        elif app.wall_mode in ['MOVE_WALL', 'MOVE_GROUP', 'EDIT']:
             app.wall_mode = None; app.wall_idx = -1
             app.set_status("Move Cancelled")
             return True
        return False

    def reset_constraint_buttons():
        for btn in [btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_coincident, btn_const_midpoint, btn_const_horiz, btn_const_vert]:
            btn.active = False

    def trigger_constraint(ctype):
        applied = False
        # Logic for applying constraints based on selection
        if ctype == 'LENGTH' and len(app.selected_walls) == 1:
            w_idx = list(app.selected_walls)[0]; w = sim.walls[w_idx]
            if isinstance(w, Line):
                sim.add_constraint_object(Length(w_idx, np.linalg.norm(w.end - w.start))); applied = True
        elif ctype == 'EQUAL' and len(app.selected_walls) == 2:
            sl = list(app.selected_walls)
            if all(isinstance(sim.walls[i], Line) for i in sl):
                sim.add_constraint_object(EqualLength(sl[0], sl[1])); applied = True
        elif ctype in ['PARALLEL', 'PERPENDICULAR'] and len(app.selected_walls) == 2:
            sl = list(app.selected_walls)
            if all(isinstance(sim.walls[i], Line) for i in sl):
                sim.add_constraint_object(Angle(ctype, sl[0], sl[1])); applied = True
        elif ctype in ['HORIZONTAL', 'VERTICAL'] and len(app.selected_walls) >= 1:
            for w_idx in app.selected_walls:
                if isinstance(sim.walls[w_idx], Line): sim.add_constraint_object(Angle(ctype, w_idx))
            applied = True
        elif ctype == 'COINCIDENT' and len(app.selected_points) == 2:
            sp = list(app.selected_points)
            sim.add_constraint_object(Coincident(sp[0][0], sp[0][1], sp[1][0], sp[1][1])); applied = True
        elif ctype == 'MIDPOINT' and len(app.selected_points) == 1 and len(app.selected_walls) == 1:
            pt_tuple = list(app.selected_points)[0]
            w_idx = list(app.selected_walls)[0]
            if isinstance(sim.walls[w_idx], Line):
                sim.add_constraint_object(Midpoint(pt_tuple[0], pt_tuple[1], w_idx)); applied = True

        if applied:
            app.set_status(f"Applied {ctype}")
            app.selected_walls.clear(); app.selected_points.clear()
            app.pending_constraint = None; reset_constraint_buttons()
        else:
            app.pending_constraint = ctype
            app.pending_targets_walls.clear(); app.pending_targets_points.clear()
            if app.selected_walls: app.pending_targets_walls.extend(list(app.selected_walls))
            if app.selected_points: app.pending_targets_points.extend(list(app.selected_points))
            app.selected_walls.clear(); app.selected_points.clear()
            app.set_status(f"Select targets for {ctype}...")
            reset_constraint_buttons()
            # Activate relevant button
            if ctype == 'LENGTH': btn_const_length.active = True
            elif ctype == 'EQUAL': btn_const_equal.active = True
            elif ctype == 'PARALLEL': btn_const_parallel.active = True
            elif ctype == 'PERPENDICULAR': btn_const_perp.active = True
            elif ctype == 'COINCIDENT': btn_const_coincident.active = True
            elif ctype == 'MIDPOINT': btn_const_midpoint.active = True
            elif ctype == 'HORIZONTAL': btn_const_horiz.active = True
            elif ctype == 'VERTICAL': btn_const_vert.active = True

    def handle_pending_constraint_click(wall_idx=None, pt_idx=None):
        if app.pending_constraint == 'MIDPOINT':
            if pt_idx is not None and len(app.pending_targets_points) == 0:
                app.pending_targets_points.append(pt_idx); app.set_status("Got Point. Select Line.")
            elif wall_idx is not None and len(app.pending_targets_walls) == 0:
                if isinstance(sim.walls[wall_idx], Line):
                    app.pending_targets_walls.append(wall_idx); app.set_status("Got Line. Select Point.")
            
            if len(app.pending_targets_points) == 1 and len(app.pending_targets_walls) == 1:
                sim.add_constraint_object(Midpoint(app.pending_targets_points[0][0], app.pending_targets_points[0][1], app.pending_targets_walls[0]))
                app.set_status("Applied Midpoint")
                app.pending_constraint = None; reset_constraint_buttons()
        
        elif app.pending_constraint in ['LENGTH', 'HORIZONTAL', 'VERTICAL']:
            if wall_idx is not None and isinstance(sim.walls[wall_idx], Line):
                if app.pending_constraint == 'LENGTH':
                    w = sim.walls[wall_idx]
                    sim.add_constraint_object(Length(wall_idx, np.linalg.norm(w.end - w.start)))
                else: sim.add_constraint_object(Angle(app.pending_constraint, wall_idx))
                app.set_status(f"Applied {app.pending_constraint}"); app.pending_constraint = None; reset_constraint_buttons()
        
        elif app.pending_constraint in ['EQUAL', 'PARALLEL', 'PERPENDICULAR']:
            if wall_idx is not None and isinstance(sim.walls[wall_idx], Line):
                if wall_idx not in app.pending_targets_walls:
                    app.pending_targets_walls.append(wall_idx)
                    app.set_status(f"Selected 1/2 for {app.pending_constraint}")
                    if len(app.pending_targets_walls) == 2:
                        if app.pending_constraint == 'EQUAL': sim.add_constraint_object(EqualLength(app.pending_targets_walls[0], app.pending_targets_walls[1]))
                        else: sim.add_constraint_object(Angle(app.pending_constraint, app.pending_targets_walls[0], app.pending_targets_walls[1]))
                        app.set_status(f"Applied {app.pending_constraint}"); app.pending_constraint = None; reset_constraint_buttons()
        
        elif app.pending_constraint == 'COINCIDENT':
            if pt_idx is not None:
                if pt_idx not in app.pending_targets_points:
                    app.pending_targets_points.append(pt_idx)
                    app.set_status(f"Selected 1/2 Points")
                    if len(app.pending_targets_points) == 2:
                        sim.add_constraint_object(Coincident(app.pending_targets_points[0][0], app.pending_targets_points[0][1], app.pending_targets_points[1][0], app.pending_targets_points[1][1]))
                        app.set_status("Applied Coincident"); app.pending_constraint = None; reset_constraint_buttons()

    # --- Mouse Handlers ---
    def on_mouse_down(event):
        mx, my = event.pos
        if layout['LEFT_X'] < mx < layout['RIGHT_X']:
            if app.placing_geo_data:
                sx, sy = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                sim.place_geometry(app.placing_geo_data, sx, sy)
                return

            if app.current_tool == config.TOOL_BRUSH and app.mode == config.MODE_SIM:
                app.is_painting = True; sim.snapshot()
                return

            # Geometry Creation
            is_drawing_tool = app.current_tool in [config.TOOL_POINT, config.TOOL_RECT, config.TOOL_CIRCLE, config.TOOL_LINE, config.TOOL_REF]
            if (app.mode == config.MODE_EDITOR and is_drawing_tool) or (app.mode == config.MODE_SIM and app.current_tool == config.TOOL_LINE):
                sx, sy, snap = utils.get_snapped_pos(mx, my, sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                sim.snapshot()
                
                if app.current_tool == config.TOOL_POINT and app.mode == config.MODE_EDITOR:
                    sim.add_wall((sx, sy), (sx, sy), is_ref=False)
                    app.wall_idx = len(sim.walls)-1
                    if snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        sim.add_constraint_object(Coincident(app.wall_idx, 0, snap[0], snap[1]))
                        
                elif app.current_tool == config.TOOL_RECT and app.mode == config.MODE_EDITOR:
                    app.rect_start = (sx, sy)
                    base_idx = len(sim.walls)
                    for _ in range(4): sim.add_wall(app.rect_start, app.rect_start)
                    app.wall_mode = 'RECT'; app.wall_idx = base_idx
                    
                elif app.current_tool == config.TOOL_CIRCLE and app.mode == config.MODE_EDITOR:
                    sim.add_circle((sx, sy), 0.1)
                    app.wall_mode = 'CIRCLE'; app.wall_idx = len(sim.walls)-1
                    
                else: # Line or Ref
                    is_ref = (app.current_tool == config.TOOL_REF)
                    sim.add_wall((sx, sy), (sx, sy), is_ref=is_ref)
                    app.wall_mode = 'NEW'; app.wall_idx = len(sim.walls)-1; app.wall_pt = 1
                    app.current_snap_target = None; app.new_wall_start_snap = None
                    if pygame.key.get_mods() & pygame.KMOD_CTRL: app.new_wall_start_snap = snap

            # Selection / Dragging
            elif app.current_tool == config.TOOL_SELECT:
                # Reuse logic from utils/main original
                sx, sy = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                point_map = utils.get_grouped_points(sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                hit_pt = None
                
                # Point Hit
                base_r, step_r = 5, 4
                found_stack = None; found_center = None
                for center_pos, items in point_map.items():
                    dist = math.hypot(mx - center_pos[0], my - center_pos[1])
                    max_r = base_r + (len(items) - 1) * step_r
                    if dist <= max_r: found_stack = items; found_center = center_pos; break
                
                if found_stack:
                    dist = math.hypot(mx - found_center[0], my - found_center[1])
                    if dist < base_r: hit_pt = found_stack[0]
                    else:
                        k = int((dist - base_r) / step_r) + 1
                        if k < len(found_stack): hit_pt = found_stack[k]
                        else: hit_pt = found_stack[-1]

                if hit_pt:
                    if app.pending_constraint: handle_pending_constraint_click(pt_idx=hit_pt)
                    else:
                        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                            if hit_pt not in app.selected_points: app.selected_walls.clear(); app.selected_points.clear()
                        if hit_pt in app.selected_points: app.selected_points.remove(hit_pt)
                        else: app.selected_points.add(hit_pt)
                        app.wall_mode = 'EDIT'; app.wall_idx = hit_pt[0]; app.wall_pt = hit_pt[1]
                        sim.snapshot()
                else:
                    # Wall Hit
                    hit_wall = -1
                    rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * app.zoom)
                    for i, w in enumerate(sim.walls):
                        if isinstance(w, Line):
                            if np.array_equal(w.start, w.end): continue
                            p1=w.start; p2=w.end; p3=np.array([sx, sy])
                            d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                            if len_sq == 0: dist = np.linalg.norm(p3-p1)
                            else:
                                t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                                proj = p1 + t*d_vec
                                dist = np.linalg.norm(p3-proj)
                            if dist < rad_sim: hit_wall = i; break
                        elif isinstance(w, Circle):
                            d = math.hypot(sx - w.center[0], sy - w.center[1])
                            if abs(d - w.radius) < rad_sim: hit_wall = i; break
                    
                    if hit_wall != -1:
                        if app.pending_constraint: handle_pending_constraint_click(wall_idx=hit_wall)
                        else:
                            if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                if hit_wall not in app.selected_walls: app.selected_walls.clear(); app.selected_points.clear()
                                app.selected_walls.add(hit_wall)
                            else:
                                if hit_wall in app.selected_walls: app.selected_walls.remove(hit_wall)
                                else: app.selected_walls.add(hit_wall)
                            
                            target_group = utils.get_connected_group(sim, hit_wall)
                            if not utils.is_group_anchored(sim, list(target_group)):
                                app.wall_mode = 'MOVE_GROUP'; app.current_group_indices = list(target_group)
                                sim.snapshot(); app.last_mouse_pos = event.pos
                            else:
                                app.wall_mode = 'MOVE_WALL'; app.wall_idx = hit_wall
                                sim.snapshot(); app.last_mouse_pos = event.pos
                                w = sim.walls[hit_wall]
                                if isinstance(w, Line) and app.mode == config.MODE_EDITOR:
                                    drag_start_length = np.linalg.norm(w.end - w.start)
                                    c = Length(hit_wall, drag_start_length); c.temp = True
                                    sim.constraints.append(c); app.temp_constraint_active = True
                    else:
                        # Click empty
                        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT): app.selected_walls.clear(); app.selected_points.clear()

    # --- Main Loop ---
    running = True
    context_menu = None; prop_dialog = None; rot_dialog = None
    context_wall_idx = -1; context_pt_idx = None

    while running:
        current_ui_list = ui_sim_elements if app.mode == config.MODE_SIM else ui_editor_elements
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # --- Input Handling ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL): sim.undo(); app.set_status("Undo")
                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL): sim.redo(); app.set_status("Redo")
                elif event.key == pygame.K_ESCAPE:
                    if app.placing_geo_data: app.placing_geo_data = None
                    elif app.pending_constraint: app.pending_constraint = None; reset_constraint_buttons()
                    elif cancel_operation(): pass
                    else: app.selected_walls.clear(); app.selected_points.clear()

            # --- Tool Switching ---
            # Use 'if' to allow clicking, but ensure tools are mutually exclusive in UI
            if btn_tool_brush.handle_event(event): change_tool(config.TOOL_BRUSH)
            if btn_tool_select.handle_event(event): change_tool(config.TOOL_SELECT)
            if btn_tool_line.handle_event(event): change_tool(config.TOOL_LINE)
            if btn_tool_rect.handle_event(event): change_tool(config.TOOL_RECT)
            if btn_tool_circle.handle_event(event): change_tool(config.TOOL_CIRCLE)
            if btn_tool_point.handle_event(event): change_tool(config.TOOL_POINT)
            if btn_tool_ref.handle_event(event): change_tool(config.TOOL_REF)

            # --- UI Events ---
            menu_clicked = menu_bar.handle_event(event)
            
            if event.type == pygame.MOUSEBUTTONDOWN and menu_bar.active_menu and not menu_clicked:
                 # Check dropdown click...
                 if menu_bar.dropdown_rect and menu_bar.dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - menu_bar.dropdown_rect.y - 5
                    idx = rel_y // 30
                    opts = menu_bar.items[menu_bar.active_menu]
                    if 0 <= idx < len(opts):
                        selection = opts[idx]
                        menu_bar.active_menu = None 
                        
                        if selection == "New Simulation":
                            sim.reset_simulation()
                            app.input_world.set_value(config.DEFAULT_WORLD_SIZE)
                        
                        elif selection == "Create New Geometry": 
                            if app.mode == config.MODE_SIM: enter_geometry_mode()
                        
                        elif selection == "Add Existing Geometry":
                            if app.mode == config.MODE_SIM and root_tk:
                                f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.json")])
                                if f: 
                                    data = file_io.load_geometry_file(f)
                                    if data: 
                                        app.placing_geo_data = data
                                        app.set_status("Place Geometry")

                        elif root_tk:
                            if selection == "Save As..." or (selection == "Save" and not app.current_filepath):
                                f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
                                if f: 
                                    app.current_filepath = f
                                    msg = file_io.save_file(sim, f)
                                    app.set_status(msg)
                            elif selection == "Save" and app.current_filepath: 
                                msg = file_io.save_file(sim, app.current_filepath)
                                app.set_status(msg)
                            elif selection == "Open...":
                                f = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
                                if f: 
                                    app.current_filepath = f
                                    success, msg, lset = file_io.load_file(sim, f)
                                    app.set_status(msg)
                                    if success: 
                                        app.input_world.set_value(sim.world_size)
                                        app.zoom=1.0; app.pan_x=0; app.pan_y=0
                    else:
                        menu_bar.active_menu = None # Close if clicked outside valid item but inside rect

            # --- Mouse Events ---
            if not menu_clicked and not context_menu and not prop_dialog:
                # Check if mouse on UI
                if event.type == pygame.MOUSEBUTTONDOWN and (event.pos[0] > layout['RIGHT_X'] or event.pos[0] < layout['LEFT_W'] or event.pos[1] < config.TOP_MENU_H):
                    pass # Handled by widget.handle_event loops below
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: on_mouse_down(event)
                    elif event.button == 2: app.is_panning = True; app.last_mouse_pos = event.pos
                    elif event.button == 3:
                        # Right click logic (Cancel or Context Menu)
                        mx, my = event.pos
                        if cancel_operation(): pass
                        else:
                            sim_x, sim_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            
                            # Hit test
                            point_map = utils.get_grouped_points(sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            hit_pt = None
                            base_r, step_r = 5, 4
                            found_stack = None
                            for center_pos, items in point_map.items():
                                dist = math.hypot(mx - center_pos[0], my - center_pos[1])
                                max_r = base_r + (len(items) - 1) * step_r
                                if dist <= max_r: found_stack = items; break
                            if found_stack: hit_pt = found_stack[0]

                            if hit_pt:
                                context_wall_idx = hit_pt[0]; context_pt_idx = hit_pt[1]
                                context_menu = ContextMenu(mx, my, ["Anchor"])
                            else:
                                # Hit Wall
                                hit_wall = -1
                                rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * app.zoom)
                                for i, w in enumerate(sim.walls):
                                    if isinstance(w, Line):
                                        if np.array_equal(w.start, w.end): continue
                                        p1=w.start; p2=w.end; p3=np.array([sim_x, sim_y])
                                        d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                                        if len_sq == 0: dist = np.linalg.norm(p3-p1)
                                        else:
                                            t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                                            proj = p1 + t*d_vec
                                            dist = np.linalg.norm(p3-proj)
                                        if dist < rad_sim: hit_wall = i; break
                                    elif isinstance(w, Circle):
                                        d = math.hypot(sim_x - w.center[0], sim_y - w.center[1])
                                        if abs(d - w.radius) < rad_sim: hit_wall = i; break
                                
                                if hit_wall != -1:
                                    context_wall_idx = hit_wall
                                    opts = ["Properties", "Delete"]
                                    if isinstance(sim.walls[hit_wall], Line): opts.extend(["Set Length...", "Set Rotation..."])
                                    context_menu = ContextMenu(mx, my, opts)
                                else:
                                    # Empty Space Right Click
                                    if app.mode == config.MODE_EDITOR:
                                        change_tool(config.TOOL_SELECT)
                                        app.set_status("Switched to Select Tool")
                                    elif app.mode == config.MODE_SIM:
                                        app.is_erasing = True; sim.snapshot()

                elif event.type == pygame.MOUSEBUTTONUP:
                    # Finalize geometry (apply constraints to new lines)
                    if app.wall_mode == 'NEW':
                        if app.new_wall_start_snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                             sim.add_constraint_object(Coincident(app.wall_idx, 0, app.new_wall_start_snap[0], app.new_wall_start_snap[1]))
                        if app.current_snap_target and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                             sim.add_constraint_object(Coincident(app.wall_idx, 1, app.current_snap_target[0], app.current_snap_target[1]))
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                             w = sim.walls[app.wall_idx]
                             dx = abs(w.start[0] - w.end[0]); dy = abs(w.start[1] - w.end[1])
                             if dy < 0.001: sim.add_constraint_object(Angle('HORIZONTAL', app.wall_idx))
                             elif dx < 0.001: sim.add_constraint_object(Angle('VERTICAL', app.wall_idx))
                    
                    elif app.wall_mode == 'RECT':
                        base = app.wall_idx
                        sim.add_constraint_object(Coincident(base, 1, base+1, 0))
                        sim.add_constraint_object(Coincident(base+1, 1, base+2, 0))
                        sim.add_constraint_object(Coincident(base+2, 1, base+3, 0))
                        sim.add_constraint_object(Coincident(base+3, 1, base, 0))
                        sim.add_constraint_object(Angle('HORIZONTAL', base))
                        sim.add_constraint_object(Angle('VERTICAL', base+1))
                        sim.add_constraint_object(Angle('HORIZONTAL', base+2))
                        sim.add_constraint_object(Angle('VERTICAL', base+3))

                    elif app.wall_mode == 'EDIT':
                        if app.current_snap_target and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                             sim.add_constraint_object(Coincident(app.wall_idx, app.wall_pt, app.current_snap_target[0], app.current_snap_target[1]))

                    app.is_panning = False; app.is_painting = False; app.is_erasing = False; app.wall_mode = None
                    if app.temp_constraint_active:
                        sim.constraints = [c for c in sim.constraints if not c.temp]
                        app.temp_constraint_active = False
                    if app.mode == config.MODE_EDITOR: sim.apply_constraints()

                elif event.type == pygame.MOUSEMOTION:
                    if app.is_panning:
                        app.pan_x += event.pos[0] - app.last_mouse_pos[0]
                        app.pan_y += event.pos[1] - app.last_mouse_pos[1]
                        app.last_mouse_pos = event.pos
                    elif app.is_painting or app.is_erasing:
                        pass # Handled in update loop
                    elif app.wall_mode is not None:
                        mx, my = event.pos
                        if app.wall_mode == 'EDIT' or app.wall_mode == 'NEW':
                            if app.wall_idx < len(sim.walls):
                                w = sim.walls[app.wall_idx]
                                if isinstance(w, Line):
                                    anchor = w.end if app.wall_pt == 0 else w.start
                                    dest_x, dest_y, dest_snap = utils.get_snapped_pos(mx, my, sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout, anchor, app.wall_idx)
                                    app.current_snap_target = dest_snap
                                    if app.wall_pt == 0: sim.update_wall(app.wall_idx, (dest_x, dest_y), w.end)
                                    else: sim.update_wall(app.wall_idx, w.start, (dest_x, dest_y))
                                elif isinstance(w, Circle):
                                    dest_x, dest_y, dest_snap = utils.get_snapped_pos(mx, my, sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout, None, app.wall_idx)
                                    app.current_snap_target = dest_snap
                                    sim.update_wall(app.wall_idx, (dest_x, dest_y), None)
                                if app.mode == config.MODE_EDITOR: sim.apply_constraints()
                        elif app.wall_mode == 'RECT':
                            cur_x, cur_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            sx, sy = app.rect_start
                            base = app.wall_idx
                            sim.update_wall(base, (sx, sy), (cur_x, sy))
                            sim.update_wall(base+1, (cur_x, sy), (cur_x, cur_y))
                            sim.update_wall(base+2, (cur_x, cur_y), (sx, cur_y))
                            sim.update_wall(base+3, (sx, cur_y), (sx, sy))
                        elif app.wall_mode == 'CIRCLE':
                            cur_x, cur_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            c = sim.walls[app.wall_idx]
                            c.radius = max(0.1, math.hypot(cur_x - c.center[0], cur_y - c.center[1]))
                        elif app.wall_mode == 'MOVE_WALL':
                            curr_sim_x, curr_sim_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            prev_sim_x, prev_sim_y = utils.screen_to_sim(app.last_mouse_pos[0], app.last_mouse_pos[1], app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            dx = curr_sim_x - prev_sim_x; dy = curr_sim_y - prev_sim_y
                            app.last_mouse_pos = (mx, my)
                            w = sim.walls[app.wall_idx]
                            if isinstance(w, Line):
                                d_start_x = 0 if w.anchored[0] else dx; d_start_y = 0 if w.anchored[0] else dy
                                d_end_x = 0 if w.anchored[1] else dx; d_end_y = 0 if w.anchored[1] else dy
                                sim.update_wall(app.wall_idx, (w.start[0] + d_start_x, w.start[1] + d_start_y), (w.end[0] + d_end_x, w.end[1] + d_end_y))
                            elif isinstance(w, Circle):
                                d_x = 0 if w.anchored[0] else dx; d_y = 0 if w.anchored[0] else dy
                                w.center[0] += d_x; w.center[1] += d_y
                            if app.mode == config.MODE_EDITOR: sim.apply_constraints()
                        elif app.wall_mode == 'MOVE_GROUP':
                            curr_sim_x, curr_sim_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            prev_sim_x, prev_sim_y = utils.screen_to_sim(app.last_mouse_pos[0], app.last_mouse_pos[1], app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            dx = curr_sim_x - prev_sim_x; dy = curr_sim_y - prev_sim_y
                            app.last_mouse_pos = (mx, my)
                            for w_i in app.current_group_indices:
                                if w_i < len(sim.walls):
                                    w = sim.walls[w_i]
                                    if isinstance(w, Line): sim.update_wall(w_i, (w.start[0]+dx, w.start[1]+dy), (w.end[0]+dx, w.end[1]+dy))
                                    elif isinstance(w, Circle): w.center[0] += dx; w.center[1] += dy
                            if app.mode == config.MODE_EDITOR: sim.apply_constraints()

                elif event.type == pygame.MOUSEWHEEL:
                    app.zoom = max(0.1, min(app.zoom * (1.1 if event.y > 0 else 0.9), 50.0))

            # --- Widget Updates ---
            ui_captured = False
            for el in current_ui_list: 
                if el.handle_event(event): 
                    ui_captured = True
                    # Button Actions (Triggered on Click Release)
                    if app.mode == config.MODE_EDITOR:
                        if el == btn_const_length: trigger_constraint('LENGTH')
                        elif el == btn_const_equal: trigger_constraint('EQUAL')
                        elif el == btn_const_parallel: trigger_constraint('PARALLEL')
                        elif el == btn_const_perp: trigger_constraint('PERPENDICULAR')
                        elif el == btn_const_coincident: trigger_constraint('COINCIDENT')
                        elif el == btn_const_midpoint: trigger_constraint('MIDPOINT')
                        elif el == btn_const_horiz: trigger_constraint('HORIZONTAL')
                        elif el == btn_const_vert: trigger_constraint('VERTICAL')
                        elif el == btn_ae_discard: exit_editor_mode(app.sim_backup_state)
                        elif el == btn_ae_save:
                            if root_tk:
                                f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Geometry Files", "*.json")])
                                if f: 
                                    msg = file_io.save_geometry_file(sim, f)
                                    app.set_status(msg)
                                    exit_editor_mode(app.sim_backup_state)
                    
                    if app.mode == config.MODE_SIM:
                        if el == btn_reset: sim.reset_simulation()
                        elif el == btn_clear: sim.clear_particles()
                        elif el == btn_resize: sim.resize_world(app.input_world.get_value(50.0))
                    
                    if el == btn_undo: sim.undo(); app.set_status("Undo")
                    elif el == btn_redo: sim.redo(); app.set_status("Redo")
            
            if app.mode == config.MODE_SIM:
                if app.input_world.handle_event(event): ui_captured = True
                
            if context_menu:
                if context_menu.handle_event(event):
                    action = context_menu.action
                    if action == "Delete": 
                        sim.remove_wall(context_wall_idx)
                        app.selected_walls.clear(); app.selected_points.clear(); context_menu = None
                    elif action == "Properties":
                        w_props = sim.walls[context_wall_idx].to_dict()
                        prop_dialog = PropertiesDialog(layout['W']//2, layout['H']//2, w_props); context_menu = None
                    elif action == "Set Rotation...":
                        rot_dialog = RotationDialog(layout['W']//2, layout['H']//2, sim.walls[context_wall_idx].anim); context_menu = None
                    elif action == "Anchor":
                        sim.toggle_anchor(context_wall_idx, context_pt_idx); context_menu = None
                    elif action == "Set Length...":
                        val = simpledialog.askfloat("Set Length", "Enter target length:")
                        if val: sim.add_constraint_object(Length(context_wall_idx, val))
                        context_menu = None
                    elif action == "CLOSE": context_menu = None
            
            if prop_dialog:
                if prop_dialog.handle_event(event):
                    if prop_dialog.apply: sim.update_wall_props(context_wall_idx, prop_dialog.get_values()); prop_dialog.apply = False
                    if prop_dialog.done: prop_dialog = None
            if rot_dialog:
                if rot_dialog.handle_event(event):
                    if rot_dialog.apply: sim.set_wall_rotation(context_wall_idx, rot_dialog.get_values()); rot_dialog.apply = False
                    if rot_dialog.done: rot_dialog = None

        # --- Update Physics ---
        if app.mode == config.MODE_SIM and not app.placing_geo_data:
            sim.paused = not btn_play.active
            # Update sim params from sliders
            sim.gravity = slider_gravity.val
            sim.target_temp = slider_temp.val
            sim.damping = slider_damping.val
            sim.dt = slider_dt.val
            sim.sigma = slider_sigma.val
            sim.epsilon = slider_epsilon.val
            sim.skin_distance = slider_skin.val
            sim.use_thermostat = btn_thermostat.active
            sim.use_boundaries = btn_boundaries.active
            
            steps = int(slider_M.val)
            if not sim.paused: sim.step(steps)
            
            # Brush Logic
            if app.is_painting or app.is_erasing:
                mx, my = pygame.mouse.get_pos()
                if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                    sx, sy = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                    if app.is_painting: sim.add_particles_brush(sx, sy, slider_brush_size.val)
                    elif app.is_erasing: sim.delete_particles_brush(sx, sy, slider_brush_size.val)

        # --- Render ---
        renderer.draw_app(app, sim, layout, current_ui_list)
        
        # UI Overlays
        menu_bar.draw(screen, font)
        if context_menu: context_menu.draw(screen, font)
        if prop_dialog: prop_dialog.draw(screen, font)
        if rot_dialog: rot_dialog.draw(screen, font)
        
        pygame.display.flip()
        clock.tick()

    if root_tk: root_tk.destroy()
    pygame.quit()

if __name__ == "__main__":
    main()