import pygame
import numpy as np
import config
import math
import json
import os
import copy 
import time 
from tkinter import filedialog, Tk, simpledialog
from simulation_state import Simulation
from ui_widgets import SmartSlider, Button, InputField, ContextMenu, PropertiesDialog, MenuBar, RotationDialog
from geometry import Line, Point
from constraints import create_constraint, Coincident, Length, EqualLength, Angle

# --- Layout Constants ---
TOP_MENU_H = 30
MODE_SIM = 0
MODE_EDITOR = 1

def get_connected_group(sim, start_wall_idx):
    group = {start_wall_idx}
    queue = [start_wall_idx]
    
    adjacency = {}
    for c in sim.constraints:
        if c.type == 'COINCIDENT':
            # c.indices is [(w1, p1), (w2, p2)]
            idx_list = c.indices
            w1, w2 = idx_list[0][0], idx_list[1][0]
            
            if w1 not in adjacency: adjacency[w1] = []
            if w2 not in adjacency: adjacency[w2] = []
            adjacency[w1].append(w2)
            adjacency[w2].append(w1)

    while queue:
        current = queue.pop(0)
        if current in adjacency:
            for neighbor in adjacency[current]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
    return group

def is_group_anchored(sim, group_indices):
    for idx in group_indices:
        w = sim.walls[idx]
        if w.anchored[0] or w.anchored[1]:
            return True
    return False

def sim_to_screen(x, y, zoom, pan_x, pan_y, world_size, layout):
    cx_world = world_size / 2.0
    cy_world = world_size / 2.0
    cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
    cy_screen = TOP_MENU_H + (layout['MID_H'] / 2.0)
    base_scale = (layout['MID_W'] - 50) / world_size
    final_scale = base_scale * zoom
    sx = cx_screen + (x - cx_world) * final_scale + pan_x
    sy = cy_screen + (y - cy_world) * final_scale + pan_y
    return int(sx), int(sy)

def screen_to_sim(sx, sy, zoom, pan_x, pan_y, world_size, layout):
    cx_world = world_size / 2.0
    cy_world = world_size / 2.0
    cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
    cy_screen = TOP_MENU_H + (layout['MID_H'] / 2.0)
    base_scale = (layout['MID_W'] - 50) / world_size
    final_scale = base_scale * zoom
    x = (sx - pan_x - cx_screen) / final_scale + cx_world
    y = (sy - pan_y - cy_screen) / final_scale + cy_world
    return x, y

def get_grouped_points(sim, zoom, pan_x, pan_y, world_size, layout):
    point_map = {}
    
    for i, w in enumerate(sim.walls):
        # Using Object attributes
        points_to_process = []
        if np.array_equal(w.start, w.end): # Standalone
             points_to_process.append((w.start, 0))
        else:
            points_to_process.append((w.start, 0))
            points_to_process.append((w.end, 1))
        
        for pt, end_idx in points_to_process:
            sx, sy = sim_to_screen(pt[0], pt[1], zoom, pan_x, pan_y, world_size, layout)
            
            found_key = None
            for k in point_map:
                if abs(k[0]-sx) <= 3 and abs(k[1]-sy) <= 3:
                    found_key = k
                    break
            
            if found_key:
                point_map[found_key].append((i, end_idx))
            else:
                point_map[(sx, sy)] = [(i, end_idx)]
                
    return point_map

def get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, world_size, layout, anchor_pos=None, exclude_wall_idx=-1):
    sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, world_size, layout)
    final_x, final_y = sim_x, sim_y
    is_snapped_to_vertex = False
    snapped_target = None
    
    if pygame.key.get_mods() & pygame.KMOD_CTRL:
        snap_threshold_px = 15 
        best_dist = float('inf')
        for i, w in enumerate(sim.walls):
            if i == exclude_wall_idx: continue
            points = [(w.start, 0), (w.end, 1)]
            for pt_pos, pt_idx in points:
                sx, sy = sim_to_screen(pt_pos[0], pt_pos[1], zoom, pan_x, pan_y, world_size, layout)
                dist = math.hypot(mx - sx, my - sy)
                if dist < snap_threshold_px and dist < best_dist:
                    best_dist = dist
                    final_x, final_y = pt_pos
                    snapped_target = (i, pt_idx)
                    is_snapped_to_vertex = True

    if (pygame.key.get_mods() & pygame.KMOD_SHIFT) and anchor_pos is not None and not is_snapped_to_vertex:
        dx = final_x - anchor_pos[0]
        dy = final_y - anchor_pos[1]
        if abs(dx) > abs(dy): final_y = anchor_pos[1] 
        else: final_x = anchor_pos[0] 

    return final_x, final_y, snapped_target

def calculate_current_temp(vel_x, vel_y, count, mass):
    if count == 0: return 0.0
    vx = vel_x[:count]
    vy = vel_y[:count]
    ke_total = 0.5 * mass * np.sum(vx**2 + vy**2)
    return ke_total / count

# Serialization Helpers
def save_file(sim, filename, settings=None):
    if not filename: return "Cancelled"
    # Serializing Objects
    walls_dicts = [w.to_dict() for w in sim.walls]
    constraints_dicts = [c.to_dict() for c in sim.constraints]
    
    state = {
        'count': sim.count,
        'pos_x': sim.pos_x[:sim.count].tolist(),
        'pos_y': sim.pos_y[:sim.count].tolist(),
        'vel_x': sim.vel_x[:sim.count].tolist(),
        'vel_y': sim.vel_y[:sim.count].tolist(),
        'is_static': sim.is_static[:sim.count].tolist(),
        'atom_sigma': sim.atom_sigma[:sim.count].tolist(),
        'atom_eps_sqrt': sim.atom_eps_sqrt[:sim.count].tolist(),
        'walls': walls_dicts,
        'constraints': constraints_dicts,
        'world_size': sim.world_size,
        'settings': settings if settings else {}
    }
    try:
        with open(filename, 'w') as f: json.dump(state, f)
        return f"Saved to {os.path.basename(filename)}"
    except Exception as e: return f"Error: {e}"

def load_file(sim, filename):
    if not filename: return False, "Cancelled", {}
    try:
        with open(filename, 'r') as f: data = json.load(f)
        
        # Hydrate Objects
        walls = [Line.from_dict(w) for w in data['walls']]
        constraints = []
        for c_data in data.get('constraints', []):
            c = create_constraint(c_data)
            if c: constraints.append(c)
            
        restored_state = {}
        for k, v in data.items():
            if k in ['walls', 'constraints']: continue # Handled above
            if k in ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'is_static', 'atom_sigma', 'atom_eps_sqrt']:
                dtype = np.float32
                if k == 'is_static': dtype = np.int32
                restored_state[k] = np.array(v, dtype=dtype)
            else: restored_state[k] = v
        
        restored_state['walls'] = walls
        restored_state['constraints'] = constraints
        
        sim.restore_state(restored_state)
        return True, f"Loaded {os.path.basename(filename)}", data.get('settings', {})
    except Exception as e: return False, f"Error: {e}", {}

def save_asset_file(sim, filename):
    if not filename: return "Cancelled"
    asset_data = sim.export_asset_data() # Already returns dicts
    if not asset_data: return "Empty Asset"
    wrapper = {'type': 'ASSET', 'data': asset_data}
    try:
        with open(filename, 'w') as f: json.dump(wrapper, f)
        return f"Asset Saved: {os.path.basename(filename)}"
    except Exception as e: return f"Error: {e}"

def load_asset_file(filename):
    if not filename: return None
    try:
        with open(filename, 'r') as f: data = json.load(f)
        if data.get('type') != 'ASSET': return None
        return data.get('data', {}) 
    except Exception as e: return None

def main():
    try:
        root_tk = Tk()
        root_tk.withdraw()
    except: root_tk = None

    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fast MD - Object Oriented Refactor")
    
    font = pygame.font.SysFont("segoeui", 15)
    big_font = pygame.font.SysFont("segoeui", 22)
    clock = pygame.time.Clock()
    
    sim = Simulation()
    app_mode = MODE_SIM
    
    layout = {
        'W': config.WINDOW_WIDTH, 'H': config.WINDOW_HEIGHT,
        'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
        'RIGHT_W': config.PANEL_RIGHT_WIDTH, 'RIGHT_X': config.WINDOW_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_X': config.PANEL_LEFT_WIDTH, 'MID_W': config.WINDOW_WIDTH - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_H': config.WINDOW_HEIGHT - TOP_MENU_H
    }

    menu_bar = MenuBar(layout['W'], TOP_MENU_H)
    menu_bar.items["File"] = ["New Simulation", "Open...", "Save", "Save As...", "---", "Create New Asset", "Add Existing Asset"] 
    
    current_filepath = None
    status_msg = ""; status_time = 0
    
    # UI Setup
    lp_margin = 10; lp_curr_y = TOP_MENU_H + 20
    btn_play = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Play/Pause", active=False, color_active=(60, 120, 60), color_inactive=(180, 60, 60)); lp_curr_y += 50
    btn_clear = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Clear", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_curr_y += 50
    btn_reset = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Reset", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_curr_y += 50
    btn_undo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Undo", active=False, toggle=False); lp_curr_y += 45
    btn_redo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Redo", active=False, toggle=False)
    
    rp_start_x = layout['RIGHT_X'] + 15; rp_width = layout['RIGHT_W'] - 30; rp_curr_y = TOP_MENU_H + 120
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
    btn_tool_brush = Button(rp_start_x, rp_curr_y, btn_w, 30, "Brush", active=True, toggle=False)
    btn_tool_line = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Line", active=False, toggle=False); rp_curr_y += 40
    btn_tool_ref = Button(rp_start_x, rp_curr_y, btn_w, 30, "Reference Line", active=False, toggle=False)
    btn_tool_point = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Point", active=False, toggle=False); rp_curr_y += 40
    slider_brush_size = SmartSlider(rp_start_x, rp_curr_y, rp_width, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5); rp_curr_y+=60
    input_world = InputField(rp_start_x + 80, rp_curr_y, 60, 25, str(config.DEFAULT_WORLD_SIZE))
    btn_resize = Button(rp_start_x + 150, rp_curr_y, rp_width - 150, 25, "Resize & Restart", active=False, toggle=False)
    
    # Asset Editor UI
    ae_curr_y = TOP_MENU_H + 40
    btn_ae_save = Button(rp_start_x, ae_curr_y, rp_width, 40, "Save Asset", active=False, toggle=False, color_inactive=(50, 120, 50)); ae_curr_y+=50
    btn_ae_discard = Button(rp_start_x, ae_curr_y, rp_width, 40, "Discard & Exit", active=False, toggle=False, color_inactive=(150, 50, 50)); ae_curr_y+=50
    
    c_btn_w = (rp_width - 10) // 2
    btn_const_coincident = Button(rp_start_x, ae_curr_y, rp_width, 30, "Coincident (Pt-Pt)", toggle=False); ae_curr_y+=35
    btn_const_length = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Fix Length", toggle=False)
    btn_const_equal = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Equal Len", toggle=False); ae_curr_y+=35
    btn_const_parallel = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Parallel", toggle=False)
    btn_const_perp = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Perpendic", toggle=False); ae_curr_y+=35
    btn_const_horiz = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Horizontal", toggle=False)
    btn_const_vert = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Vertical", toggle=False); ae_curr_y+=35

    ui_sim_elements = [btn_play, btn_clear, btn_reset, btn_undo, btn_redo, slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, btn_thermostat, btn_boundaries, slider_brush_size, btn_resize]
    ui_editor_elements = [btn_ae_save, btn_ae_discard, btn_undo, btn_redo, btn_clear, btn_const_coincident, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert]
    right_panel_elements = [slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, btn_thermostat, btn_boundaries, btn_tool_brush, btn_tool_line, btn_tool_ref, btn_tool_point, slider_brush_size, input_world, btn_resize, btn_ae_save, btn_ae_discard, btn_const_coincident, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert]

    # App State
    current_tool = 0; zoom = 1.0; pan_x = 0.0; pan_y = 0.0
    is_panning = False; last_mouse_pos = (0, 0); is_painting = False; is_erasing = False
    wall_mode = None; wall_idx = -1; wall_pt = -1
    context_menu = None; prop_dialog = None; rot_dialog = None; context_wall_idx = -1; context_pt_idx = None
    placing_asset_data = None; sim_backup_state = None
    
    selected_walls = set(); selected_points = set()
    pending_constraint = None; pending_targets_walls = []; pending_targets_points = []
    temp_constraint_active = False; current_group_indices = []; current_snap_target = None; new_wall_start_snap = None

    def enter_editor_mode():
        nonlocal app_mode, zoom, pan_x, pan_y, status_msg, status_time, sim_backup_state, current_tool
        # Sim backup uses deepcopy of object lists now
        sim_backup_state = {
            'count': sim.count, 'walls': copy.deepcopy(sim.walls), 'constraints': copy.deepcopy(sim.constraints), 'world_size': sim.world_size,
            # (Truncated: other arrays handled by snapshot logic in sim but we do manual here)
        }
        # Actually easier to use sim.snapshot logic but we need to return to a blank state.
        # Let's rely on sim methods.
        sim.snapshot() # Push current sim state to undo stack
        sim_backup_state = sim.undo_stack.pop() # Grab it back (hacky but works)
        
        sim.clear_particles(snapshot=False)
        sim.world_size = 30.0; sim.walls = []; sim.constraints = []
        app_mode = MODE_EDITOR
        current_tool = 1; btn_tool_line.active = True; btn_tool_brush.active = False; btn_tool_ref.active = False; btn_tool_point.active = False
        zoom = 1.5; pan_x = 0; pan_y = 0
        status_msg = "Entered Asset Editor"; status_time = time.time()

    def exit_editor_mode(restore_state):
        nonlocal app_mode, zoom, pan_x, pan_y, status_msg, status_time
        sim.clear_particles(snapshot=False)
        if restore_state: sim.restore_state(restore_state)
        else: sim.world_size = config.DEFAULT_WORLD_SIZE
        app_mode = MODE_SIM
        zoom = 1.0; pan_x = 0; pan_y = 0
        status_msg = "Returned to Simulation"; status_time = time.time()

    def trigger_constraint(ctype):
        nonlocal pending_constraint, status_msg, status_time
        applied = False
        if ctype == 'LENGTH' and len(selected_walls) == 1:
            w_idx = list(selected_walls)[0]
            w = sim.walls[w_idx]
            curr_len = np.linalg.norm(w.end - w.start)
            sim.add_constraint_object(Length(w_idx, curr_len))
            applied = True
        elif ctype == 'EQUAL' and len(selected_walls) == 2:
            sl = list(selected_walls)
            sim.add_constraint_object(EqualLength(sl[0], sl[1])); applied = True
        elif ctype in ['PARALLEL', 'PERPENDICULAR'] and len(selected_walls) == 2:
            sl = list(selected_walls)
            sim.add_constraint_object(Angle(ctype, sl[0], sl[1])); applied = True
        elif ctype in ['HORIZONTAL', 'VERTICAL'] and len(selected_walls) >= 1:
            for w_idx in selected_walls: sim.add_constraint_object(Angle(ctype, w_idx))
            applied = True
        elif ctype == 'COINCIDENT' and len(selected_points) == 2:
            sp = list(selected_points) # [(w1, p1), (w2, p2)]
            sim.add_constraint_object(Coincident(sp[0][0], sp[0][1], sp[1][0], sp[1][1])); applied = True
            
        if applied:
            status_msg = f"Applied {ctype}"; status_time = time.time()
            selected_walls.clear(); selected_points.clear(); pending_constraint = None; reset_constraint_buttons()
        else:
            pending_constraint = ctype
            pending_targets_walls.clear(); pending_targets_points.clear()
            if selected_walls: pending_targets_walls.extend(list(selected_walls))
            if selected_points: pending_targets_points.extend(list(selected_points))
            selected_walls.clear(); selected_points.clear()
            status_msg = f"Select targets for {ctype}..."; status_time = time.time()
            reset_constraint_buttons()
            if ctype == 'LENGTH': btn_const_length.active = True
            elif ctype == 'EQUAL': btn_const_equal.active = True
            elif ctype == 'PARALLEL': btn_const_parallel.active = True
            elif ctype == 'PERPENDICULAR': btn_const_perp.active = True
            elif ctype == 'COINCIDENT': btn_const_coincident.active = True
            elif ctype == 'HORIZONTAL': btn_const_horiz.active = True
            elif ctype == 'VERTICAL': btn_const_vert.active = True

    def reset_constraint_buttons():
        btn_const_length.active = False; btn_const_equal.active = False; btn_const_parallel.active = False
        btn_const_perp.active = False; btn_const_coincident.active = False; btn_const_horiz.active = False; btn_const_vert.active = False

    def handle_pending_constraint_click(wall_idx=None, pt_idx=None):
        nonlocal pending_constraint, status_msg, status_time
        if pending_constraint in ['LENGTH', 'HORIZONTAL', 'VERTICAL']:
            if wall_idx is not None:
                if pending_constraint == 'LENGTH':
                    w = sim.walls[wall_idx]
                    curr_len = np.linalg.norm(w.end - w.start)
                    sim.add_constraint_object(Length(wall_idx, curr_len))
                else:
                    sim.add_constraint_object(Angle(pending_constraint, wall_idx))
                status_msg = f"Applied {pending_constraint}"; status_time = time.time()
                pending_constraint = None; reset_constraint_buttons()
        elif pending_constraint in ['EQUAL', 'PARALLEL', 'PERPENDICULAR']:
            if wall_idx is not None:
                if wall_idx not in pending_targets_walls:
                    pending_targets_walls.append(wall_idx)
                    status_msg = f"Selected 1/2 for {pending_constraint}"
                    if len(pending_targets_walls) == 2:
                        if pending_constraint == 'EQUAL': sim.add_constraint_object(EqualLength(pending_targets_walls[0], pending_targets_walls[1]))
                        else: sim.add_constraint_object(Angle(pending_constraint, pending_targets_walls[0], pending_targets_walls[1]))
                        status_msg = f"Applied {pending_constraint}"; status_time = time.time()
                        pending_constraint = None; reset_constraint_buttons()
        elif pending_constraint == 'COINCIDENT':
            if pt_idx is not None:
                if pt_idx not in pending_targets_points:
                    pending_targets_points.append(pt_idx)
                    status_msg = f"Selected 1/2 Points"
                    if len(pending_targets_points) == 2:
                        sim.add_constraint_object(Coincident(pending_targets_points[0][0], pending_targets_points[0][1], pending_targets_points[1][0], pending_targets_points[1][1]))
                        status_msg = "Applied Coincident"; status_time = time.time()
                        pending_constraint = None; reset_constraint_buttons()

    running = True
    while running:
        current_ui_list = ui_sim_elements if app_mode == MODE_SIM else ui_editor_elements
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                diff_w = new_w - layout['W']
                layout['W'] = new_w; layout['H'] = new_h
                layout['RIGHT_X'] = new_w - layout['RIGHT_W']
                layout['MID_W'] = new_w - layout['LEFT_W'] - layout['RIGHT_W']
                layout['MID_H'] = new_h - TOP_MENU_H
                for widget in right_panel_elements: widget.move(diff_w, 0)
                menu_bar.resize(new_w)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL): sim.undo(); status_msg = "Undo"; status_time = time.time()
                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL): sim.redo(); status_msg = "Redo"; status_time = time.time()
                if event.key == pygame.K_ESCAPE:
                    if placing_asset_data: placing_asset_data = None; status_msg = "Cancelled"
                    elif pending_constraint: pending_constraint = None; reset_constraint_buttons(); status_msg = "Cancelled"
                    else: selected_walls.clear(); selected_points.clear()
            
            menu_clicked = menu_bar.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and menu_bar.active_menu:
                if menu_bar.dropdown_rect and menu_bar.dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - menu_bar.dropdown_rect.y - 5
                    idx = rel_y // 30
                    opts = menu_bar.items[menu_bar.active_menu]
                    if 0 <= idx < len(opts):
                        selection = opts[idx]
                        menu_bar.active_menu = None 
                        if selection == "New Simulation": sim.reset_simulation(); input_world.set_value(config.DEFAULT_WORLD_SIZE)
                        elif selection == "Create New Asset": 
                            if app_mode == MODE_SIM: enter_editor_mode()
                        elif selection == "Add Existing Asset":
                            if app_mode == MODE_SIM and root_tk:
                                f = filedialog.askopenfilename(filetypes=[("Asset Files", "*.json")])
                                if f: 
                                    data = load_asset_file(f)
                                    if data: placing_asset_data = data; status_msg = "Place Asset"; status_time = time.time()
                        elif root_tk:
                            if selection == "Save As..." or (selection == "Save" and not current_filepath):
                                f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
                                if f: current_filepath = f; msg = save_file(sim, f); status_msg = msg; status_time = time.time()
                            elif selection == "Save" and current_filepath: msg = save_file(sim, current_filepath); status_msg = msg; status_time = time.time()
                            elif selection == "Open...":
                                f = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
                                if f: 
                                    current_filepath = f; success, msg, lset = load_file(sim, f); status_msg = msg; status_time = time.time()
                                    if success: input_world.set_value(sim.world_size); zoom=1.0; pan_x=0; pan_y=0
                    continue 
                elif not menu_clicked: menu_bar.active_menu = None
            if menu_clicked: continue

            if context_menu:
                if context_menu.handle_event(event):
                    action = context_menu.action
                    if action == "Delete": 
                        sim.remove_wall(context_wall_idx)
                        selected_walls.clear(); selected_points.clear(); context_menu = None
                    elif action == "Properties":
                        # We pass the wall dict-like interface via vars() or just handle prop dialog update
                        # Prop dialog expects dict, let's adapt it
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
                continue 
            if prop_dialog:
                if prop_dialog.handle_event(event):
                    if prop_dialog.apply: sim.update_wall_props(context_wall_idx, prop_dialog.get_values()); prop_dialog.apply = False
                    if prop_dialog.done: prop_dialog = None
                continue
            if rot_dialog:
                if rot_dialog.handle_event(event):
                    if rot_dialog.apply: sim.set_wall_rotation(context_wall_idx, rot_dialog.get_values()); rot_dialog.apply = False
                    if rot_dialog.done: rot_dialog = None
                continue

            mouse_in_ui = (event.type == pygame.MOUSEBUTTONDOWN and (event.pos[0] > layout['RIGHT_X'] or event.pos[0] < layout['LEFT_W'] or event.pos[1] < TOP_MENU_H))
            ui_captured = False
            for el in current_ui_list:
                if el.handle_event(event): ui_captured = True
            if app_mode == MODE_SIM and input_world.handle_event(event): ui_captured = True
            
            # Tools Logic
            if btn_tool_line.handle_event(event): current_tool = 1; btn_tool_line.active = True; btn_tool_brush.active = False; btn_tool_ref.active = False; btn_tool_point.active=False; ui_captured = True
            if btn_tool_ref.handle_event(event): current_tool = 2; btn_tool_ref.active = True; btn_tool_line.active = False; btn_tool_brush.active = False; btn_tool_point.active=False; ui_captured = True
            if btn_tool_point.handle_event(event): current_tool = 3; btn_tool_point.active = True; btn_tool_ref.active = False; btn_tool_line.active = False; btn_tool_brush.active = False; ui_captured = True
            if app_mode == MODE_SIM and btn_tool_brush.handle_event(event): current_tool = 0; btn_tool_brush.active = True; btn_tool_line.active = False; btn_tool_ref.active = False; ui_captured = True
            
            if app_mode == MODE_EDITOR:
                if btn_const_length.clicked: trigger_constraint('LENGTH')
                if btn_const_equal.clicked: trigger_constraint('EQUAL')
                if btn_const_parallel.clicked: trigger_constraint('PARALLEL')
                if btn_const_perp.clicked: trigger_constraint('PERPENDICULAR')
                if btn_const_coincident.clicked: trigger_constraint('COINCIDENT')
                if btn_const_horiz.clicked: trigger_constraint('HORIZONTAL')
                if btn_const_vert.clicked: trigger_constraint('VERTICAL')
                if btn_ae_discard.clicked: exit_editor_mode(sim_backup_state); ui_captured = True
                if btn_ae_save.clicked and root_tk:
                    f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Asset Files", "*.json")])
                    if f: msg = save_asset_file(sim, f); status_msg = msg; status_time = time.time(); exit_editor_mode(sim_backup_state)
                    ui_captured = True
            
            if mouse_in_ui or ui_captured:
                if app_mode == MODE_SIM:
                    if btn_reset.clicked: sim.reset_simulation()
                    if btn_clear.clicked: sim.clear_particles()
                    if btn_resize.clicked: sim.resize_world(input_world.get_value(50.0))
                if btn_undo.clicked: sim.undo()
                if btn_redo.clicked: sim.redo()
                continue 

            if event.type == pygame.MOUSEWHEEL:
                zoom = max(0.1, min(zoom * (1.1 if event.y > 0 else 0.9), 50.0))
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2: is_panning = True; last_mouse_pos = event.pos
                elif event.button == 3: # Right Click
                    mx, my = event.pos
                    if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                        if placing_asset_data: placing_asset_data = None
                        elif pending_constraint: pending_constraint = None; reset_constraint_buttons(); status_msg = "Cancelled"
                        else:
                            sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                            point_map = get_grouped_points(sim, zoom, pan_x, pan_y, sim.world_size, layout)
                            hit_pt = None
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
                            
                            if hit_pt and app_mode == MODE_EDITOR:
                                context_wall_idx = hit_pt[0]; context_pt_idx = hit_pt[1]
                                context_menu = ContextMenu(mx, my, ["Anchor"])
                            else:
                                hit = -1
                                rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * zoom)
                                for i, w in enumerate(sim.walls):
                                    if np.array_equal(w.start, w.end): continue
                                    p1=w.start; p2=w.end; p3=np.array([sim_x, sim_y])
                                    d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                                    if len_sq == 0: dist = np.linalg.norm(p3-p1)
                                    else:
                                        t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                                        proj = p1 + t*d_vec
                                        dist = np.linalg.norm(p3-proj)
                                    if dist < rad_sim: hit = i; break
                                
                                if hit != -1: context_menu = ContextMenu(mx, my, ["Properties", "Set Length...", "Set Rotation...", "Delete"]); context_wall_idx = hit
                                else: is_erasing = True; sim.snapshot()
                
                elif event.button == 1: # Left Click
                    mx, my = event.pos
                    if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                        if placing_asset_data:
                            sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                            sim.place_asset(placing_asset_data, sim_x, sim_y)
                        elif current_tool == 0 and app_mode == MODE_SIM: is_painting = True; sim.snapshot()
                        elif current_tool == 3 and app_mode == MODE_EDITOR:
                            # Add Point
                            start_x, start_y, start_snap = get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, sim.world_size, layout)
                            sim.snapshot()
                            sim.add_wall((start_x, start_y), (start_x, start_y), is_ref=False)
                            wall_idx = len(sim.walls)-1
                            if start_snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                                sim.add_constraint_object(Coincident(wall_idx, 0, start_snap[0], start_snap[1]))
                        
                        elif current_tool == 1 or current_tool == 2 or app_mode == MODE_EDITOR:
                            sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                            rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * zoom)
                            
                            point_map = get_grouped_points(sim, zoom, pan_x, pan_y, sim.world_size, layout)
                            hit_pt = None
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
                                if pending_constraint: handle_pending_constraint_click(pt_idx=hit_pt)
                                else:
                                    if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                        if hit_pt not in selected_points: selected_walls.clear(); selected_points.clear()
                                    if hit_pt in selected_points: selected_points.remove(hit_pt)
                                    else: selected_points.add(hit_pt)
                                    wall_mode = 'EDIT'; wall_idx = hit_pt[0]; wall_pt = hit_pt[1]
                                    sim.snapshot()
                            else:
                                hit_wall = -1
                                for i, w in enumerate(sim.walls):
                                    if np.array_equal(w.start, w.end): continue
                                    p1=w.start; p2=w.end; p3=np.array([sim_x, sim_y])
                                    d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                                    if len_sq == 0: dist = np.linalg.norm(p3-p1)
                                    else:
                                        t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                                        proj = p1 + t*d_vec
                                        dist = np.linalg.norm(p3-proj)
                                    if dist < rad_sim: hit_wall = i; break
                                
                                if hit_wall != -1:
                                    if pending_constraint: handle_pending_constraint_click(wall_idx=hit_wall)
                                    else:
                                        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                            if hit_wall not in selected_walls: selected_walls.clear(); selected_points.clear()
                                            selected_walls.add(hit_wall)
                                        else:
                                            if hit_wall in selected_walls: selected_walls.remove(hit_wall)
                                            else: selected_walls.add(hit_wall)
                                        
                                        target_group = get_connected_group(sim, hit_wall)
                                        if not is_group_anchored(sim, target_group):
                                            wall_mode = 'MOVE_GROUP'; current_group_indices = list(target_group)
                                            sim.snapshot(); last_mouse_pos = event.pos
                                        else:
                                            wall_mode = 'MOVE_WALL'; wall_idx = hit_wall
                                            sim.snapshot(); last_mouse_pos = event.pos
                                            w = sim.walls[hit_wall]
                                            drag_start_length = np.linalg.norm(w.end - w.start)
                                            if app_mode == MODE_EDITOR:
                                                c = Length(hit_wall, drag_start_length)
                                                c.temp = True
                                                sim.constraints.append(c)
                                                temp_constraint_active = True
                                else:
                                    if not (pygame.key.get_mods() & pygame.KMOD_SHIFT): selected_walls.clear(); selected_points.clear()
                                    if not pending_constraint:
                                        wall_mode = 'NEW'; sim.snapshot()
                                        start_x, start_y, start_snap = get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, sim.world_size, layout)
                                        is_ref = (current_tool == 2)
                                        sim.add_wall((start_x, start_y), (start_x, start_y), is_ref=is_ref)
                                        wall_idx = len(sim.walls)-1; wall_pt = 1
                                        current_snap_target = None; new_wall_start_snap = None
                                        if pygame.key.get_mods() & pygame.KMOD_CTRL: new_wall_start_snap = start_snap

            elif event.type == pygame.MOUSEBUTTONUP:
                if wall_mode == 'NEW':
                    if new_wall_start_snap and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                         sim.add_constraint_object(Coincident(wall_idx, 0, new_wall_start_snap[0], new_wall_start_snap[1]))
                    if current_snap_target and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                         sim.add_constraint_object(Coincident(wall_idx, 1, current_snap_target[0], current_snap_target[1]))
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                         w = sim.walls[wall_idx]
                         dx = abs(w.start[0] - w.end[0]); dy = abs(w.start[1] - w.end[1])
                         if dy < 0.001: sim.add_constraint_object(Angle('HORIZONTAL', wall_idx))
                         elif dx < 0.001: sim.add_constraint_object(Angle('VERTICAL', wall_idx))

                elif wall_mode == 'EDIT':
                    if current_snap_target and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                         sim.add_constraint_object(Coincident(wall_idx, wall_pt, current_snap_target[0], current_snap_target[1]))

                is_panning = False; is_painting = False; is_erasing = False
                wall_mode = None; wall_idx = -1; current_group_indices = []
                if temp_constraint_active:
                    sim.constraints = [c for c in sim.constraints if not c.temp]
                    temp_constraint_active = False
                if app_mode == MODE_EDITOR: sim.apply_constraints()

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if is_panning: pan_x += mx - last_mouse_pos[0]; pan_y += my - last_mouse_pos[1]; last_mouse_pos = (mx, my)
                
                if wall_mode is not None:
                    if wall_mode == 'EDIT' or wall_mode == 'NEW':
                        if wall_idx < len(sim.walls):
                            w = sim.walls[wall_idx]
                            anchor = w.end if wall_pt == 0 else w.start
                            dest_x, dest_y, dest_snap = get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, sim.world_size, layout, anchor, wall_idx)
                            current_snap_target = dest_snap
                            
                            if wall_pt == 0: sim.update_wall(wall_idx, (dest_x, dest_y), w.end)
                            else: sim.update_wall(wall_idx, w.start, (dest_x, dest_y))
                            if app_mode == MODE_EDITOR: sim.apply_constraints()
                    
                    elif wall_mode == 'MOVE_WALL':
                         if wall_idx < len(sim.walls):
                            curr_sim_x, curr_sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                            prev_sim_x, prev_sim_y = screen_to_sim(last_mouse_pos[0], last_mouse_pos[1], zoom, pan_x, pan_y, sim.world_size, layout)
                            dx = curr_sim_x - prev_sim_x; dy = curr_sim_y - prev_sim_y
                            last_mouse_pos = (mx, my)
                            w = sim.walls[wall_idx]
                            
                            d_start_x = 0 if w.anchored[0] else dx; d_start_y = 0 if w.anchored[0] else dy
                            d_end_x = 0 if w.anchored[1] else dx; d_end_y = 0 if w.anchored[1] else dy
                            sim.update_wall(wall_idx, (w.start[0] + d_start_x, w.start[1] + d_start_y), (w.end[0] + d_end_x, w.end[1] + d_end_y))
                            if app_mode == MODE_EDITOR: sim.apply_constraints()
                    
                    elif wall_mode == 'MOVE_GROUP':
                        curr_sim_x, curr_sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                        prev_sim_x, prev_sim_y = screen_to_sim(last_mouse_pos[0], last_mouse_pos[1], zoom, pan_x, pan_y, sim.world_size, layout)
                        dx = curr_sim_x - prev_sim_x; dy = curr_sim_y - prev_sim_y
                        last_mouse_pos = (mx, my)
                        for w_i in current_group_indices:
                            if w_i < len(sim.walls):
                                w = sim.walls[w_i]
                                sim.update_wall(w_i, (w.start[0]+dx, w.start[1]+dy), (w.end[0]+dx, w.end[1]+dy))
                        if app_mode == MODE_EDITOR: sim.apply_constraints()

        # Update
        if not prop_dialog and not rot_dialog:
            if app_mode == MODE_SIM:
                sim.paused = not btn_play.active
                sim.gravity = slider_gravity.val; sim.target_temp = slider_temp.val; sim.damping = slider_damping.val
                sim.dt = slider_dt.val; sim.sigma = slider_sigma.val; sim.epsilon = slider_epsilon.val
                sim.skin_distance = slider_skin.val; sim.use_thermostat = btn_thermostat.active; sim.use_boundaries = btn_boundaries.active
                steps = int(slider_M.val)
            else: sim.paused = True; steps = 0
            
            mx, my = pygame.mouse.get_pos()
            if is_painting or is_erasing:
                    if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                        sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                        if is_painting: sim.add_particles_brush(sim_x, sim_y, slider_brush_size.val)
                        elif is_erasing: sim.delete_particles_brush(sim_x, sim_y, slider_brush_size.val)
            if not sim.paused: sim.step(steps)
            
        # Render
        screen.fill(config.BACKGROUND_COLOR)
        sim_rect = pygame.Rect(layout['MID_X'], TOP_MENU_H, layout['MID_W'], layout['MID_H'])
        screen.set_clip(sim_rect)
        tl = sim_to_screen(0, 0, zoom, pan_x, pan_y, sim.world_size, layout)
        br = sim_to_screen(sim.world_size, sim.world_size, zoom, pan_x, pan_y, sim.world_size, layout)
        g_col = config.GRID_COLOR if app_mode == MODE_SIM else (50, 60, 50)
        pygame.draw.rect(screen, g_col, (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]), 2)
        
        if app_mode == MODE_SIM:
            for i in range(sim.count):
                sx, sy = sim_to_screen(sim.pos_x[i], sim.pos_y[i], zoom, pan_x, pan_y, sim.world_size, layout)
                if 0 < sx < layout['W'] and 0 < sy < layout['H']:
                    is_stat = sim.is_static[i]
                    col = config.COLOR_STATIC if is_stat else config.COLOR_DYNAMIC
                    atom_sig = sim.atom_sigma[i]
                    rad = max(2, int(atom_sig * config.PARTICLE_RADIUS_SCALE * ((layout['MID_W']-50)/sim.world_size) * zoom))
                    pygame.draw.circle(screen, col, (sx, sy), rad)
        else:
            cx, cy = sim_to_screen(sim.world_size/2, sim.world_size/2, zoom, pan_x, pan_y, sim.world_size, layout)
            pygame.draw.line(screen, (50, 50, 50), (cx-10, cy), (cx+10, cy))
            pygame.draw.line(screen, (50, 50, 50), (cx, cy-10), (cx, cy+10))
        
        # Draw Walls
        transform = lambda x, y: sim_to_screen(x, y, zoom, pan_x, pan_y, sim.world_size, layout)
        for i, w in enumerate(sim.walls):
            if np.array_equal(w.start, w.end): continue
            
            selected = (i in selected_walls)
            pending = (pending_constraint and i in pending_targets_walls)
            w.render(screen, transform, selected, pending)

        # Draw Points (Stacked)
        if app_mode == MODE_EDITOR:
            point_map = get_grouped_points(sim, zoom, pan_x, pan_y, sim.world_size, layout)
            anchored_points_draw_list = []
            base_r = 5; step_r = 4
            
            for center_pos, items in point_map.items():
                cx, cy = center_pos
                count = len(items)
                for k in range(count - 1, -1, -1):
                    w_idx, pt_idx = items[k]
                    radius = base_r + (k * step_r)
                    color = (200, 200, 200) 
                    if (w_idx, pt_idx) in selected_points: color = (0, 255, 255)
                    elif pending_constraint and (w_idx, pt_idx) in pending_targets_points: color = (100, 255, 100)
                    
                    pygame.draw.circle(screen, (30,30,30), (cx, cy), radius)
                    pygame.draw.circle(screen, color, (cx, cy), radius, 2)
                    
                    w = sim.walls[w_idx]
                    if not np.array_equal(w.start, w.end):
                        p_my = w.start if pt_idx == 0 else w.end
                        p_other = w.end if pt_idx == 0 else w.start
                        dx = p_other[0] - p_my[0]; dy = p_other[1] - p_my[1]
                        l = math.hypot(dx, dy)
                        if l > 0.0001:
                            dx /= l; dy /= l
                            sx = cx + dx * radius; sy = cy + dy * radius
                            ex = cx + dx * (radius + 8); ey = cy + dy * (radius + 8)
                            pygame.draw.line(screen, color, (sx, sy), (ex, ey), 2)
                    
                    if w.anchored[pt_idx]: anchored_points_draw_list.append((cx, cy))
            
            for pt in anchored_points_draw_list: pygame.draw.circle(screen, (255, 50, 50), pt, 3)

        if placing_asset_data:
            mx, my = pygame.mouse.get_pos()
            if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                sim_mx, sim_my = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                for wd in placing_asset_data.get('walls', []):
                    # quick render based on dict
                    ws = wd['start']; we = wd['end']
                    s1 = sim_to_screen(sim_mx + ws[0], sim_my + ws[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    s2 = sim_to_screen(sim_mx + we[0], sim_my + we[1], zoom, pan_x, pan_y, sim.world_size, layout)
                    pygame.draw.line(screen, (100, 255, 100), s1, s2, 2)

        screen.set_clip(None)
        # Panels
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, (0, TOP_MENU_H, layout['LEFT_W'], layout['H']))
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (layout['LEFT_W'], TOP_MENU_H), (layout['LEFT_W'], layout['H']))
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, (layout['RIGHT_X'], TOP_MENU_H, layout['RIGHT_W'], layout['H']))
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (layout['RIGHT_X'], TOP_MENU_H), (layout['RIGHT_X'], layout['H']))
        pygame.draw.rect(screen, (40, 40, 45), (layout['RIGHT_X'], TOP_MENU_H, layout['RIGHT_W'], 90))
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (layout['RIGHT_X'], TOP_MENU_H + 90), (layout['W'], TOP_MENU_H + 90))
        
        metric_x = layout['RIGHT_X'] + 15
        if app_mode == MODE_SIM:
            curr_t = calculate_current_temp(sim.vel_x, sim.vel_y, sim.count, config.ATOM_MASS)
            screen.blit(big_font.render(f"Particles: {sim.count}", True, (255, 255, 255)), (metric_x, TOP_MENU_H + 10))
            screen.blit(font.render(f"Pairs: {sim.pair_count} | T: {curr_t:.3f}", True, (180, 180, 180)), (metric_x, TOP_MENU_H + 40))
            screen.blit(font.render(f"SPS: {int(sim.sps)}  FPS: {clock.get_fps():.1f}", True, (100, 255, 100)), (metric_x, TOP_MENU_H + 60))
        else:
            screen.blit(big_font.render("ASSET EDITOR", True, (255, 200, 100)), (metric_x, TOP_MENU_H + 10))
            screen.blit(font.render(f"Walls: {len(sim.walls)}", True, (200, 200, 200)), (metric_x, TOP_MENU_H + 40))
            screen.blit(font.render(f"Constrs: {len(sim.constraints)}", True, (150, 150, 150)), (metric_x, TOP_MENU_H + 60))

        for el in current_ui_list: el.draw(screen, font)
        if app_mode == MODE_SIM:
            btn_tool_brush.draw(screen, font); btn_tool_line.draw(screen, font)
            screen.blit(font.render("World Size:", True, (200, 200, 200)), (layout['RIGHT_X'] + 15, input_world.rect.y + 4))
            input_world.draw(screen, font)
        if app_mode == MODE_EDITOR:
            btn_tool_line.draw(screen, font); btn_tool_ref.draw(screen, font); btn_tool_point.draw(screen, font)

        if time.time() - status_time < 3.0:
            status_surf = font.render(status_msg, True, (100, 255, 100))
            pygame.draw.rect(screen, (30,30,30), (layout['MID_X'] + 10, TOP_MENU_H + 10, status_surf.get_width()+10, 25), border_radius=5)
            screen.blit(status_surf, (layout['MID_X'] + 15, TOP_MENU_H + 15))
        
        menu_bar.draw(screen, font)
        if context_menu: context_menu.draw(screen, font)
        if prop_dialog: prop_dialog.draw(screen, font)
        if rot_dialog: rot_dialog.draw(screen, font)
        pygame.display.flip()
        clock.tick(0)
        
    if root_tk: root_tk.destroy() 
    pygame.quit()

if __name__ == "__main__":
    main()