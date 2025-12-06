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

# --- Layout Constants ---
TOP_MENU_H = 30

# --- Modes ---
MODE_SIM = 0
MODE_EDITOR = 1

# --- Helper Functions for Group Movement ---
def get_connected_group(sim, start_wall_idx):
    """
    Returns a set of wall indices that are topologically connected 
    to the start_wall_idx via COINCIDENT constraints.
    """
    group = {start_wall_idx}
    queue = [start_wall_idx]
    
    # Build an adjacency map for faster lookup 
    adjacency = {}
    for c in sim.constraints:
        if c['type'] == 'COINCIDENT':
            # c['indices'] is typically [(w1, p1), (w2, p2)]
            idx_list = c['indices']
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
    """Checks if any wall in the group has an anchored point."""
    for idx in group_indices:
        w = sim.walls[idx]
        anchors = w.get('anchored', [False, False])
        if anchors[0] or anchors[1]:
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

def get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, world_size, layout, anchor_pos=None, exclude_wall_idx=-1):
    sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, world_size, layout)
    final_x, final_y = sim_x, sim_y
    is_snapped_to_vertex = False
    
    if pygame.key.get_mods() & pygame.KMOD_CTRL:
        snap_threshold_px = 15 
        best_dist = float('inf')
        for i, w in enumerate(sim.walls):
            if i == exclude_wall_idx: continue
            points = [w['start'], w['end']]
            for pt in points:
                sx, sy = sim_to_screen(pt[0], pt[1], zoom, pan_x, pan_y, world_size, layout)
                dist = math.hypot(mx - sx, my - sy)
                if dist < snap_threshold_px and dist < best_dist:
                    best_dist = dist
                    final_x, final_y = pt
                    is_snapped_to_vertex = True

    if (pygame.key.get_mods() & pygame.KMOD_SHIFT) and anchor_pos is not None and not is_snapped_to_vertex:
        dx = final_x - anchor_pos[0]
        dy = final_y - anchor_pos[1]
        if abs(dx) > abs(dy): final_y = anchor_pos[1] 
        else: final_x = anchor_pos[0] 

    return final_x, final_y

def calculate_current_temp(vel_x, vel_y, count, mass):
    if count == 0: return 0.0
    vx = vel_x[:count]
    vy = vel_y[:count]
    ke_total = 0.5 * mass * np.sum(vx**2 + vy**2)
    return ke_total / count

# --- Helpers ---
def convert_state_to_serializable(state):
    serializable_state = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray): serializable_state[k] = v.tolist()
        else: serializable_state[k] = v
    return serializable_state

def convert_state_from_serializable(state):
    restored_state = {}
    for k, v in state.items():
        if k in ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'is_static', 'atom_sigma', 'atom_eps_sqrt']:
            dtype = np.float32
            if k == 'is_static': dtype = np.int32
            restored_state[k] = np.array(v, dtype=dtype)
        else: restored_state[k] = v
    return restored_state

def save_file(sim, filename, settings=None):
    if not filename: return "Cancelled"
    state = {
        'count': sim.count,
        'pos_x': sim.pos_x[:sim.count],
        'pos_y': sim.pos_y[:sim.count],
        'vel_x': sim.vel_x[:sim.count],
        'vel_y': sim.vel_y[:sim.count],
        'is_static': sim.is_static[:sim.count],
        'atom_sigma': sim.atom_sigma[:sim.count],
        'atom_eps_sqrt': sim.atom_eps_sqrt[:sim.count],
        'walls': sim.walls,
        'constraints': sim.constraints,
        'world_size': sim.world_size,
        'settings': settings if settings else {}
    }
    data = convert_state_to_serializable(state)
    try:
        with open(filename, 'w') as f: json.dump(data, f)
        return f"Saved to {os.path.basename(filename)}"
    except Exception as e: return f"Error: {e}"

def load_file(sim, filename):
    if not filename: return False, "Cancelled", {}
    try:
        with open(filename, 'r') as f: data = json.load(f)
        state = convert_state_from_serializable(data)
        sim.restore_state(state)
        return True, f"Loaded {os.path.basename(filename)}", data.get('settings', {})
    except Exception as e: return False, f"Error: {e}", {}

def save_asset_file(sim, filename):
    if not filename: return "Cancelled"
    asset_data = sim.export_asset_data()
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
        return data.get('data', {}) # Returns dict with 'walls' and 'constraints'
    except Exception as e: return None

def main():
    try:
        root_tk = Tk()
        root_tk.withdraw()
    except:
        root_tk = None

    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Fast MD - Panel Layout")
    
    try:
        font = pygame.font.SysFont("segoeui", 15)
        big_font = pygame.font.SysFont("segoeui", 22)
    except:
        font = pygame.font.SysFont("arial", 15)
        big_font = pygame.font.SysFont("arial", 22)
        
    clock = pygame.time.Clock()
    
    sim = Simulation()
    app_mode = MODE_SIM
    
    layout = {
        'W': config.WINDOW_WIDTH,
        'H': config.WINDOW_HEIGHT,
        'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
        'RIGHT_W': config.PANEL_RIGHT_WIDTH,
        'RIGHT_X': config.WINDOW_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_X': config.PANEL_LEFT_WIDTH,
        'MID_W': config.WINDOW_WIDTH - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
        'MID_H': config.WINDOW_HEIGHT - TOP_MENU_H
    }

    menu_bar = MenuBar(layout['W'], TOP_MENU_H)
    menu_bar.items["File"] = ["New Simulation", "Open...", "Save", "Save As...", "---", "Create New Asset", "Add Existing Asset"] 
    
    current_filepath = None
    status_msg = ""
    status_time = 0
    
    # -- Elements --
    lp_margin = 10
    lp_curr_y = TOP_MENU_H + 20
    
    btn_play = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Play/Pause", active=False, color_active=(60, 120, 60), color_inactive=(180, 60, 60))
    lp_curr_y += 50
    btn_clear = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Clear", active=False, toggle=False, color_inactive=(80, 80, 80))
    lp_curr_y += 50
    btn_reset = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Reset", active=False, toggle=False, color_inactive=(80, 80, 80))
    lp_curr_y += 50
    btn_undo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Undo", active=False, toggle=False)
    lp_curr_y += 45
    btn_redo = Button(layout['LEFT_X'] + lp_margin, lp_curr_y, layout['LEFT_W']-20, 35, "Redo", active=False, toggle=False)
    
    rp_start_x = layout['RIGHT_X'] + 15
    rp_width = layout['RIGHT_W'] - 30
    rp_curr_y = TOP_MENU_H + 120
    
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
    btn_boundaries = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Bounds", active=False)
    rp_curr_y += 40
    btn_tool_brush = Button(rp_start_x, rp_curr_y, btn_w, 30, "Brush", active=True, toggle=False)
    btn_tool_wall = Button(rp_start_x + btn_w + 10, rp_curr_y, btn_w, 30, "Wall", active=False, toggle=False)
    rp_curr_y += 40
    slider_brush_size = SmartSlider(rp_start_x, rp_curr_y, rp_width, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5); rp_curr_y+=60
    lbl_resize = font.render("World Size:", True, (200, 200, 200))
    input_world = InputField(rp_start_x + 80, rp_curr_y, 60, 25, str(config.DEFAULT_WORLD_SIZE))
    btn_resize = Button(rp_start_x + 150, rp_curr_y, rp_width - 150, 25, "Resize & Restart", active=False, toggle=False)
    
    # Asset Editor UI
    ae_curr_y = TOP_MENU_H + 40
    btn_ae_save = Button(rp_start_x, ae_curr_y, rp_width, 40, "Save Asset", active=False, toggle=False, color_inactive=(50, 120, 50)); ae_curr_y+=50
    btn_ae_discard = Button(rp_start_x, ae_curr_y, rp_width, 40, "Discard & Exit", active=False, toggle=False, color_inactive=(150, 50, 50)); ae_curr_y+=50
    
    # Constraint Buttons
    c_btn_w = (rp_width - 10) // 2
    btn_const_coincident = Button(rp_start_x, ae_curr_y, rp_width, 30, "Coincident (Pt-Pt)", toggle=False); ae_curr_y+=35
    btn_const_length = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Fix Length", toggle=False)
    btn_const_equal = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Equal Len", toggle=False); ae_curr_y+=35
    btn_const_parallel = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Parallel", toggle=False)
    btn_const_perp = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Perpendic", toggle=False); ae_curr_y+=35
    
    # New Constraint Buttons
    btn_const_horiz = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Horizontal", toggle=False)
    btn_const_vert = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Vertical", toggle=False); ae_curr_y+=35

    ui_sim_elements = [btn_play, btn_clear, btn_reset, btn_undo, btn_redo,
                       slider_gravity, slider_temp, slider_damping, slider_dt, 
                       slider_sigma, slider_epsilon, slider_M, slider_skin,
                       btn_thermostat, btn_boundaries, slider_brush_size, btn_resize]
    
    ui_editor_elements = [btn_tool_wall, btn_ae_save, btn_ae_discard, btn_undo, btn_redo, btn_clear,
                          btn_const_coincident, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp,
                          btn_const_horiz, btn_const_vert]
    
    right_panel_elements = [
        slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin,
        btn_thermostat, btn_boundaries, btn_tool_brush, btn_tool_wall, slider_brush_size, input_world, btn_resize,
        btn_ae_save, btn_ae_discard, btn_const_coincident, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp,
        btn_const_horiz, btn_const_vert
    ]

    # --- APP STATE ---
    current_tool = 0 
    zoom = 1.0; pan_x = 0.0; pan_y = 0.0
    is_panning = False; last_mouse_pos = (0, 0)
    is_painting = False; is_erasing = False
    wall_mode = None; wall_idx = -1; wall_pt = -1
    
    context_menu = None; prop_dialog = None; rot_dialog = None; context_wall_idx = -1
    context_pt_idx = None # Added for anchor toggle
    placing_asset_data = None; sim_backup_state = None
    
    # Selection State
    selected_walls = set() # Set of indices
    selected_points = set() # Set of (wall_idx, pt_idx) tuples
    
    # Pending Constraint State
    pending_constraint = None # 'LENGTH', 'EQUAL', etc.
    pending_targets_walls = []
    pending_targets_points = []
    
    # Drag State
    drag_start_length = None
    temp_constraint_active = False
    current_group_indices = [] # For group movement

    def enter_editor_mode():
        nonlocal app_mode, zoom, pan_x, pan_y, status_msg, status_time, sim_backup_state
        sim_backup_state = {
            'count': sim.count,
            'pos_x': np.copy(sim.pos_x[:sim.count]),
            'pos_y': np.copy(sim.pos_y[:sim.count]),
            'vel_x': np.copy(sim.vel_x[:sim.count]),
            'vel_y': np.copy(sim.vel_y[:sim.count]),
            'is_static': np.copy(sim.is_static[:sim.count]),
            'atom_sigma': np.copy(sim.atom_sigma[:sim.count]),
            'atom_eps_sqrt': np.copy(sim.atom_eps_sqrt[:sim.count]),
            'walls': copy.deepcopy(sim.walls),
            'constraints': copy.deepcopy(sim.constraints),
            'world_size': sim.world_size
        }
        sim.clear_particles(snapshot=False)
        sim.world_size = 30.0
        sim.walls = []
        sim.constraints = []
        app_mode = MODE_EDITOR
        btn_tool_wall.active = True
        btn_tool_brush.active = False
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
        
        # 1. Check if we have enough selected items to apply immediately
        applied = False
        if ctype == 'LENGTH' and len(selected_walls) == 1:
            val = simpledialog.askfloat("Length", "Enter length:")
            if val: sim.add_constraint('LENGTH', list(selected_walls), val); applied = True
        elif ctype in ['EQUAL', 'PARALLEL', 'PERPENDICULAR'] and len(selected_walls) == 2:
            sim.add_constraint(ctype, list(selected_walls)); applied = True
        elif ctype in ['HORIZONTAL', 'VERTICAL'] and len(selected_walls) >= 1:
            # Apply to all selected
            for w_idx in selected_walls: sim.add_constraint(ctype, [w_idx])
            applied = True
        elif ctype == 'COINCIDENT' and len(selected_points) == 2:
            sim.add_constraint('COINCIDENT', list(selected_points)); applied = True
            
        if applied:
            status_msg = f"Applied {ctype}"; status_time = time.time()
            selected_walls.clear(); selected_points.clear()
            pending_constraint = None
            reset_constraint_buttons()
        else:
            # Enter pending mode
            pending_constraint = ctype
            pending_targets_walls.clear()
            pending_targets_points.clear()
            # If items are already selected, add them to pending targets
            if selected_walls: pending_targets_walls.extend(list(selected_walls))
            if selected_points: pending_targets_points.extend(list(selected_points))
            
            selected_walls.clear(); selected_points.clear()
            status_msg = f"Select targets for {ctype}..."; status_time = time.time()
            
            # Highlight the button
            reset_constraint_buttons()
            if ctype == 'LENGTH': btn_const_length.active = True
            elif ctype == 'EQUAL': btn_const_equal.active = True
            elif ctype == 'PARALLEL': btn_const_parallel.active = True
            elif ctype == 'PERPENDICULAR': btn_const_perp.active = True
            elif ctype == 'COINCIDENT': btn_const_coincident.active = True
            elif ctype == 'HORIZONTAL': btn_const_horiz.active = True
            elif ctype == 'VERTICAL': btn_const_vert.active = True

    def reset_constraint_buttons():
        btn_const_length.active = False
        btn_const_equal.active = False
        btn_const_parallel.active = False
        btn_const_perp.active = False
        btn_const_coincident.active = False
        btn_const_horiz.active = False
        btn_const_vert.active = False

    def handle_pending_constraint_click(wall_idx=None, pt_idx=None):
        nonlocal pending_constraint, status_msg, status_time
        
        if pending_constraint in ['LENGTH', 'HORIZONTAL', 'VERTICAL']:
            if wall_idx is not None:
                if pending_constraint == 'LENGTH':
                    val = simpledialog.askfloat("Length", "Enter length:")
                    if val: sim.add_constraint('LENGTH', [wall_idx], val)
                else:
                    sim.add_constraint(pending_constraint, [wall_idx])
                
                status_msg = f"Applied {pending_constraint}"; status_time = time.time()
                pending_constraint = None
                reset_constraint_buttons()
        
        elif pending_constraint in ['EQUAL', 'PARALLEL', 'PERPENDICULAR']:
            if wall_idx is not None:
                if wall_idx not in pending_targets_walls:
                    pending_targets_walls.append(wall_idx)
                    status_msg = f"Selected 1/2 for {pending_constraint}"
                    
                    if len(pending_targets_walls) == 2:
                        sim.add_constraint(pending_constraint, pending_targets_walls)
                        status_msg = f"Applied {pending_constraint}"; status_time = time.time()
                        pending_constraint = None
                        reset_constraint_buttons()
        
        elif pending_constraint == 'COINCIDENT':
            if pt_idx is not None: # pt_idx is (wall, end)
                if pt_idx not in pending_targets_points:
                    pending_targets_points.append(pt_idx)
                    status_msg = f"Selected 1/2 Points"
                    
                    if len(pending_targets_points) == 2:
                        sim.add_constraint('COINCIDENT', pending_targets_points)
                        status_msg = "Applied Coincident"; status_time = time.time()
                        pending_constraint = None
                        reset_constraint_buttons()

    running = True
    try:
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
                    if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        sim.undo(); status_msg = "Undo"; status_time = time.time()
                    elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                        sim.redo(); status_msg = "Redo"; status_time = time.time()
                    if event.key == pygame.K_ESCAPE:
                        if placing_asset_data: placing_asset_data = None; status_msg = "Cancelled"; status_time = time.time()
                        elif pending_constraint:
                            pending_constraint = None
                            reset_constraint_buttons()
                            status_msg = "Constraint Cancelled"; status_time = time.time()
                        else:
                            selected_walls.clear(); selected_points.clear() # Clear selection
                
                menu_clicked = menu_bar.handle_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and menu_bar.active_menu:
                    if menu_bar.dropdown_rect and menu_bar.dropdown_rect.collidepoint(event.pos):
                        rel_y = event.pos[1] - menu_bar.dropdown_rect.y - 5
                        idx = rel_y // 30
                        opts = menu_bar.items[menu_bar.active_menu]
                        if 0 <= idx < len(opts):
                            selection = opts[idx]
                            menu_bar.active_menu = None 
                            if selection == "New Simulation":
                                sim.reset_simulation(); input_world.set_value(config.DEFAULT_WORLD_SIZE)
                            elif selection == "Create New Asset":
                                if app_mode == MODE_SIM: enter_editor_mode()
                            elif selection == "Add Existing Asset":
                                if app_mode == MODE_SIM and root_tk:
                                    f = filedialog.askopenfilename(filetypes=[("Asset Files", "*.json")])
                                    if f:
                                        data = load_asset_file(f)
                                        if data:
                                            # data is {walls: [], constraints: []}
                                            placing_asset_data = data
                                            status_msg = "Place Asset"; status_time = time.time()
                            elif root_tk:
                                if selection == "Save As..." or (selection == "Save" and not current_filepath):
                                    f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
                                    if f: current_filepath = f; msg = save_file(sim, f); status_msg = msg; status_time = time.time()
                                elif selection == "Save" and current_filepath:
                                    msg = save_file(sim, current_filepath); status_msg = msg; status_time = time.time()
                                elif selection == "Open...":
                                    f = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
                                    if f: current_filepath = f; success, msg, lset = load_file(sim, f); status_msg = msg; status_time = time.time()
                                    if success:
                                        input_world.set_value(sim.world_size); zoom=1.0; pan_x=0; pan_y=0
                        continue 
                    elif not menu_clicked: menu_bar.active_menu = None
                if menu_clicked: continue

                if context_menu:
                    if context_menu.handle_event(event):
                        action = context_menu.action
                        if action == "Delete": 
                            sim.remove_wall(context_wall_idx)
                            selected_walls.clear(); selected_points.clear()
                            context_menu = None
                        elif action == "Properties":
                            prop_dialog = PropertiesDialog(layout['W']//2, layout['H']//2, sim.walls[context_wall_idx]); context_menu = None
                        elif action == "Set Rotation...":
                            rot_dialog = RotationDialog(layout['W']//2, layout['H']//2, sim.walls[context_wall_idx].get('anim')); context_menu = None
                        elif action == "Anchor": # NEW
                            # Toggle Anchor
                            sim.toggle_anchor(context_wall_idx, context_pt_idx)
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
                if btn_tool_wall.handle_event(event):
                    current_tool = 1; btn_tool_wall.active = True; btn_tool_brush.active = False; ui_captured = True
                if app_mode == MODE_SIM and btn_tool_brush.handle_event(event):
                    current_tool = 0; btn_tool_brush.active = True; btn_tool_wall.active = False; ui_captured = True
                
                # Constraint Logic
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
                    elif event.button == 3:
                        mx, my = event.pos
                        if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                            if placing_asset_data: placing_asset_data = None
                            elif pending_constraint:
                                pending_constraint = None; reset_constraint_buttons(); status_msg = "Cancelled"; status_time = time.time()
                            else:
                                sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                                rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * zoom)
                                hit_pt = None
                                # Check for Point first
                                for i, w in enumerate(sim.walls):
                                    if math.hypot(w['start'][0]-sim_x, w['start'][1]-sim_y) < rad_sim: hit_pt=(i,0); break
                                    if math.hypot(w['end'][0]-sim_x, w['end'][1]-sim_y) < rad_sim: hit_pt=(i,1); break
                                
                                if hit_pt and app_mode == MODE_EDITOR:
                                    context_wall_idx = hit_pt[0]
                                    context_pt_idx = hit_pt[1]
                                    context_menu = ContextMenu(mx, my, ["Anchor"])
                                else:
                                    # Fallback to Wall
                                    hit = -1
                                    for i, w in enumerate(sim.walls):
                                        if math.hypot(w['start'][0]-sim_x, w['start'][1]-sim_y) < rad_sim or math.hypot(w['end'][0]-sim_x, w['end'][1]-sim_y) < rad_sim:
                                            hit = i; break
                                    if hit != -1: context_menu = ContextMenu(mx, my, ["Properties", "Set Rotation...", "Delete"]); context_wall_idx = hit
                                    else: is_erasing = True; sim.snapshot()
                    elif event.button == 1:
                        mx, my = event.pos
                        if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                            if placing_asset_data:
                                sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                                sim.place_asset(placing_asset_data, sim_x, sim_y)
                            elif current_tool == 0 and app_mode == MODE_SIM: is_painting = True; sim.snapshot()
                            elif current_tool == 1 or app_mode == MODE_EDITOR:
                                # SELECTION & WALL LOGIC
                                sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                                rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * zoom)
                                
                                # Check for Point Click
                                hit_pt = None
                                for i, w in enumerate(sim.walls):
                                    if math.hypot(w['start'][0]-sim_x, w['start'][1]-sim_y) < rad_sim: hit_pt=(i,0); break
                                    if math.hypot(w['end'][0]-sim_x, w['end'][1]-sim_y) < rad_sim: hit_pt=(i,1); break
                                
                                if hit_pt:
                                    if pending_constraint:
                                        handle_pending_constraint_click(pt_idx=hit_pt)
                                    else:
                                        # Point Click Logic
                                        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                            if hit_pt not in selected_points: # If not adding to selection, clear others
                                                selected_walls.clear(); selected_points.clear()
                                        
                                        if hit_pt in selected_points: selected_points.remove(hit_pt)
                                        else: selected_points.add(hit_pt)
                                        
                                        wall_mode = 'EDIT'; wall_idx = hit_pt[0]; wall_pt = hit_pt[1]
                                        sim.snapshot()
                                else:
                                    # Check for Wall Click (Line)
                                    hit_wall = -1
                                    for i, w in enumerate(sim.walls):
                                        # Dist point to line segment
                                        p1=np.array(w['start']); p2=np.array(w['end']); p3=np.array([sim_x, sim_y])
                                        d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                                        if len_sq == 0: dist = np.linalg.norm(p3-p1)
                                        else:
                                            t = max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))
                                            proj = p1 + t*d_vec
                                            dist = np.linalg.norm(p3-proj)
                                        if dist < rad_sim: hit_wall = i; break
                                    
                                    if hit_wall != -1:
                                        if pending_constraint:
                                            handle_pending_constraint_click(wall_idx=hit_wall)
                                        else:
                                            if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                                if hit_wall not in selected_walls:
                                                    selected_walls.clear(); selected_points.clear()
                                                    selected_walls.add(hit_wall) # Just select
                                            else:
                                                if hit_wall in selected_walls: selected_walls.remove(hit_wall)
                                                else: selected_walls.add(hit_wall)
                                            
                                            # NEW: Group Move Logic
                                            target_group = get_connected_group(sim, hit_wall)
                                            
                                            if not is_group_anchored(sim, target_group):
                                                wall_mode = 'MOVE_GROUP'
                                                current_group_indices = list(target_group)
                                                sim.snapshot()
                                                last_mouse_pos = event.pos
                                            else:
                                                # Standard Single Wall Drag (PBD Deformation)
                                                wall_mode = 'MOVE_WALL'; wall_idx = hit_wall
                                                sim.snapshot()
                                                last_mouse_pos = event.pos # Capture start position
                                                
                                                # Capture start length for Rigid Drag Check
                                                w = sim.walls[hit_wall]
                                                drag_start_length = math.hypot(w['end'][0]-w['start'][0], w['end'][1]-w['start'][1])
                                                
                                                # Add temporary length constraint for rigidity
                                                if app_mode == MODE_EDITOR:
                                                    sim.constraints.append({
                                                        'type': 'LENGTH', 
                                                        'indices': [hit_wall], 
                                                        'value': drag_start_length,
                                                        'temp': True
                                                    })
                                                    temp_constraint_active = True
                                    else:
                                        # Click on empty space
                                        if not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                                            selected_walls.clear(); selected_points.clear()
                                        
                                        if not pending_constraint:
                                            wall_mode = 'NEW'; sim.snapshot()
                                            start_x, start_y = get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, sim.world_size, layout)
                                            sim.add_wall((start_x, start_y), (start_x, start_y))
                                            wall_idx = len(sim.walls)-1; wall_pt = 1

                elif event.type == pygame.MOUSEBUTTONUP:
                    is_panning = False; is_painting = False; is_erasing = False
                    wall_mode = None; wall_idx = -1
                    current_group_indices = [] # Clear group
                    
                    # Remove temporary constraint
                    if temp_constraint_active:
                        sim.constraints = [c for c in sim.constraints if not c.get('temp')]
                        temp_constraint_active = False
                    
                    # Solve constraints after move
                    if app_mode == MODE_EDITOR: sim.apply_constraints()

                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if is_panning: pan_x += mx - last_mouse_pos[0]; pan_y += my - last_mouse_pos[1]; last_mouse_pos = (mx, my)
                    
                    if wall_mode is not None:
                        if wall_mode == 'EDIT' or wall_mode == 'NEW':
                            if wall_idx < len(sim.walls):
                                w = sim.walls[wall_idx]
                                anchor = w['end'] if wall_pt == 0 else w['start']
                                dest_x, dest_y = get_snapped_pos(mx, my, sim, zoom, pan_x, pan_y, sim.world_size, layout, anchor, wall_idx)
                                if wall_pt == 0: sim.update_wall(wall_idx, (dest_x, dest_y), w['end'])
                                else: sim.update_wall(wall_idx, w['start'], (dest_x, dest_y))
                                if app_mode == MODE_EDITOR: sim.apply_constraints()
                        
                        elif wall_mode == 'MOVE_WALL':
                             if wall_idx < len(sim.walls):
                                # Use sim coordinates difference for 1:1 movement
                                curr_sim_x, curr_sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                                prev_sim_x, prev_sim_y = screen_to_sim(last_mouse_pos[0], last_mouse_pos[1], zoom, pan_x, pan_y, sim.world_size, layout)
                                
                                dx = curr_sim_x - prev_sim_x
                                dy = curr_sim_y - prev_sim_y
                                
                                last_mouse_pos = (mx, my) # Update for next delta
                                
                                w = sim.walls[wall_idx]
                                anchors = w.get('anchored', [False, False])
                                
                                # RESPECT ANCHORS: Only move if not anchored
                                # If anchored, delta for that point is 0
                                d_start_x = 0 if anchors[0] else dx
                                d_start_y = 0 if anchors[0] else dy
                                d_end_x = 0 if anchors[1] else dx
                                d_end_y = 0 if anchors[1] else dy
                                
                                new_s = (w['start'][0] + d_start_x, w['start'][1] + d_start_y)
                                new_e = (w['end'][0] + d_end_x, w['end'][1] + d_end_y)
                                
                                sim.update_wall(wall_idx, new_s, new_e)
                                
                                # Just apply constraints; the temporary LENGTH constraint handles rigidity
                                if app_mode == MODE_EDITOR: 
                                    sim.apply_constraints()
                        
                        elif wall_mode == 'MOVE_GROUP':
                            curr_sim_x, curr_sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                            prev_sim_x, prev_sim_y = screen_to_sim(last_mouse_pos[0], last_mouse_pos[1], zoom, pan_x, pan_y, sim.world_size, layout)
                            
                            dx = curr_sim_x - prev_sim_x
                            dy = curr_sim_y - prev_sim_y
                            
                            last_mouse_pos = (mx, my)
                            
                            for w_i in current_group_indices:
                                if w_i < len(sim.walls):
                                    w = sim.walls[w_i]
                                    # Group move is only allowed for unanchored groups, so we don't check per-point anchors here
                                    # (The check is done in MOUSEBUTTONDOWN)
                                    new_s = (w['start'][0] + dx, w['start'][1] + dy)
                                    new_e = (w['end'][0] + dx, w['end'][1] + dy)
                                    sim.update_wall(w_i, new_s, new_e)

                            if app_mode == MODE_EDITOR: 
                                sim.apply_constraints()

            # --- UPDATE ---
            if not prop_dialog and not rot_dialog:
                if app_mode == MODE_SIM:
                    sim.paused = not btn_play.active
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
                else: sim.paused = True; steps = 0
                mx, my = pygame.mouse.get_pos()
                if is_painting or is_erasing:
                     if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                        sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                        if is_painting: sim.add_particles_brush(sim_x, sim_y, slider_brush_size.val)
                        elif is_erasing: sim.delete_particles_brush(sim_x, sim_y, slider_brush_size.val)
                if not sim.paused: sim.step(steps)
                
            # --- RENDER ---
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
            
            # --- RENDER WALLS (Layered Approach) ---
            # 1. Draw Lines (Walls)
            for i, w in enumerate(sim.walls):
                s1 = sim_to_screen(w['start'][0], w['start'][1], zoom, pan_x, pan_y, sim.world_size, layout)
                s2 = sim_to_screen(w['end'][0], w['end'][1], zoom, pan_x, pan_y, sim.world_size, layout)
                
                color = (255, 255, 255)
                if i in selected_walls: color = (255, 200, 50) # Orange highlight
                if pending_constraint and i in pending_targets_walls: color = (100, 255, 100) # Green pending
                
                if app_mode == MODE_EDITOR: 
                    pygame.draw.line(screen, color, s1, s2, 2 if (i in selected_walls or (pending_constraint and i in pending_targets_walls)) else 1)

            # 2. Draw Points & Collect Anchors
            anchored_points_draw_list = []
            
            for i, w in enumerate(sim.walls):
                s1 = sim_to_screen(w['start'][0], w['start'][1], zoom, pan_x, pan_y, sim.world_size, layout)
                s2 = sim_to_screen(w['end'][0], w['end'][1], zoom, pan_x, pan_y, sim.world_size, layout)
                
                # Standard Point (White)
                pygame.draw.rect(screen, (255, 255, 255), (s1[0]-3, s1[1]-3, 6, 6))
                pygame.draw.rect(screen, (255, 255, 255), (s2[0]-3, s2[1]-3, 6, 6))
                
                # Highlight Selected Points
                if (i, 0) in selected_points: pygame.draw.rect(screen, (0, 255, 255), (s1[0]-4, s1[1]-4, 8, 8), 2)
                if (i, 1) in selected_points: pygame.draw.rect(screen, (0, 255, 255), (s2[0]-4, s2[1]-4, 8, 8), 2)
                
                # Highlight Pending Points
                if pending_constraint and (i, 0) in pending_targets_points: pygame.draw.rect(screen, (100, 255, 100), (s1[0]-4, s1[1]-4, 8, 8), 2)
                if pending_constraint and (i, 1) in pending_targets_points: pygame.draw.rect(screen, (100, 255, 100), (s2[0]-4, s2[1]-4, 8, 8), 2)
                
                # Collect Anchors for late rendering
                anchors = w.get('anchored', [False, False])
                if anchors[0]: anchored_points_draw_list.append(s1)
                if anchors[1]: anchored_points_draw_list.append(s2)

            # 3. Draw Anchors on Top
            for pt in anchored_points_draw_list:
                # Red, slightly larger to ensure it covers the white/selection box if coincident
                pygame.draw.rect(screen, (255, 50, 50), (pt[0]-4, pt[1]-4, 8, 8))

            if placing_asset_data:
                mx, my = pygame.mouse.get_pos()
                if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                    sim_mx, sim_my = screen_to_sim(mx, my, zoom, pan_x, pan_y, sim.world_size, layout)
                    walls_to_draw = placing_asset_data.get('walls', [])
                    for w in walls_to_draw:
                        wx1 = sim_mx + w['start'][0]; wy1 = sim_my + w['start'][1]
                        wx2 = sim_mx + w['end'][0]; wy2 = sim_my + w['end'][1]
                        s1 = sim_to_screen(wx1, wy1, zoom, pan_x, pan_y, sim.world_size, layout)
                        s2 = sim_to_screen(wx2, wy2, zoom, pan_x, pan_y, sim.world_size, layout)
                        pygame.draw.line(screen, (100, 255, 100), s1, s2, 2)
                        pygame.draw.rect(screen, (100, 255, 100), (s1[0]-2, s1[1]-2, 4, 4))
                        pygame.draw.rect(screen, (100, 255, 100), (s2[0]-2, s2[1]-2, 4, 4))

            screen.set_clip(None)
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
                btn_tool_brush.draw(screen, font); btn_tool_wall.draw(screen, font)
                screen.blit(lbl_resize, (layout['RIGHT_X'] + 15, input_world.rect.y + 4)); input_world.draw(screen, font)
            if time.time() - status_time < 3.0:
                status_surf = font.render(status_msg, True, (100, 255, 100))
                pygame.draw.rect(screen, (30,30,30), (layout['MID_X'] + 10, TOP_MENU_H + 10, status_surf.get_width()+10, 25), border_radius=5)
                screen.blit(status_surf, (layout['MID_X'] + 15, TOP_MENU_H + 15))
            menu_bar.draw(screen, font)
            if context_menu: context_menu.draw(screen, font)
            if prop_dialog: prop_dialog.draw(screen, font)
            if rot_dialog: rot_dialog.draw(screen, font)
            pygame.display.flip()
            clock.tick(60)
    finally:
        if root_tk: root_tk.destroy() 
    pygame.quit()

if __name__ == "__main__":
    main()