import pygame
import config
import utils
import file_io
import time
import math
import numpy as np
from tkinter import filedialog, Tk, simpledialog

# Modules
from simulation_state import Simulation
from ui_widgets import SmartSlider, Button, InputField, ContextMenu, PropertiesDialog, MenuBar, RotationDialog
from geometry import Line, Circle
from constraints import Length, EqualLength, Angle, Midpoint, Coincident, Collinear
from renderer import Renderer
from app_state import AppState, InteractionState
from tools import SelectTool, BrushTool, LineTool, RectTool, CircleTool, PointTool

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
    
    # --- Register Tools ---
    tool_registry = [
        (config.TOOL_SELECT, SelectTool, None), (config.TOOL_BRUSH, BrushTool, None),
        (config.TOOL_LINE, LineTool, None), (config.TOOL_RECT, RectTool, None),
        (config.TOOL_CIRCLE, CircleTool, None), (config.TOOL_POINT, PointTool, None),
        (config.TOOL_REF, LineTool, "Ref Line"), 
    ]
    for tid, cls, name in tool_registry:
        app.tools[tid] = cls(app, sim)
        if name: app.tools[tid].name = name
    app.change_tool(config.TOOL_BRUSH)

    # --- Layout ---
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
    
    # Tools Buttons
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
    
    # Editor Constraint Buttons
    ae_curr_y = config.TOP_MENU_H + 40
    btn_ae_save = Button(rp_start_x, ae_curr_y, rp_width, 40, "Save Geometry", active=False, toggle=False, color_inactive=(50, 120, 50)); ae_curr_y+=50
    btn_ae_discard = Button(rp_start_x, ae_curr_y, rp_width, 40, "Discard & Exit", active=False, toggle=False, color_inactive=(150, 50, 50)); ae_curr_y+=50
    
    c_btn_w = (rp_width - 10) // 2
    btn_const_coincident = Button(rp_start_x, ae_curr_y, rp_width, 30, "Coincident (Pt-Pt/Ln/Circ)", toggle=False); ae_curr_y+=35
    btn_const_collinear = Button(rp_start_x, ae_curr_y, rp_width, 30, "Collinear (Pt-Ln)", toggle=False); ae_curr_y+=35
    btn_const_midpoint = Button(rp_start_x, ae_curr_y, rp_width, 30, "Midpoint (Pt-Ln)", toggle=False); ae_curr_y+=35
    btn_const_length = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Fix Length", toggle=False)
    btn_const_equal = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Equal Len", toggle=False); ae_curr_y+=35
    btn_const_parallel = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Parallel", toggle=False)
    btn_const_perp = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Perpendic", toggle=False); ae_curr_y+=35
    btn_const_horiz = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Horizontal", toggle=False)
    btn_const_vert = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Vertical", toggle=False); ae_curr_y+=35
    
    btn_util_extend = Button(rp_start_x, ae_curr_y, rp_width, 30, "Extend Infinite", toggle=False); ae_curr_y+=35

    # --- UI Mappings & Definitions (REFACTORED) ---
    tool_btn_map = {
        btn_tool_brush: config.TOOL_BRUSH, btn_tool_select: config.TOOL_SELECT,
        btn_tool_line: config.TOOL_LINE, btn_tool_rect: config.TOOL_RECT,
        btn_tool_circle: config.TOOL_CIRCLE, btn_tool_point: config.TOOL_POINT, btn_tool_ref: config.TOOL_REF
    }
    constraint_btn_map = {
        btn_const_length: 'LENGTH', btn_const_equal: 'EQUAL', btn_const_parallel: 'PARALLEL',
        btn_const_perp: 'PERPENDICULAR', btn_const_coincident: 'COINCIDENT', btn_const_collinear: 'COLLINEAR',
        btn_const_midpoint: 'MIDPOINT', btn_const_horiz: 'HORIZONTAL', btn_const_vert: 'VERTICAL'
    }

    # Constraint Rules Table
    # Sig: list of {w: num_walls, p: num_points, type: allowed_types, msg: status_msg, factory: lambda sim, walls, points: Constraint}
    def get_l(s, w): return np.linalg.norm(s.walls[w].end - s.walls[w].start)
    constraint_defs = {
        'LENGTH':   [{'w':1, 'p':0, 't':Line, 'msg':"Select 1 Line", 'f': lambda s,w,p: Length(w[0], get_l(s, w[0]))}],
        'EQUAL':    [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: EqualLength(w[0], w[1])}],
        'PARALLEL': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PARALLEL', w[0], w[1])}],
        'PERPENDICULAR': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PERPENDICULAR', w[0], w[1])}],
        'HORIZONTAL': [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('HORIZONTAL', w[0]), 'multi':True}],
        'VERTICAL':   [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('VERTICAL', w[0]), 'multi':True}],
        'MIDPOINT':   [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Midpoint(p[0][0], p[0][1], w[0])}],
        'COLLINEAR':  [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Collinear(p[0][0], p[0][1], w[0])}],
        'COINCIDENT': [
            {'w':0, 'p':2, 't':None, 'msg':"Select 2 Points", 'f': lambda s,w,p: Coincident(p[0][0], p[0][1], p[1][0], p[1][1])},
            {'w':1, 'p':1, 't':(Line, Circle), 'msg':"Select Point & Entity", 'f': lambda s,w,p: Coincident(p[0][0], p[0][1], w[0], -1)}
        ]
    }

    # UI Lists
    ui_sim_elements = [
        btn_play, btn_clear, btn_reset, btn_undo, btn_redo, 
        slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, 
        btn_thermostat, btn_boundaries, btn_tool_brush, btn_tool_line, slider_brush_size, btn_resize
    ]
    ui_editor_elements = [
        btn_ae_save, btn_ae_discard, btn_undo, btn_redo, btn_clear, 
        btn_tool_select, btn_tool_line, btn_tool_rect, btn_tool_circle, btn_tool_point, btn_tool_ref,
        btn_const_coincident, btn_const_collinear, btn_const_midpoint, btn_const_length, btn_const_equal, 
        btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert, btn_util_extend
    ]

    # --- Helper Functions ---
    def change_tool(tool_id):
        app.change_tool(tool_id) 
        for btn, tid in tool_btn_map.items(): btn.active = (app.current_tool == app.tools.get(tid))

    def enter_geometry_mode():
        sim.snapshot(); app.sim_backup_state = sim.undo_stack.pop()
        sim.clear_particles(snapshot=False); sim.world_size = 30.0; sim.walls = []; sim.constraints = []
        app.mode = config.MODE_EDITOR; change_tool(config.TOOL_SELECT); app.zoom = 1.5; app.pan_x = 0; app.pan_y = 0
        app.set_status("Entered Geometry Editor")

    def exit_editor_mode(restore_state):
        sim.clear_particles(snapshot=False)
        if restore_state: sim.restore_state(restore_state)
        else: sim.world_size = config.DEFAULT_WORLD_SIZE
        app.mode = config.MODE_SIM; app.zoom = 1.0; app.pan_x = 0; app.pan_y = 0
        app.set_status("Returned to Simulation")

    def toggle_extend():
        if app.selected_walls:
            for idx in app.selected_walls:
                if idx < len(sim.walls) and isinstance(sim.walls[idx], Line): sim.walls[idx].infinite = not sim.walls[idx].infinite
            sim.rebuild_static_atoms(); app.set_status("Toggled Extend")

    def save_geo_dialog():
        if root_tk:
            f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Geometry Files", "*.json")])
            if f: app.set_status(file_io.save_geometry_file(sim, f)); exit_editor_mode(app.sim_backup_state)

    # Action Map for Standard Buttons
    ui_action_map = {
        btn_reset: lambda: sim.reset_simulation(),
        btn_clear: lambda: sim.clear_particles(),
        btn_undo: lambda: (sim.undo(), app.set_status("Undo")),
        btn_redo: lambda: (sim.redo(), app.set_status("Redo")),
        btn_resize: lambda: sim.resize_world(app.input_world.get_value(50.0)),
        btn_ae_discard: lambda: exit_editor_mode(app.sim_backup_state),
        btn_ae_save: save_geo_dialog,
        btn_util_extend: toggle_extend
    }

    def try_apply_constraint(ctype, wall_idxs, pt_idxs):
        """Generic constraint solver logic based on definitions"""
        rules = constraint_defs.get(ctype, [])
        for r in rules:
            if len(wall_idxs) == r['w'] and len(pt_idxs) == r['p']:
                # Validate Types
                valid = True
                if r.get('t'):
                    for w_idx in wall_idxs:
                        if not isinstance(sim.walls[w_idx], r['t']): valid = False; break
                if valid:
                    sim.add_constraint_object(r['f'](sim, wall_idxs, pt_idxs))
                    return True
        return False

    def trigger_constraint(ctype):
        # Update Visuals
        for btn, c_val in constraint_btn_map.items(): btn.active = (c_val == ctype)
        
        # Check Multi-select Apply (Horizontal/Vertical)
        is_multi = False
        if ctype in constraint_defs and constraint_defs[ctype][0].get('multi'):
            walls = list(app.selected_walls)
            if walls:
                count = 0
                for w_idx in walls:
                    if try_apply_constraint(ctype, [w_idx], []): count += 1
                if count > 0:
                    app.set_status(f"Applied {ctype} to {count} items")
                    app.selected_walls.clear(); app.selected_points.clear()
                    sim.apply_constraints(); is_multi = True
        
        if is_multi: return

        # Standard Apply
        walls = list(app.selected_walls); pts = list(app.selected_points)
        if try_apply_constraint(ctype, walls, pts):
            app.set_status(f"Applied {ctype}")
            app.selected_walls.clear(); app.selected_points.clear()
            app.pending_constraint = None
            for btn in constraint_btn_map.keys(): btn.active = False
            sim.apply_constraints()
        else:
            # Enter Pending Mode
            app.pending_constraint = ctype
            app.pending_targets_walls = list(app.selected_walls)
            app.pending_targets_points = list(app.selected_points)
            app.selected_walls.clear(); app.selected_points.clear()
            msg = constraint_defs[ctype][0]['msg'] if ctype in constraint_defs else "Select targets..."
            app.set_status(f"{ctype}: {msg}")

    def handle_pending_constraint_click(wall_idx=None, pt_idx=None):
        if not app.pending_constraint: return
        
        if wall_idx is not None and wall_idx not in app.pending_targets_walls:
            app.pending_targets_walls.append(wall_idx)
        if pt_idx is not None and pt_idx not in app.pending_targets_points:
            app.pending_targets_points.append(pt_idx)
            
        if try_apply_constraint(app.pending_constraint, app.pending_targets_walls, app.pending_targets_points):
            app.set_status(f"Applied {app.pending_constraint}")
            app.pending_constraint = None
            for btn in constraint_btn_map.keys(): btn.active = False
            sim.apply_constraints()
        else:
            # Update Status hint
            ctype = app.pending_constraint
            msg = constraint_defs[ctype][0]['msg']
            app.set_status(f"{ctype} ({len(app.pending_targets_walls)}W, {len(app.pending_targets_points)}P): {msg}")

    # --- Main Loop ---
    running = True
    context_menu = None; prop_dialog = None; rot_dialog = None
    context_wall_idx = -1; context_pt_idx = None; context_constraint_idx = -1 

    while running:
        current_ui_list = ui_sim_elements if app.mode == config.MODE_SIM else ui_editor_elements
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if app.current_tool: app.current_tool.cancel()
                    app.pending_constraint = None
                    app.selected_walls.clear(); app.selected_points.clear()
                    for btn in constraint_btn_map.keys(): btn.active = False
                    app.set_status("Cancelled")
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL): 
                    if app.current_tool: app.current_tool.cancel(); sim.undo(); app.set_status("Undo")
                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL): 
                    if app.current_tool: app.current_tool.cancel(); sim.redo(); app.set_status("Redo")
            
            for btn, tool_id in tool_btn_map.items():
                if btn.handle_event(event): change_tool(tool_id)
            
            # 3. Menus
            menu_clicked = menu_bar.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and menu_bar.active_menu and not menu_clicked:
                if menu_bar.dropdown_rect and menu_bar.dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - menu_bar.dropdown_rect.y - 5; idx = rel_y // 30
                    opts = menu_bar.items[menu_bar.active_menu]
                    if 0 <= idx < len(opts):
                        selection = opts[idx]; menu_bar.active_menu = None 
                        if selection == "New Simulation": sim.reset_simulation(); app.input_world.set_value(config.DEFAULT_WORLD_SIZE)
                        elif selection == "Create New Geometry" and app.mode == config.MODE_SIM: enter_geometry_mode()
                        elif selection == "Add Existing Geometry" and app.mode == config.MODE_SIM and root_tk:
                            f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.json")])
                            if f: 
                                data = file_io.load_geometry_file(f)
                                if data: app.placing_geo_data = data; app.set_status("Place Geometry")
                        elif root_tk:
                            if selection == "Save As..." or (selection == "Save" and not app.current_filepath):
                                f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
                                if f: app.current_filepath = f; app.set_status(file_io.save_file(sim, f))
                            elif selection == "Save" and app.current_filepath: app.set_status(file_io.save_file(sim, app.current_filepath))
                            elif selection == "Open...":
                                f = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
                                if f: 
                                    app.current_filepath = f
                                    success, msg, lset = file_io.load_file(sim, f)
                                    app.set_status(msg)
                                    if success: app.input_world.set_value(sim.world_size); app.zoom=1.0; app.pan_x=0; app.pan_y=0
                    else: menu_bar.active_menu = None

            # 4. Context/Dialogs
            if context_menu and context_menu.handle_event(event):
                action = context_menu.action
                if action == "Delete Constraint":
                    if 0 <= context_constraint_idx < len(sim.constraints):
                        sim.snapshot(); sim.constraints.pop(context_constraint_idx); sim.apply_constraints()
                    context_menu = None
                elif action == "Delete": 
                    sim.remove_wall(context_wall_idx); app.selected_walls.clear(); app.selected_points.clear(); context_menu = None
                elif action == "Properties":
                    w_props = sim.walls[context_wall_idx].to_dict()
                    prop_dialog = PropertiesDialog(layout['W']//2, layout['H']//2, w_props); context_menu = None
                elif action == "Set Rotation...":
                    rot_dialog = RotationDialog(layout['W']//2, layout['H']//2, sim.walls[context_wall_idx].anim); context_menu = None
                elif action == "Anchor": sim.toggle_anchor(context_wall_idx, context_pt_idx); context_menu = None
                elif action == "Set Length...":
                    val = simpledialog.askfloat("Set Length", "Enter target length:")
                    if val: sim.add_constraint_object(Length(context_wall_idx, val))
                    context_menu = None
                elif action == "CLOSE": context_menu = None
            if prop_dialog and prop_dialog.handle_event(event):
                if prop_dialog.apply: sim.update_wall_props(context_wall_idx, prop_dialog.get_values()); prop_dialog.apply = False
                if prop_dialog.done: prop_dialog = None
            if rot_dialog and rot_dialog.handle_event(event):
                if rot_dialog.apply: sim.set_wall_rotation(context_wall_idx, rot_dialog.get_values()); rot_dialog.apply = False
                if rot_dialog.done: rot_dialog = None

            # 5. Scene Interaction
            mouse_on_ui = (event.type == pygame.MOUSEBUTTONDOWN and (event.pos[0] > layout['RIGHT_X'] or event.pos[0] < layout['LEFT_W'] or event.pos[1] < config.TOP_MENU_H))
            if not mouse_on_ui and not menu_clicked and not context_menu and not prop_dialog:
                # Global Pan
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    app.state = InteractionState.PANNING; app.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                    app.state = InteractionState.IDLE
                elif event.type == pygame.MOUSEMOTION and app.state == InteractionState.PANNING:
                    app.pan_x += event.pos[0] - app.last_mouse_pos[0]; app.pan_y += event.pos[1] - app.last_mouse_pos[1]; app.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    app.zoom = max(0.1, min(app.zoom * (1.1 if event.y > 0 else 0.9), 50.0))
                
                # Tool Interaction
                if app.current_tool:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: # Right Click
                        if app.state == InteractionState.DRAGGING_GEOMETRY: app.current_tool.cancel()
                        else:
                            # Context Menu Logic
                            mx, my = event.pos
                            sim_x, sim_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            
                            hit_const = -1
                            for i, c in enumerate(sim.constraints):
                                if c.hit_test(mx, my): hit_const = i; break
                            
                            if hit_const != -1:
                                context_constraint_idx = hit_const; context_menu = ContextMenu(mx, my, ["Delete Constraint"])
                            else:
                                point_map = utils.get_grouped_points(sim, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                                hit_pt = None; base_r, step_r = 5, 4
                                for center_pos, items in point_map.items():
                                    if math.hypot(mx - center_pos[0], my - center_pos[1]) <= base_r + (len(items) - 1) * step_r: hit_pt = items[0]; break

                                if hit_pt:
                                    context_wall_idx = hit_pt[0]; context_pt_idx = hit_pt[1]; context_menu = ContextMenu(mx, my, ["Anchor"])
                                    # Handle Pending Constraint Click on Point
                                    if app.pending_constraint: handle_pending_constraint_click(pt_idx=hit_pt)
                                else:
                                    # Hit Test Walls
                                    hit_wall = -1; rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * app.zoom)
                                    for i, w in enumerate(sim.walls):
                                        if isinstance(w, Line):
                                            p1=w.start; p2=w.end; p3=np.array([sim_x, sim_y]); d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                                            dist = np.linalg.norm(p3-p1) if len_sq == 0 else np.linalg.norm(p3 - (p1 + max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))*d_vec))
                                            if dist < rad_sim: hit_wall = i; break
                                        elif isinstance(w, Circle):
                                            if abs(math.hypot(sim_x - w.center[0], sim_y - w.center[1]) - w.radius) < rad_sim: hit_wall = i; break
                                    
                                    if hit_wall != -1:
                                        # Handle Pending Constraint Click on Wall
                                        if app.pending_constraint: handle_pending_constraint_click(wall_idx=hit_wall)
                                        else:
                                            context_wall_idx = hit_wall
                                            opts = ["Properties", "Delete"]
                                            if isinstance(sim.walls[hit_wall], Line): opts.extend(["Set Length...", "Set Rotation..."])
                                            context_menu = ContextMenu(mx, my, opts)
                                    else:
                                        if app.mode == config.MODE_EDITOR: change_tool(config.TOOL_SELECT); app.set_status("Switched to Select Tool")
                    else:
                        app.current_tool.handle_event(event, layout)

            # 6. UI Widgets (REFACTORED)
            for el in current_ui_list:
                if el.handle_event(event):
                    if app.mode == config.MODE_EDITOR:
                        if el in constraint_btn_map: trigger_constraint(constraint_btn_map[el])
                        elif el in ui_action_map: ui_action_map[el]()
                    
                    if app.mode == config.MODE_SIM:
                        if el in ui_action_map: ui_action_map[el]()

        # --- Update ---
        if app.mode == config.MODE_SIM:
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
            if not sim.paused: sim.step(steps)
            
            if app.state == InteractionState.PAINTING:
                mx, my = pygame.mouse.get_pos()
                if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                    sx, sy = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                    if pygame.mouse.get_pressed()[0]: sim.add_particles_brush(sx, sy, slider_brush_size.val)
                    elif pygame.mouse.get_pressed()[2]: sim.delete_particles_brush(sx, sy, slider_brush_size.val)

        # --- Render ---
        renderer.draw_app(app, sim, layout, current_ui_list)
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