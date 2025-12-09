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
    app.tools[config.TOOL_SELECT] = SelectTool(app, sim)
    app.tools[config.TOOL_BRUSH] = BrushTool(app, sim)
    app.tools[config.TOOL_LINE] = LineTool(app, sim)
    app.tools[config.TOOL_RECT] = RectTool(app, sim)
    app.tools[config.TOOL_CIRCLE] = CircleTool(app, sim)
    app.tools[config.TOOL_POINT] = PointTool(app, sim)
    # Ref line uses LineTool but we handle logic inside tool or separate subclass. 
    app.tools[config.TOOL_REF] = LineTool(app, sim) 
    app.tools[config.TOOL_REF].name = "Ref Line"

    # Set Default
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
    btn_const_coincident = Button(rp_start_x, ae_curr_y, rp_width, 30, "Coincident (Pt-Pt/Ln)", toggle=False); ae_curr_y+=35
    btn_const_collinear = Button(rp_start_x, ae_curr_y, rp_width, 30, "Collinear (Pt-Ln)", toggle=False); ae_curr_y+=35
    btn_const_midpoint = Button(rp_start_x, ae_curr_y, rp_width, 30, "Midpoint (Pt-Ln)", toggle=False); ae_curr_y+=35
    btn_const_length = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Fix Length", toggle=False)
    btn_const_equal = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Equal Len", toggle=False); ae_curr_y+=35
    btn_const_parallel = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Parallel", toggle=False)
    btn_const_perp = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Perpendic", toggle=False); ae_curr_y+=35
    btn_const_horiz = Button(rp_start_x, ae_curr_y, c_btn_w, 30, "Horizontal", toggle=False)
    btn_const_vert = Button(rp_start_x + c_btn_w + 10, ae_curr_y, c_btn_w, 30, "Vertical", toggle=False); ae_curr_y+=35

    # UI Lists
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
        btn_const_coincident, btn_const_collinear, btn_const_midpoint, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert
    ]
    
    # Right panel specifically for layout updates
    right_panel_elements = [
        slider_gravity, slider_temp, slider_damping, slider_dt, slider_sigma, slider_epsilon, slider_M, slider_skin, 
        btn_thermostat, btn_boundaries, 
        btn_tool_brush, btn_tool_select, btn_tool_line, btn_tool_rect, btn_tool_circle, btn_tool_point, btn_tool_ref, 
        slider_brush_size, app.input_world, btn_resize, 
        btn_ae_save, btn_ae_discard, 
        btn_const_coincident, btn_const_collinear, btn_const_midpoint, btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_horiz, btn_const_vert
    ]

    # --- Helper Functions (Logic) ---
    
    def change_tool(tool_id):
        """Updates app state and button visuals"""
        app.change_tool(tool_id) # Updates AppState
        update_tool_buttons()

    def update_tool_buttons():
        """Updates the visual state of tool buttons based on current tool"""
        t_map = {
            config.TOOL_BRUSH: btn_tool_brush, config.TOOL_SELECT: btn_tool_select,
            config.TOOL_LINE: btn_tool_line, config.TOOL_RECT: btn_tool_rect,
            config.TOOL_CIRCLE: btn_tool_circle, config.TOOL_POINT: btn_tool_point,
            config.TOOL_REF: btn_tool_ref
        }
        for k, btn in t_map.items():
            # Match against tool name or ID. app.current_tool is the instance.
            # We know the map keys correspond to the tools dict in app
            is_active = (app.current_tool == app.tools.get(k))
            btn.active = is_active

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

    def reset_constraint_buttons():
        for btn in [btn_const_length, btn_const_equal, btn_const_parallel, btn_const_perp, btn_const_coincident, btn_const_collinear, btn_const_midpoint, btn_const_horiz, btn_const_vert]:
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
            
        elif ctype == 'COINCIDENT':
            # Case 1: Point-Point
            if len(app.selected_points) == 2:
                sp = list(app.selected_points)
                sim.add_constraint_object(Coincident(sp[0][0], sp[0][1], sp[1][0], sp[1][1])); applied = True
            # Case 2: Point-Line
            elif len(app.selected_points) == 1 and len(app.selected_walls) == 1:
                pt_tuple = list(app.selected_points)[0]
                w_idx = list(app.selected_walls)[0]
                if isinstance(sim.walls[w_idx], Line):
                    # -1 denotes the line itself
                    sim.add_constraint_object(Coincident(pt_tuple[0], pt_tuple[1], w_idx, -1)); applied = True

        elif ctype == 'COLLINEAR' and len(app.selected_points) == 1 and len(app.selected_walls) == 1:
            pt_tuple = list(app.selected_points)[0]
            w_idx = list(app.selected_walls)[0]
            if isinstance(sim.walls[w_idx], Line):
                sim.add_constraint_object(Collinear(pt_tuple[0], pt_tuple[1], w_idx)); applied = True

        elif ctype == 'MIDPOINT' and len(app.selected_points) == 1 and len(app.selected_walls) == 1:
            pt_tuple = list(app.selected_points)[0]
            w_idx = list(app.selected_walls)[0]
            if isinstance(sim.walls[w_idx], Line):
                sim.add_constraint_object(Midpoint(pt_tuple[0], pt_tuple[1], w_idx)); applied = True

        if applied:
            app.set_status(f"Applied {ctype}")
            app.selected_walls.clear(); app.selected_points.clear()
            app.pending_constraint = None; reset_constraint_buttons()
            sim.apply_constraints() # Apply immediately
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
            elif ctype == 'COLLINEAR': btn_const_collinear.active = True
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
                sim.apply_constraints()
        
        elif app.pending_constraint == 'COLLINEAR':
            if pt_idx is not None and len(app.pending_targets_points) == 0:
                app.pending_targets_points.append(pt_idx); app.set_status("Got Point. Select Line.")
            elif wall_idx is not None and len(app.pending_targets_walls) == 0:
                if isinstance(sim.walls[wall_idx], Line):
                    app.pending_targets_walls.append(wall_idx); app.set_status("Got Line. Select Point.")
            
            if len(app.pending_targets_points) == 1 and len(app.pending_targets_walls) == 1:
                sim.add_constraint_object(Collinear(app.pending_targets_points[0][0], app.pending_targets_points[0][1], app.pending_targets_walls[0]))
                app.set_status("Applied Collinear")
                app.pending_constraint = None; reset_constraint_buttons()
                sim.apply_constraints()

        elif app.pending_constraint in ['LENGTH', 'HORIZONTAL', 'VERTICAL']:
            if wall_idx is not None and isinstance(sim.walls[wall_idx], Line):
                if app.pending_constraint == 'LENGTH':
                    w = sim.walls[wall_idx]
                    sim.add_constraint_object(Length(wall_idx, np.linalg.norm(w.end - w.start)))
                else: sim.add_constraint_object(Angle(app.pending_constraint, wall_idx))
                app.set_status(f"Applied {app.pending_constraint}"); app.pending_constraint = None; reset_constraint_buttons()
                sim.apply_constraints()
        
        elif app.pending_constraint in ['EQUAL', 'PARALLEL', 'PERPENDICULAR']:
            if wall_idx is not None and isinstance(sim.walls[wall_idx], Line):
                if wall_idx not in app.pending_targets_walls:
                    app.pending_targets_walls.append(wall_idx)
                    app.set_status(f"Selected 1/2 for {app.pending_constraint}")
                    if len(app.pending_targets_walls) == 2:
                        if app.pending_constraint == 'EQUAL': sim.add_constraint_object(EqualLength(app.pending_targets_walls[0], app.pending_targets_walls[1]))
                        else: sim.add_constraint_object(Angle(app.pending_constraint, app.pending_targets_walls[0], app.pending_targets_walls[1]))
                        app.set_status(f"Applied {app.pending_constraint}"); app.pending_constraint = None; reset_constraint_buttons()
                        sim.apply_constraints()
        
        elif app.pending_constraint == 'COINCIDENT':
            if pt_idx is not None:
                if pt_idx not in app.pending_targets_points:
                    app.pending_targets_points.append(pt_idx)
                    app.set_status(f"Selected Point (1/2)")
            
            # Allow picking line for pending coincident
            if wall_idx is not None and isinstance(sim.walls[wall_idx], Line):
                if wall_idx not in app.pending_targets_walls:
                    app.pending_targets_walls.append(wall_idx)
                    app.set_status(f"Selected Line (1/2)")

            # Check if we have Pt-Pt
            if len(app.pending_targets_points) == 2:
                sim.add_constraint_object(Coincident(app.pending_targets_points[0][0], app.pending_targets_points[0][1], app.pending_targets_points[1][0], app.pending_targets_points[1][1]))
                app.set_status("Applied Coincident (Pt-Pt)"); app.pending_constraint = None; reset_constraint_buttons()
                sim.apply_constraints()
            
            # Check if we have Pt-Line
            elif len(app.pending_targets_points) == 1 and len(app.pending_targets_walls) == 1:
                pt = app.pending_targets_points[0]
                line_idx = app.pending_targets_walls[0]
                sim.add_constraint_object(Coincident(pt[0], pt[1], line_idx, -1))
                app.set_status("Applied Coincident (Pt-Ln)"); app.pending_constraint = None; reset_constraint_buttons()
                sim.apply_constraints()

    # --- Main Loop ---
    running = True
    context_menu = None; prop_dialog = None; rot_dialog = None
    context_wall_idx = -1; context_pt_idx = None

    while running:
        current_ui_list = ui_sim_elements if app.mode == config.MODE_SIM else ui_editor_elements
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # 1. Global Keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if app.current_tool: app.current_tool.cancel()
                    app.pending_constraint = None
                    app.selected_walls.clear(); app.selected_points.clear()
                    reset_constraint_buttons()
                    app.set_status("Cancelled")
                
                # Undo / Redo
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL): 
                    if app.current_tool: app.current_tool.cancel() # Cancel current tool action first
                    sim.undo()
                    app.set_status("Undo")
                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL): 
                    if app.current_tool: app.current_tool.cancel()
                    sim.redo()
                    app.set_status("Redo")
            
            # 2. Tool Switching
            if btn_tool_brush.handle_event(event): change_tool(config.TOOL_BRUSH)
            if btn_tool_select.handle_event(event): change_tool(config.TOOL_SELECT)
            if btn_tool_line.handle_event(event): change_tool(config.TOOL_LINE)
            if btn_tool_rect.handle_event(event): change_tool(config.TOOL_RECT)
            if btn_tool_circle.handle_event(event): change_tool(config.TOOL_CIRCLE)
            if btn_tool_point.handle_event(event): change_tool(config.TOOL_POINT)
            if btn_tool_ref.handle_event(event): change_tool(config.TOOL_REF)
            
            update_tool_buttons()

            # 3. Menus
            menu_clicked = menu_bar.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and menu_bar.active_menu and not menu_clicked:
                if menu_bar.dropdown_rect and menu_bar.dropdown_rect.collidepoint(event.pos):
                    # Handle menu logic 
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
                        menu_bar.active_menu = None

            # 4. Context/Dialogs
            if context_menu and context_menu.handle_event(event):
                # Handle context actions (same as before but using app state if needed)
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
                
            if prop_dialog and prop_dialog.handle_event(event):
                if prop_dialog.apply: sim.update_wall_props(context_wall_idx, prop_dialog.get_values()); prop_dialog.apply = False
                if prop_dialog.done: prop_dialog = None
            
            if rot_dialog and rot_dialog.handle_event(event):
                if rot_dialog.apply: sim.set_wall_rotation(context_wall_idx, rot_dialog.get_values()); rot_dialog.apply = False
                if rot_dialog.done: rot_dialog = None

            # 5. Scene Interaction (Delegate to Tool)
            mouse_on_ui = (event.type == pygame.MOUSEBUTTONDOWN and 
                           (event.pos[0] > layout['RIGHT_X'] or event.pos[0] < layout['LEFT_W'] or event.pos[1] < config.TOP_MENU_H))
            
            if not mouse_on_ui and not menu_clicked and not context_menu and not prop_dialog:
                # Global Pan (Middle Mouse)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    app.state = InteractionState.PANNING
                    app.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                    app.state = InteractionState.IDLE
                elif event.type == pygame.MOUSEMOTION and app.state == InteractionState.PANNING:
                    app.pan_x += event.pos[0] - app.last_mouse_pos[0]
                    app.pan_y += event.pos[1] - app.last_mouse_pos[1]
                    app.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    app.zoom = max(0.1, min(app.zoom * (1.1 if event.y > 0 else 0.9), 50.0))
                
                # Tool Interaction
                if app.current_tool:
                    # Right click global cancel or context
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                        if app.state == InteractionState.DRAGGING_GEOMETRY:
                            app.current_tool.cancel()
                        else:
                            # Context Menu
                            mx, my = event.pos
                            sim_x, sim_y = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                            
                            # Re-implement simple hit test for context menu
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
                                hit_wall = -1
                                rad_sim = 5.0 / (((layout['MID_W'] - 50) / sim.world_size) * app.zoom)
                                for i, w in enumerate(sim.walls):
                                    # ... Hit logic same as tool ...
                                    # For brevity, reuse select tool hit logic or copy-paste core checks
                                    # (Simplest: if SelectTool has public hit test, use it. If not, copy logic)
                                    pass # (Assume logic similar to SelectTool._hit_test_walls is here)
                                    if isinstance(w, Line):
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
                                    if app.mode == config.MODE_EDITOR:
                                        change_tool(config.TOOL_SELECT)
                                        app.set_status("Switched to Select Tool")
                    else:
                        app.current_tool.handle_event(event, layout)

            # 6. UI Widgets
            for el in current_ui_list:
                if el.handle_event(event):
                    if app.mode == config.MODE_EDITOR:
                        if el == btn_const_length: trigger_constraint('LENGTH')
                        elif el == btn_const_midpoint: trigger_constraint('MIDPOINT')
                        elif el == btn_const_equal: trigger_constraint('EQUAL')
                        elif el == btn_const_parallel: trigger_constraint('PARALLEL')
                        elif el == btn_const_perp: trigger_constraint('PERPENDICULAR')
                        elif el == btn_const_coincident: trigger_constraint('COINCIDENT')
                        elif el == btn_const_collinear: trigger_constraint('COLLINEAR')
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
                        elif el == btn_clear: sim.clear_particles()
                        elif el == btn_undo: sim.undo(); app.set_status("Undo")
                        elif el == btn_redo: sim.redo(); app.set_status("Redo")
                    
                    if app.mode == config.MODE_SIM:
                        if el == btn_reset: sim.reset_simulation()
                        elif el == btn_clear: sim.clear_particles()
                        elif el == btn_resize: sim.resize_world(app.input_world.get_value(50.0))
                        elif el == btn_undo: sim.undo(); app.set_status("Undo")
                        elif el == btn_redo: sim.redo(); app.set_status("Redo")

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
            
            # Brush Update
            if app.state == InteractionState.PAINTING:
                mx, my = pygame.mouse.get_pos()
                if layout['LEFT_X'] < mx < layout['RIGHT_X']:
                    sx, sy = utils.screen_to_sim(mx, my, app.zoom, app.pan_x, app.pan_y, sim.world_size, layout)
                    if pygame.mouse.get_pressed()[0]: sim.add_particles_brush(sx, sy, slider_brush_size.val)
                    elif pygame.mouse.get_pressed()[2]: sim.delete_particles_brush(sx, sy, slider_brush_size.val)

        # --- Render ---
        renderer.draw_app(app, sim, layout, current_ui_list)
        
        # Overlays
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