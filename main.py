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

class FastMDEditor:
    def __init__(self):
        # 1. System Setup
        try: self.root_tk = Tk(); self.root_tk.withdraw()
        except: self.root_tk = None

        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Fast MD - Geometry Editor")
        
        self.font = pygame.font.SysFont("segoeui", 15)
        self.big_font = pygame.font.SysFont("segoeui", 22)
        self.renderer = Renderer(self.screen, self.font, self.big_font)
        self.clock = pygame.time.Clock()
        self.running = True

        # 2. App Logic
        self.sim = Simulation()
        self.app = AppState()

        # 3. State Holders for Dialogs
        self.context_menu = None
        self.prop_dialog = None
        self.rot_dialog = None
        self.ctx_vars = {'wall': -1, 'pt': None, 'const': -1} # Store context indices

        # 4. Initialization
        self.init_layout()
        self.init_ui_elements()
        self.init_mappings() # Moved up: Must exist before tools need it
        self.init_tools()    # Moved down: Relies on mappings for visual updates

    def init_layout(self):
        self.layout = {
            'W': config.WINDOW_WIDTH, 'H': config.WINDOW_HEIGHT,
            'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
            'RIGHT_W': config.PANEL_RIGHT_WIDTH, 'RIGHT_X': config.WINDOW_WIDTH - config.PANEL_RIGHT_WIDTH,
            'MID_X': config.PANEL_LEFT_WIDTH, 'MID_W': config.WINDOW_WIDTH - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
            'MID_H': config.WINDOW_HEIGHT - config.TOP_MENU_H
        }

    def init_tools(self):
        tool_registry = [
            (config.TOOL_SELECT, SelectTool, None), (config.TOOL_BRUSH, BrushTool, None),
            (config.TOOL_LINE, LineTool, None), (config.TOOL_RECT, RectTool, None),
            (config.TOOL_CIRCLE, CircleTool, None), (config.TOOL_POINT, PointTool, None),
            (config.TOOL_REF, LineTool, "Ref Line"), 
        ]
        for tid, cls, name in tool_registry:
            self.app.tools[tid] = cls(self.app, self.sim)
            if name: self.app.tools[tid].name = name
        self.change_tool(config.TOOL_BRUSH)

    def init_ui_elements(self):
        # Menu
        self.menu_bar = MenuBar(self.layout['W'], config.TOP_MENU_H)
        self.menu_bar.items["File"] = ["New Simulation", "Open...", "Save", "Save As...", "---", "Create New Geometry", "Add Existing Geometry"] 
        
        # Left Panel
        lp_y = config.TOP_MENU_H + 20; lp_m = 10
        self.btn_play = Button(self.layout['LEFT_X'] + lp_m, lp_y, self.layout['LEFT_W']-20, 35, "Play/Pause", active=False, color_active=(60, 120, 60), color_inactive=(180, 60, 60)); lp_y += 50
        self.btn_clear = Button(self.layout['LEFT_X'] + lp_m, lp_y, self.layout['LEFT_W']-20, 35, "Clear", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_y += 50
        self.btn_reset = Button(self.layout['LEFT_X'] + lp_m, lp_y, self.layout['LEFT_W']-20, 35, "Reset", active=False, toggle=False, color_inactive=(80, 80, 80)); lp_y += 50
        self.btn_undo = Button(self.layout['LEFT_X'] + lp_m, lp_y, self.layout['LEFT_W']-20, 35, "Undo", active=False, toggle=False); lp_y += 45
        self.btn_redo = Button(self.layout['LEFT_X'] + lp_m, lp_y, self.layout['LEFT_W']-20, 35, "Redo", active=False, toggle=False)

        # Right Panel
        rp_x = self.layout['RIGHT_X'] + 15; rp_w = self.layout['RIGHT_W'] - 30; rp_y = config.TOP_MENU_H + 120
        self.sliders = {
            'gravity': SmartSlider(rp_x, rp_y, rp_w, 0.0, 50.0, config.DEFAULT_GRAVITY, "Gravity", hard_min=0.0),
            'temp': SmartSlider(rp_x, rp_y+60, rp_w, 0.0, 5.0, 0.5, "Temperature", hard_min=0.0),
            'damping': SmartSlider(rp_x, rp_y+120, rp_w, 0.90, 1.0, config.DEFAULT_DAMPING, "Damping", hard_min=0.0, hard_max=1.0),
            'dt': SmartSlider(rp_x, rp_y+180, rp_w, 0.0001, 0.01, config.DEFAULT_DT, "Time Step (dt)", hard_min=0.00001),
            'sigma': SmartSlider(rp_x, rp_y+240, rp_w, 0.5, 2.0, config.ATOM_SIGMA, "Sigma (Size)", hard_min=0.1),
            'epsilon': SmartSlider(rp_x, rp_y+300, rp_w, 0.1, 5.0, config.ATOM_EPSILON, "Epsilon (Strength)", hard_min=0.0),
            'speed': SmartSlider(rp_x, rp_y+360, rp_w, 1.0, 100.0, float(config.DEFAULT_DRAW_M), "Speed (Steps/Frame)", hard_min=1.0),
            'skin': SmartSlider(rp_x, rp_y+420, rp_w, 0.1, 2.0, config.DEFAULT_SKIN_DISTANCE, "Skin Distance", hard_min=0.05)
        }
        rp_y += 480
        
        btn_half = (rp_w - 10) // 2
        self.btn_thermostat = Button(rp_x, rp_y, btn_half, 30, "Thermostat", active=False)
        self.btn_boundaries = Button(rp_x + btn_half + 10, rp_y, btn_half, 30, "Bounds", active=False); rp_y += 40
        
        # Tools
        self.tool_buttons = {
            'brush': Button(rp_x, rp_y, btn_half, 30, "Brush", active=True, toggle=False),
            'select': Button(rp_x + btn_half + 10, rp_y, btn_half, 30, "Select", active=False, toggle=False),
            'line': Button(rp_x, rp_y+40, btn_half, 30, "Line", active=False, toggle=False),
            'rect': Button(rp_x + btn_half + 10, rp_y+40, btn_half, 30, "Rectangle", active=False, toggle=False),
            'circle': Button(rp_x, rp_y+80, btn_half, 30, "Circle", active=False, toggle=False),
            'point': Button(rp_x + btn_half + 10, rp_y+80, btn_half, 30, "Point", active=False, toggle=False),
            'ref': Button(rp_x, rp_y+120, btn_half, 30, "Ref Line", active=False, toggle=False)
        }
        rp_y += 160
        
        self.slider_brush = SmartSlider(rp_x, rp_y, rp_w, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5); rp_y+=60
        self.app.input_world = InputField(rp_x + 80, rp_y, 60, 25, str(config.DEFAULT_WORLD_SIZE))
        self.btn_resize = Button(rp_x + 150, rp_y, rp_w - 150, 25, "Resize & Restart", active=False, toggle=False)

        # Editor Buttons
        ae_y = config.TOP_MENU_H + 40
        self.btn_ae_save = Button(rp_x, ae_y, rp_w, 40, "Save Geometry", active=False, toggle=False, color_inactive=(50, 120, 50)); ae_y+=50
        self.btn_ae_discard = Button(rp_x, ae_y, rp_w, 40, "Discard & Exit", active=False, toggle=False, color_inactive=(150, 50, 50)); ae_y+=50
        
        self.const_buttons = {
            'coincident': Button(rp_x, ae_y, rp_w, 30, "Coincident (Pt-Pt/Ln/Circ)", toggle=False),
            'collinear': Button(rp_x, ae_y+35, rp_w, 30, "Collinear (Pt-Ln)", toggle=False),
            'midpoint': Button(rp_x, ae_y+70, rp_w, 30, "Midpoint (Pt-Ln)", toggle=False),
            'length': Button(rp_x, ae_y+105, btn_half, 30, "Fix Length", toggle=False),
            'equal': Button(rp_x + btn_half + 10, ae_y+105, btn_half, 30, "Equal Len", toggle=False),
            'parallel': Button(rp_x, ae_y+140, btn_half, 30, "Parallel", toggle=False),
            'perp': Button(rp_x + btn_half + 10, ae_y+140, btn_half, 30, "Perpendic", toggle=False),
            'horiz': Button(rp_x, ae_y+175, btn_half, 30, "Horizontal", toggle=False),
            'vert': Button(rp_x + btn_half + 10, ae_y+175, btn_half, 30, "Vertical", toggle=False)
        }
        ae_y += 210
        self.btn_extend = Button(rp_x, ae_y, rp_w, 30, "Extend Infinite", toggle=False)

    def init_mappings(self):
        # 1. Tool Mapping
        self.tool_btn_map = {
            self.tool_buttons['brush']: config.TOOL_BRUSH, self.tool_buttons['select']: config.TOOL_SELECT,
            self.tool_buttons['line']: config.TOOL_LINE, self.tool_buttons['rect']: config.TOOL_RECT,
            self.tool_buttons['circle']: config.TOOL_CIRCLE, self.tool_buttons['point']: config.TOOL_POINT,
            self.tool_buttons['ref']: config.TOOL_REF
        }
        
        # 2. Constraint Mapping
        self.constraint_btn_map = {
            self.const_buttons['length']: 'LENGTH', self.const_buttons['equal']: 'EQUAL',
            self.const_buttons['parallel']: 'PARALLEL', self.const_buttons['perp']: 'PERPENDICULAR',
            self.const_buttons['coincident']: 'COINCIDENT', self.const_buttons['collinear']: 'COLLINEAR',
            self.const_buttons['midpoint']: 'MIDPOINT', self.const_buttons['horiz']: 'HORIZONTAL',
            self.const_buttons['vert']: 'VERTICAL'
        }

        # 3. Actions Mapping
        self.ui_action_map = {
            self.btn_reset: lambda: self.sim.reset_simulation(),
            self.btn_clear: lambda: self.sim.clear_particles(),
            self.btn_undo: lambda: (self.sim.undo(), self.app.set_status("Undo")),
            self.btn_redo: lambda: (self.sim.redo(), self.app.set_status("Redo")),
            self.btn_resize: lambda: self.sim.resize_world(self.app.input_world.get_value(50.0)),
            self.btn_ae_discard: lambda: self.exit_editor_mode(self.app.sim_backup_state),
            self.btn_ae_save: self.save_geo_dialog,
            self.btn_extend: self.toggle_extend
        }

        # 4. UI Groups
        self.sim_elements = [
            self.btn_play, self.btn_clear, self.btn_reset, self.btn_undo, self.btn_redo,
            *self.sliders.values(), self.btn_thermostat, self.btn_boundaries,
            self.tool_buttons['brush'], self.tool_buttons['line'], self.slider_brush, self.btn_resize
        ]
        self.editor_elements = [
            self.btn_ae_save, self.btn_ae_discard, self.btn_undo, self.btn_redo, self.btn_clear,
            *self.tool_buttons.values(), *self.const_buttons.values(), self.btn_extend
        ]
        
        # 5. Constraint Rules
        def get_l(s, w): return np.linalg.norm(s.walls[w].end - s.walls[w].start)
        self.constraint_defs = {
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

    # --- Core Loop ---
    def run(self):
        while self.running:
            self.handle_input()
            self.update_physics()
            self.render()
            self.clock.tick()
        
        if self.root_tk: self.root_tk.destroy()
        pygame.quit()

    # --- Event Dispatching ---
    def handle_input(self):
        ui_list = self.sim_elements if self.app.mode == config.MODE_SIM else self.editor_elements
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            # 1. Global Input
            if self._handle_keys(event): continue
            
            # 2. Tool Switching
            tool_switched = False
            for btn, tid in self.tool_btn_map.items():
                if btn.handle_event(event): 
                    self.change_tool(tid); tool_switched = True
            if tool_switched: continue

            # 3. Menus & Dialogs (Blocking)
            if self._handle_menus(event): continue
            if self._handle_dialogs(event): continue

            # 4. UI Widgets & Action Buttons
            ui_interacted = False
            for el in ui_list:
                if el.handle_event(event):
                    ui_interacted = True
                    if self.app.mode == config.MODE_EDITOR and el in self.constraint_btn_map:
                        self.trigger_constraint(self.constraint_btn_map[el])
                    elif el in self.ui_action_map:
                        self.ui_action_map[el]()
            
            # 5. Scene Interaction (Mouse in Viewport)
            mouse_on_ui = (event.type == pygame.MOUSEBUTTONDOWN and 
                          (event.pos[0] > self.layout['RIGHT_X'] or event.pos[0] < self.layout['LEFT_W'] or event.pos[1] < config.TOP_MENU_H))
            
            if not mouse_on_ui and not ui_interacted:
                self._handle_scene_mouse(event)

    def _handle_keys(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.app.current_tool: self.app.current_tool.cancel()
                self.app.pending_constraint = None
                self.app.selected_walls.clear(); self.app.selected_points.clear()
                for btn in self.constraint_btn_map.keys(): btn.active = False
                self.app.set_status("Cancelled")
                return True
            if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                if self.app.current_tool: self.app.current_tool.cancel()
                self.sim.undo(); self.app.set_status("Undo"); return True
            if event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                if self.app.current_tool: self.app.current_tool.cancel()
                self.sim.redo(); self.app.set_status("Redo"); return True
        return False

    def _handle_menus(self, event):
        if self.menu_bar.handle_event(event): return True
        if event.type == pygame.MOUSEBUTTONDOWN and self.menu_bar.active_menu:
            if self.menu_bar.dropdown_rect and self.menu_bar.dropdown_rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.menu_bar.dropdown_rect.y - 5; idx = rel_y // 30
                opts = self.menu_bar.items[self.menu_bar.active_menu]
                if 0 <= idx < len(opts): self._execute_menu(opts[idx])
            self.menu_bar.active_menu = None
            return True
        return False

    def _execute_menu(self, selection):
        if selection == "New Simulation": self.sim.reset_simulation(); self.app.input_world.set_value(config.DEFAULT_WORLD_SIZE)
        elif selection == "Create New Geometry" and self.app.mode == config.MODE_SIM: self.enter_geometry_mode()
        elif selection == "Add Existing Geometry" and self.app.mode == config.MODE_SIM and self.root_tk:
            f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.json")])
            if f: 
                data = file_io.load_geometry_file(f)
                if data: self.app.placing_geo_data = data; self.app.set_status("Place Geometry")
        elif self.root_tk:
            if selection == "Save As..." or (selection == "Save" and not self.app.current_filepath):
                f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
                if f: self.app.current_filepath = f; self.app.set_status(file_io.save_file(self.sim, f))
            elif selection == "Save" and self.app.current_filepath: self.app.set_status(file_io.save_file(self.sim, self.app.current_filepath))
            elif selection == "Open...":
                f = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
                if f: 
                    self.app.current_filepath = f
                    success, msg, lset = file_io.load_file(self.sim, f)
                    self.app.set_status(msg)
                    if success: self.app.input_world.set_value(self.sim.world_size); self.app.zoom=1.0; self.app.pan_x=0; self.app.pan_y=0

    def _handle_dialogs(self, event):
        captured = False
        # Context Menu
        if self.context_menu and self.context_menu.handle_event(event):
            action = self.context_menu.action
            if action == "Delete Constraint":
                if 0 <= self.ctx_vars['const'] < len(self.sim.constraints):
                    self.sim.snapshot(); self.sim.constraints.pop(self.ctx_vars['const']); self.sim.apply_constraints()
            elif action == "Delete": 
                self.sim.remove_wall(self.ctx_vars['wall']); self.app.selected_walls.clear(); self.app.selected_points.clear()
            elif action == "Properties":
                w_props = self.sim.walls[self.ctx_vars['wall']].to_dict()
                self.prop_dialog = PropertiesDialog(self.layout['W']//2, self.layout['H']//2, w_props)
            elif action == "Set Rotation...":
                self.rot_dialog = RotationDialog(self.layout['W']//2, self.layout['H']//2, self.sim.walls[self.ctx_vars['wall']].anim)
            elif action == "Anchor": self.sim.toggle_anchor(self.ctx_vars['wall'], self.ctx_vars['pt'])
            elif action == "Set Length...":
                val = simpledialog.askfloat("Set Length", "Enter target length:")
                if val: self.sim.add_constraint_object(Length(self.ctx_vars['wall'], val))
            self.context_menu = None; captured = True

        # Property Dialogs
        if self.prop_dialog and self.prop_dialog.handle_event(event):
            if self.prop_dialog.apply: self.sim.update_wall_props(self.ctx_vars['wall'], self.prop_dialog.get_values()); self.prop_dialog.apply = False
            if self.prop_dialog.done: self.prop_dialog = None
            captured = True
        if self.rot_dialog and self.rot_dialog.handle_event(event):
            if self.rot_dialog.apply: self.sim.set_wall_rotation(self.ctx_vars['wall'], self.rot_dialog.get_values()); self.rot_dialog.apply = False
            if self.rot_dialog.done: self.rot_dialog = None
            captured = True
        return captured

    def _handle_scene_mouse(self, event):
        # Global Pan
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            self.app.state = InteractionState.PANNING; self.app.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self.app.state = InteractionState.IDLE
        elif event.type == pygame.MOUSEMOTION and self.app.state == InteractionState.PANNING:
            self.app.pan_x += event.pos[0] - self.app.last_mouse_pos[0]; self.app.pan_y += event.pos[1] - self.app.last_mouse_pos[1]; self.app.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            self.app.zoom = max(0.1, min(self.app.zoom * (1.1 if event.y > 0 else 0.9), 50.0))
        
        # Tool / Context
        if self.app.current_tool:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: # Right Click
                if self.app.state == InteractionState.DRAGGING_GEOMETRY: self.app.current_tool.cancel()
                else: self._spawn_context_menu(event.pos)
            else:
                self.app.current_tool.handle_event(event, self.layout)

    def _spawn_context_menu(self, pos):
        mx, my = pos
        sim_x, sim_y = utils.screen_to_sim(mx, my, self.app.zoom, self.app.pan_x, self.app.pan_y, self.sim.world_size, self.layout)
        
        # Hit Test Constraints
        for i, c in enumerate(self.sim.constraints):
            if c.hit_test(mx, my): 
                self.ctx_vars['const'] = i; self.context_menu = ContextMenu(mx, my, ["Delete Constraint"]); return

        # Hit Test Points
        point_map = utils.get_grouped_points(self.sim, self.app.zoom, self.app.pan_x, self.app.pan_y, self.sim.world_size, self.layout)
        hit_pt = None; base_r, step_r = 5, 4
        for center_pos, items in point_map.items():
            if math.hypot(mx - center_pos[0], my - center_pos[1]) <= base_r + (len(items) - 1) * step_r: hit_pt = items[0]; break

        if hit_pt:
            self.ctx_vars['wall'] = hit_pt[0]; self.ctx_vars['pt'] = hit_pt[1]
            self.context_menu = ContextMenu(mx, my, ["Anchor"])
            if self.app.pending_constraint: self.handle_pending_constraint_click(pt_idx=hit_pt)
            return

        # Hit Test Walls
        hit_wall = -1; rad_sim = 5.0 / (((self.layout['MID_W'] - 50) / self.sim.world_size) * self.app.zoom)
        for i, w in enumerate(self.sim.walls):
            if isinstance(w, Line):
                p1=w.start; p2=w.end; p3=np.array([sim_x, sim_y]); d_vec = p2-p1; len_sq = np.dot(d_vec, d_vec)
                dist = np.linalg.norm(p3-p1) if len_sq == 0 else np.linalg.norm(p3 - (p1 + max(0, min(1, np.dot(p3-p1, d_vec)/len_sq))*d_vec))
                if dist < rad_sim: hit_wall = i; break
            elif isinstance(w, Circle):
                if abs(math.hypot(sim_x - w.center[0], sim_y - w.center[1]) - w.radius) < rad_sim: hit_wall = i; break
        
        if hit_wall != -1:
            if self.app.pending_constraint: self.handle_pending_constraint_click(wall_idx=hit_wall)
            else:
                self.ctx_vars['wall'] = hit_wall
                opts = ["Properties", "Delete"]
                if isinstance(self.sim.walls[hit_wall], Line): opts.extend(["Set Length...", "Set Rotation..."])
                self.context_menu = ContextMenu(mx, my, opts)
        else:
            if self.app.mode == config.MODE_EDITOR: self.change_tool(config.TOOL_SELECT); self.app.set_status("Switched to Select Tool")

    # --- Logic Helpers ---
    def update_physics(self):
        if self.app.mode == config.MODE_SIM:
            self.sim.paused = not self.btn_play.active
            self.sim.gravity = self.sliders['gravity'].val; self.sim.target_temp = self.sliders['temp'].val
            self.sim.damping = self.sliders['damping'].val; self.sim.dt = self.sliders['dt'].val
            self.sim.sigma = self.sliders['sigma'].val; self.sim.epsilon = self.sliders['epsilon'].val
            self.sim.skin_distance = self.sliders['skin'].val
            self.sim.use_thermostat = self.btn_thermostat.active; self.sim.use_boundaries = self.btn_boundaries.active
            if not self.sim.paused: self.sim.step(int(self.sliders['speed'].val))
            
            if self.app.state == InteractionState.PAINTING:
                mx, my = pygame.mouse.get_pos()
                if self.layout['LEFT_X'] < mx < self.layout['RIGHT_X']:
                    sx, sy = utils.screen_to_sim(mx, my, self.app.zoom, self.app.pan_x, self.app.pan_y, self.sim.world_size, self.layout)
                    if pygame.mouse.get_pressed()[0]: self.sim.add_particles_brush(sx, sy, self.slider_brush.val)
                    elif pygame.mouse.get_pressed()[2]: self.sim.delete_particles_brush(sx, sy, self.slider_brush.val)

    def render(self):
        ui_list = self.sim_elements if self.app.mode == config.MODE_SIM else self.editor_elements
        self.renderer.draw_app(self.app, self.sim, self.layout, ui_list)
        self.menu_bar.draw(self.screen, self.font)
        if self.context_menu: self.context_menu.draw(self.screen, self.font)
        if self.prop_dialog: self.prop_dialog.draw(self.screen, self.font)
        if self.rot_dialog: self.rot_dialog.draw(self.screen, self.font)
        pygame.display.flip()

    def change_tool(self, tool_id):
        self.app.change_tool(tool_id) 
        for btn, tid in self.tool_btn_map.items(): btn.active = (self.app.current_tool == self.app.tools.get(tid))

    def enter_geometry_mode(self):
        self.sim.snapshot(); self.app.sim_backup_state = self.sim.undo_stack.pop()
        self.sim.clear_particles(snapshot=False); self.sim.world_size = 30.0; self.sim.walls = []; self.sim.constraints = []
        self.app.mode = config.MODE_EDITOR; self.change_tool(config.TOOL_SELECT); self.app.zoom = 1.5; self.app.pan_x = 0; self.app.pan_y = 0
        self.app.set_status("Entered Geometry Editor")

    def exit_editor_mode(self, restore_state):
        self.sim.clear_particles(snapshot=False)
        if restore_state: self.sim.restore_state(restore_state)
        else: self.sim.world_size = config.DEFAULT_WORLD_SIZE
        self.app.mode = config.MODE_SIM; self.app.zoom = 1.0; self.app.pan_x = 0; self.app.pan_y = 0
        self.app.set_status("Returned to Simulation")

    def toggle_extend(self):
        if self.app.selected_walls:
            for idx in self.app.selected_walls:
                if idx < len(self.sim.walls) and isinstance(self.sim.walls[idx], Line): self.sim.walls[idx].infinite = not self.sim.walls[idx].infinite
            self.sim.rebuild_static_atoms(); self.app.set_status("Toggled Extend")

    def save_geo_dialog(self):
        if self.root_tk:
            f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Geometry Files", "*.json")])
            if f: self.app.set_status(file_io.save_geometry_file(self.sim, f)); self.exit_editor_mode(self.app.sim_backup_state)

    def trigger_constraint(self, ctype):
        for btn, c_val in self.constraint_btn_map.items(): btn.active = (c_val == ctype)
        is_multi = False
        if ctype in self.constraint_defs and self.constraint_defs[ctype][0].get('multi'):
            walls = list(self.app.selected_walls)
            if walls:
                count = 0
                for w_idx in walls:
                    if self.try_apply_constraint(ctype, [w_idx], []): count += 1
                if count > 0:
                    self.app.set_status(f"Applied {ctype} to {count} items")
                    self.app.selected_walls.clear(); self.app.selected_points.clear()
                    self.sim.apply_constraints(); is_multi = True
        
        if is_multi: return
        walls = list(self.app.selected_walls); pts = list(self.app.selected_points)
        if self.try_apply_constraint(ctype, walls, pts):
            self.app.set_status(f"Applied {ctype}")
            self.app.selected_walls.clear(); self.app.selected_points.clear()
            self.app.pending_constraint = None
            for btn in self.constraint_btn_map.keys(): btn.active = False
            self.sim.apply_constraints()
        else:
            self.app.pending_constraint = ctype
            self.app.pending_targets_walls = list(self.app.selected_walls)
            self.app.pending_targets_points = list(self.app.selected_points)
            self.app.selected_walls.clear(); self.app.selected_points.clear()
            msg = self.constraint_defs[ctype][0]['msg'] if ctype in self.constraint_defs else "Select targets..."
            self.app.set_status(f"{ctype}: {msg}")

    def handle_pending_constraint_click(self, wall_idx=None, pt_idx=None):
        if not self.app.pending_constraint: return
        if wall_idx is not None and wall_idx not in self.app.pending_targets_walls: self.app.pending_targets_walls.append(wall_idx)
        if pt_idx is not None and pt_idx not in self.app.pending_targets_points: self.app.pending_targets_points.append(pt_idx)
        
        if self.try_apply_constraint(self.app.pending_constraint, self.app.pending_targets_walls, self.app.pending_targets_points):
            self.app.set_status(f"Applied {self.app.pending_constraint}")
            self.app.pending_constraint = None
            for btn in self.constraint_btn_map.keys(): btn.active = False
            self.sim.apply_constraints()
        else:
            ctype = self.app.pending_constraint; msg = self.constraint_defs[ctype][0]['msg']
            self.app.set_status(f"{ctype} ({len(self.app.pending_targets_walls)}W, {len(self.app.pending_targets_points)}P): {msg}")

    def try_apply_constraint(self, ctype, wall_idxs, pt_idxs):
        rules = self.constraint_defs.get(ctype, [])
        for r in rules:
            if len(wall_idxs) == r['w'] and len(pt_idxs) == r['p']:
                valid = True
                if r.get('t'):
                    for w_idx in wall_idxs:
                        if not isinstance(self.sim.walls[w_idx], r['t']): valid = False; break
                if valid: self.sim.add_constraint_object(r['f'](self.sim, wall_idxs, pt_idxs)); return True
        return False

def main():
    editor = FastMDEditor()
    editor.run()

if __name__ == "__main__":
    main()