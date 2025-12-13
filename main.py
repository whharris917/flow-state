import pygame
import config
import utils
import file_io
import time
import math
import numpy as np
from tkinter import filedialog, Tk, simpledialog, messagebox

# Modules
from simulation_state import Simulation
from ui_widgets import SmartSlider, Button, InputField, ContextMenu, PropertiesDialog, MenuBar, RotationDialog
from geometry import Line, Circle
from constraints import Length, EqualLength, Angle, Midpoint, Coincident, Collinear, FixedAngle
from renderer import Renderer
from app_state import AppState, InteractionState
from tools import SelectTool, BrushTool, LineTool, RectTool, CircleTool, PointTool

class AnimationDialog:
    def __init__(self, x, y, driver_data):
        self.rect = pygame.Rect(x, y, 300, 280)
        self.driver = driver_data if driver_data else {'type': 'sin', 'amp': 15.0, 'freq': 0.5, 'phase': 0.0, 'rate': 10.0}
        self.done = False
        self.apply = False
        self.current_tab = self.driver.get('type', 'sin')
        
        # Shared UI
        self.btn_stop = Button(x + 20, y + 230, 100, 30, "Remove/Stop", toggle=False, color_inactive=(160, 60, 60))
        self.btn_ok = Button(x + 180, y + 230, 100, 30, "Start/Update", toggle=False, color_inactive=(60, 160, 60))
        
        # Tabs
        self.btn_tab_sin = Button(x + 20, y + 50, 120, 25, "Sinusoidal", toggle=False, active=(self.current_tab == 'sin'))
        self.btn_tab_lin = Button(x + 160, y + 50, 120, 25, "Linear", toggle=False, active=(self.current_tab == 'lin'))

        # Sinusoidal Inputs
        self.in_amp = InputField(x + 150, y + 90, 100, 25, str(self.driver.get('amp', 15.0)))
        self.in_freq = InputField(x + 150, y + 130, 100, 25, str(self.driver.get('freq', 0.5)))
        self.in_phase = InputField(x + 150, y + 170, 100, 25, str(self.driver.get('phase', 0.0)))
        
        # Linear Inputs
        self.in_rate = InputField(x + 150, y + 90, 100, 25, str(self.driver.get('rate', 10.0)))

    def handle_event(self, event):
        # Tab Switching
        if self.btn_tab_sin.handle_event(event):
            self.current_tab = 'sin'
            self.btn_tab_sin.active = True; self.btn_tab_lin.active = False
            return True
        if self.btn_tab_lin.handle_event(event):
            self.current_tab = 'lin'
            self.btn_tab_sin.active = False; self.btn_tab_lin.active = True
            return True

        # Input Fields based on Tab
        if self.current_tab == 'sin':
            if self.in_amp.handle_event(event): return True
            if self.in_freq.handle_event(event): return True
            if self.in_phase.handle_event(event): return True
        else:
            if self.in_rate.handle_event(event): return True
        
        if self.btn_stop.handle_event(event):
            self.driver = None 
            self.apply = True; self.done = True
            return True
            
        if self.btn_ok.handle_event(event):
            if self.current_tab == 'sin':
                self.driver = {
                    'type': 'sin',
                    'amp': self.in_amp.get_value(0.0),
                    'freq': self.in_freq.get_value(0.0),
                    'phase': self.in_phase.get_value(0.0)
                }
            else:
                self.driver = {
                    'type': 'lin',
                    'rate': self.in_rate.get_value(0.0)
                }
            self.apply = True; self.done = True
            return True
        return False

    def get_values(self):
        return self.driver

    def draw(self, screen, font):
        # Background
        shadow = self.rect.copy(); shadow.x += 5; shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, (45, 45, 48), self.rect, border_radius=6)
        pygame.draw.rect(screen, (0, 122, 204), self.rect, 1, border_radius=6)
        
        title = font.render("Drive Constraint", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 10))
        
        # Tabs
        self.btn_tab_sin.draw(screen, font)
        self.btn_tab_lin.draw(screen, font)
        
        if self.current_tab == 'sin':
            screen.blit(font.render("Amplitude:", True, (220, 220, 220)), (self.rect.x + 20, self.rect.y + 95))
            screen.blit(font.render("Freq (Hz):", True, (220, 220, 220)), (self.rect.x + 20, self.rect.y + 135))
            screen.blit(font.render("Phase (deg):", True, (220, 220, 220)), (self.rect.x + 20, self.rect.y + 175))
            self.in_amp.draw(screen, font)
            self.in_freq.draw(screen, font)
            self.in_phase.draw(screen, font)
        else:
            screen.blit(font.render("Rate (deg/s):", True, (220, 220, 220)), (self.rect.x + 20, self.rect.y + 95))
            self.in_rate.draw(screen, font)
            
        self.btn_stop.draw(screen, font)
        self.btn_ok.draw(screen, font)

class FastMDEditor:
    def __init__(self):
        # 1. System Setup
        try: self.root_tk = Tk(); self.root_tk.withdraw()
        except: self.root_tk = None

        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Flow State - Chemical Engineering Simulation")
        
        self.font = pygame.font.SysFont("segoeui", 15)
        self.big_font = pygame.font.SysFont("segoeui", 22)
        self.renderer = Renderer(self.screen, self.font, self.big_font)
        self.clock = pygame.time.Clock()
        self.running = True

        # 2. App Logic
        self.sim = Simulation()
        self.app = AppState()
        
        # Initialize Editor State
        self.app.editor_paused = False
        self.app.show_constraints = True
        self.app.geo_time = 0.0
        self.last_time = time.time()

        # 3. State Holders for Dialogs
        self.context_menu = None
        self.prop_dialog = None
        self.rot_dialog = None
        self.anim_dialog = None
        self.ctx_vars = {'wall': -1, 'pt': None, 'const': -1} # Store context indices

        # 4. Initialization
        self.init_layout()
        self.init_ui_elements()
        self.init_mappings() 
        self.init_tools()    

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
        self.menu_bar.items["File"] = ["New", "Open...", "Save", "Save As...", "---", "Import Geometry"] 
        
        # --- Mode Tabs (Top Center) ---
        tab_w = 120
        tab_h = 25
        tab_y = 2
        center_x = self.layout['W'] // 2
        
        # Tabs are just buttons that switch mode
        self.btn_tab_sim = Button(center_x - tab_w - 5, tab_y, tab_w, tab_h, "Simulation", toggle=False, active=True)
        self.btn_tab_edit = Button(center_x + 5, tab_y, tab_w, tab_h, "Geometry Editor", toggle=False, active=False)

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
        
        # New Editor Controls
        self.btn_editor_play = Button(rp_x, ae_y, btn_half, 35, "Pause", active=False, toggle=False); 
        self.btn_show_const = Button(rp_x + btn_half + 10, ae_y, btn_half, 35, "Hide Cnstr", active=False, toggle=False); ae_y+=45

        self.const_buttons = {
            'coincident': Button(rp_x, ae_y, rp_w, 30, "Coincident (Pt-Pt/Ln/Circ)", toggle=False),
            'collinear': Button(rp_x, ae_y+35, rp_w, 30, "Collinear (Pt-Ln)", toggle=False),
            'midpoint': Button(rp_x, ae_y+70, rp_w, 30, "Midpoint (Pt-Ln)", toggle=False),
            'length': Button(rp_x, ae_y+105, btn_half, 30, "Fix Length", toggle=False),
            'equal': Button(rp_x + btn_half + 10, ae_y+105, btn_half, 30, "Equal Len", toggle=False),
            'parallel': Button(rp_x, ae_y+140, btn_half, 30, "Parallel", toggle=False),
            'perp': Button(rp_x + btn_half + 10, ae_y+140, btn_half, 30, "Perpendic", toggle=False),
            'horiz': Button(rp_x, ae_y+175, btn_half, 30, "Horizontal", toggle=False),
            'vert': Button(rp_x + btn_half + 10, ae_y+175, btn_half, 30, "Vertical", toggle=False),
            'angle': Button(rp_x, ae_y+210, btn_half, 30, "Angle", toggle=False)
        }
        ae_y += 210
        # Move extend button down
        self.btn_extend = Button(rp_x + btn_half + 10, ae_y, btn_half, 30, "Extend Infinite", toggle=False)

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
            self.const_buttons['vert']: 'VERTICAL', self.const_buttons['angle']: 'ANGLE'
        }

        # 3. Actions Mapping
        self.ui_action_map = {
            self.btn_reset: lambda: self.sim.reset_simulation(),
            self.btn_clear: lambda: self.sim.clear_particles(),
            self.btn_undo: lambda: (self.sim.undo(), self.app.set_status("Undo")),
            self.btn_redo: lambda: (self.sim.redo(), self.app.set_status("Redo")),
            self.btn_resize: lambda: self.sim.resize_world(self.app.input_world.get_value(50.0)),
            self.btn_ae_discard: lambda: self.exit_editor_mode(self.app.sim_backup_state), # Legacy hook
            self.btn_ae_save: self.save_geo_dialog,
            self.btn_extend: self.toggle_extend,
            self.btn_editor_play: self.toggle_editor_play,
            self.btn_show_const: self.toggle_show_constraints,
            self.btn_tab_sim: lambda: self.switch_mode(config.MODE_SIM),
            self.btn_tab_edit: lambda: self.switch_mode(config.MODE_EDITOR)
        }

        # 4. UI Groups
        self.sim_elements = [
            self.btn_play, self.btn_clear, self.btn_reset, self.btn_undo, self.btn_redo,
            self.btn_tab_sim, self.btn_tab_edit, # Tabs
            *self.sliders.values(), self.btn_thermostat, self.btn_boundaries,
            self.tool_buttons['brush'], self.tool_buttons['line'], self.slider_brush, self.btn_resize
        ]
        self.editor_elements = [
            self.btn_ae_save, self.btn_ae_discard, self.btn_undo, self.btn_redo, self.btn_clear,
            self.btn_editor_play, self.btn_show_const,
            self.btn_tab_sim, self.btn_tab_edit, # Tabs
            *self.tool_buttons.values(), *self.const_buttons.values(), self.btn_extend
        ]
        
        # 5. Constraint Rules
        def get_l(s, w): return np.linalg.norm(s.walls[w].end - s.walls[w].start)
        def get_angle(s, w1, w2):
            l1 = s.walls[w1]; l2 = s.walls[w2]
            v1 = l1.end - l1.start
            v2 = l2.end - l2.start
            return math.degrees(math.atan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1]))

        self.constraint_defs = {
            'LENGTH':   [{'w':1, 'p':0, 't':Line, 'msg':"Select 1 Line", 'f': lambda s,w,p: Length(w[0], get_l(s, w[0]))}],
            'EQUAL':    [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: EqualLength(w[0], w[1])}],
            'PARALLEL': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PARALLEL', w[0], w[1])}],
            'PERPENDICULAR': [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: Angle('PERPENDICULAR', w[0], w[1])}],
            'HORIZONTAL': [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('HORIZONTAL', w[0]), 'multi':True}],
            'VERTICAL':   [{'w':1, 'p':0, 't':Line, 'msg':"Select Line(s)", 'f': lambda s,w,p: Angle('VERTICAL', w[0]), 'multi':True}],
            'MIDPOINT':   [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Midpoint(p[0][0], p[0][1], w[0])}],
            'COLLINEAR':  [{'w':1, 'p':1, 't':Line, 'msg':"Select Line & Point", 'f': lambda s,w,p: Collinear(p[0][0], p[0][1], w[0])}],
            'ANGLE':      [{'w':2, 'p':0, 't':Line, 'msg':"Select 2 Lines", 'f': lambda s,w,p: FixedAngle(w[0], w[1], get_angle(s, w[0], w[1]))}],
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
                if self.app.placing_geo_data:
                    self.app.placing_geo_data = None
                    self.app.set_status("Placement Cancelled")
                    return True
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
        if selection == "New": 
            if self.app.mode == config.MODE_SIM:
                self.sim.reset_simulation()
                self.app.input_world.set_value(config.DEFAULT_WORLD_SIZE)
                self.app.current_sim_filepath = None
            else:
                self.sim.walls = []
                self.sim.constraints = []
                self.app.current_geom_filepath = None
                self.app.set_status("New Geometry Created")
        
        elif selection == "Import Geometry":
            if self.root_tk:
                f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.geom"), ("All Files", "*.*")])
                if f:
                    if f.endswith(".sim"):
                        self.app.set_status("Error: Cannot import .sim file as geometry")
                        return
                    data, _ = file_io.load_geometry_file(f)
                    if data: 
                        self.app.placing_geo_data = data
                        self.app.set_status("Place Geometry")
        
        elif self.root_tk:
            if selection == "Save As..." or (selection == "Save"):
                is_save_as = (selection == "Save As...")
                
                if self.app.mode == config.MODE_EDITOR:
                    if is_save_as or not self.app.current_geom_filepath:
                        f = filedialog.asksaveasfilename(defaultextension=".geom", filetypes=[("Geometry Files", "*.geom")])
                        if f: self.app.current_geom_filepath = f
                    
                    if self.app.current_geom_filepath:
                        self.app.set_status(file_io.save_geometry_file(self.sim, self.app, self.app.current_geom_filepath))
                else:
                    if is_save_as or not self.app.current_sim_filepath:
                        f = filedialog.asksaveasfilename(defaultextension=".sim", filetypes=[("Simulation Files", "*.sim")])
                        if f: self.app.current_sim_filepath = f
                    
                    if self.app.current_sim_filepath:
                        self.app.set_status(file_io.save_file(self.sim, self.app, self.app.current_sim_filepath))
            
            elif selection == "Open...":
                if self.app.mode == config.MODE_EDITOR:
                    f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.geom")])
                    if f:
                        if not f.endswith(".geom"):
                            self.app.set_status("Error: Only .geom files allowed in Editor")
                            return
                        self.app.current_geom_filepath = f
                        data, view_state = file_io.load_geometry_file(f)
                        if data:
                            self.sim.clear_particles(snapshot=False)
                            self.sim.place_geometry(data, 0, 0, use_original_coordinates=True)
                            self.app.set_status(f"Loaded Geometry: {f}")
                            if view_state:
                                self.app.zoom = view_state['zoom']
                                self.app.pan_x = view_state['pan_x']
                                self.app.pan_y = view_state['pan_y']
                else:
                    f = filedialog.askopenfilename(filetypes=[("Simulation Files", "*.sim")])
                    if f:
                        if not f.endswith(".sim"):
                            self.app.set_status("Error: Only .sim files allowed in Simulation")
                            return
                        self.app.current_sim_filepath = f
                        success, msg, view_state = file_io.load_file(self.sim, f)
                        self.app.set_status(msg)
                        if success: 
                            self.app.input_world.set_value(self.sim.world_size)
                            if view_state:
                                self.app.zoom = view_state['zoom']
                                self.app.pan_x = view_state['pan_x']
                                self.app.pan_y = view_state['pan_y']

    def _handle_dialogs(self, event):
        captured = False
        # Context Menu
        if self.context_menu and self.context_menu.handle_event(event):
            action = self.context_menu.action
            if action == "Delete Constraint":
                if 0 <= self.ctx_vars['const'] < len(self.sim.constraints):
                    self.sim.snapshot(); self.sim.constraints.pop(self.ctx_vars['const']); self.sim.apply_constraints()
            elif action == "Set Angle...":
                val = simpledialog.askfloat("Set Angle", "Enter target angle (degrees):")
                if val is not None:
                    if 0 <= self.ctx_vars['const'] < len(self.sim.constraints):
                        self.sim.constraints[self.ctx_vars['const']].value = val
                        self.sim.apply_constraints()
            elif action == "Animate...":
                c = self.sim.constraints[self.ctx_vars['const']]
                driver = getattr(c, 'driver', None)
                self.anim_dialog = AnimationDialog(self.layout['W']//2, self.layout['H']//2, driver)
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
        if self.anim_dialog and self.anim_dialog.handle_event(event):
            if self.anim_dialog.apply:
                c = self.sim.constraints[self.ctx_vars['const']]
                if self.anim_dialog.get_values() is None:
                    if hasattr(c, 'driver'): del c.driver
                    c.base_value = None # Reset base on removal
                else:
                    c.driver = self.anim_dialog.get_values()
                    # Capture base value if not already captured or if starting fresh
                    if c.base_value is None: c.base_value = c.value 
                    # Capture start time relative to geo_time
                    if c.driver['type'] == 'lin': c.base_time = self.app.geo_time 
                self.anim_dialog.apply = False
            if self.anim_dialog.done: self.anim_dialog = None
            captured = True
        return captured

    def _handle_scene_mouse(self, event):
        # 0. Handle Geometry Placement (High Priority)
        if self.app.placing_geo_data:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    sx, sy = utils.screen_to_sim(mx, my, self.app.zoom, self.app.pan_x, self.app.pan_y, self.sim.world_size, self.layout)
                    self.sim.place_geometry(self.app.placing_geo_data, sx, sy)
                    self.app.placing_geo_data = None
                    self.app.set_status("Geometry Placed")
                elif event.button == 3:
                    self.app.placing_geo_data = None
                    self.app.set_status("Placement Cancelled")
            return

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
        if self.app.show_constraints:
            for i, c in enumerate(self.sim.constraints):
                if c.hit_test(mx, my): 
                    self.ctx_vars['const'] = i
                    opts = ["Delete Constraint"]
                    if getattr(c, 'type', '') == 'ANGLE':
                        opts.insert(0, "Set Angle...")
                        opts.insert(1, "Animate...")
                    self.context_menu = ContextMenu(mx, my, opts)
                    return

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
        # Update Time
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if self.app.mode == config.MODE_EDITOR and not self.app.editor_paused:
            self.app.geo_time += dt

        # Update Driven Constraints
        t = self.app.geo_time
        for c in self.sim.constraints:
            if hasattr(c, 'driver') and c.driver:
                d = c.driver
                
                # Fix: Ensure base_value is set. If not, set it now.
                if c.base_value is None:
                    c.base_value = c.value
                
                base = c.base_value
                
                if d['type'] == 'sin':
                    # Sinusoidal: A * sin(2*pi*f*t + phase)
                    offset = d['amp'] * math.sin(2 * math.pi * d['freq'] * t + math.radians(d['phase']))
                    c.value = base + offset
                elif d['type'] == 'lin':
                    # Linear drive: Angle = base + rate * (t - start_time)
                    # Use relative time since start of drive
                    t0 = getattr(c, 'base_time', 0.0)
                    dt_drive = t - t0
                    c.value = base + d['rate'] * dt_drive
        
        if self.app.mode == config.MODE_EDITOR:
            self.sim.apply_constraints()

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
        
        # Hide constraints hack
        held_constraints = self.sim.constraints
        if self.app.mode == config.MODE_EDITOR and not self.app.show_constraints:
            self.sim.constraints = []
            
        self.renderer.draw_app(self.app, self.sim, self.layout, ui_list)
        
        if self.app.mode == config.MODE_EDITOR and not self.app.show_constraints:
            self.sim.constraints = held_constraints

        # Buttons and Overlay
        self.menu_bar.draw(self.screen, self.font)
        self.btn_tab_sim.active = (self.app.mode == config.MODE_SIM)
        self.btn_tab_edit.active = (self.app.mode == config.MODE_EDITOR)
        self.btn_tab_sim.draw(self.screen, self.font)
        self.btn_tab_edit.draw(self.screen, self.font)

        if self.context_menu: self.context_menu.draw(self.screen, self.font)
        if self.prop_dialog: self.prop_dialog.draw(self.screen, self.font)
        if self.rot_dialog: self.rot_dialog.draw(self.screen, self.font)
        if self.anim_dialog: self.anim_dialog.draw(self.screen, self.font)
        pygame.display.flip()

    def change_tool(self, tool_id):
        self.app.change_tool(tool_id) 
        for btn, tid in self.tool_btn_map.items(): btn.active = (self.app.current_tool == self.app.tools.get(tid))

    def switch_mode(self, mode):
        if mode == self.app.mode: return

        # 1. Save current View State
        if self.app.mode == config.MODE_SIM:
            self.app.sim_view['zoom'] = self.app.zoom
            self.app.sim_view['pan_x'] = self.app.pan_x
            self.app.sim_view['pan_y'] = self.app.pan_y
            self.exit_sim_mode_logic() # Prepare to leave Sim
        else:
            self.app.editor_view['zoom'] = self.app.zoom
            self.app.editor_view['pan_x'] = self.app.pan_x
            self.app.editor_view['pan_y'] = self.app.pan_y
            self.exit_editor_mode_logic() # Prepare to leave Editor

        # 2. Switch
        self.app.mode = mode

        # 3. Restore Target View State
        if mode == config.MODE_SIM:
            self.app.zoom = self.app.sim_view['zoom']
            self.app.pan_x = self.app.sim_view['pan_x']
            self.app.pan_y = self.app.sim_view['pan_y']
            self.enter_sim_mode_logic()
        else:
            self.app.zoom = self.app.editor_view['zoom']
            self.app.pan_x = self.app.editor_view['pan_x']
            self.app.pan_y = self.app.editor_view['pan_y']
            self.enter_editor_mode_logic()

    def exit_sim_mode_logic(self):
        # Taking snapshot of running sim before entering editor
        self.sim.snapshot()
        self.app.sim_backup_state = self.sim.undo_stack.pop() # Get that snapshot back to hold it
        
        # Save running state
        self.app.was_sim_running = not self.sim.paused
        
        # Always Pause Sim when leaving
        self.btn_play.active = False
        self.sim.paused = True

    def enter_editor_mode_logic(self):
        # 1. Clear everything (physics + walls)
        self.sim.clear_particles(snapshot=False)
        # 2. Load Editor Geometry
        self.sim.walls = self.app.editor_storage['walls']
        self.sim.constraints = self.app.editor_storage['constraints']
        
        self.change_tool(self.app.editor_tool)
        # Removed status message

    def exit_editor_mode_logic(self):
        # Save current geometry to editor storage
        self.app.editor_storage['walls'] = self.sim.walls
        self.app.editor_storage['constraints'] = self.sim.constraints

    def enter_sim_mode_logic(self):
        # Restore the particle state
        if self.app.sim_backup_state:
            self.sim.restore_state(self.app.sim_backup_state)
            
        # Resume Sim ONLY if it was running before
        if self.app.was_sim_running:
            self.btn_play.active = True
            self.sim.paused = False
        else:
            self.btn_play.active = False
            self.sim.paused = True
            
        self.change_tool(self.app.sim_tool)
        # Removed status message

    def enter_geometry_mode(self):
        # Legacy stub - replaced by switch_mode
        pass

    def exit_editor_mode(self, restore_state):
        # Legacy stub - replaced by switch_mode
        pass

    def toggle_extend(self):
        if self.app.selected_walls:
            for idx in self.app.selected_walls:
                if idx < len(self.sim.walls) and isinstance(self.sim.walls[idx], Line): self.sim.walls[idx].infinite = not self.sim.walls[idx].infinite
            self.sim.rebuild_static_atoms(); self.app.set_status("Toggled Extend")
            
    def toggle_editor_play(self):
        self.app.editor_paused = not self.app.editor_paused
        self.btn_editor_play.text = "Play" if self.app.editor_paused else "Pause"
        self.btn_editor_play.cached_surf = None # Force redraw
        
    def toggle_show_constraints(self):
        self.app.show_constraints = not self.app.show_constraints
        self.btn_show_const.text = "Show Cnstr" if not self.app.show_constraints else "Hide Cnstr"
        self.btn_show_const.cached_surf = None # Force redraw

    def save_geo_dialog(self):
        if self.root_tk:
            f = filedialog.asksaveasfilename(defaultextension=".geom", filetypes=[("Geometry Files", "*.geom")])
            if f: 
                self.app.current_geom_filepath = f
                self.app.set_status(file_io.save_geometry_file(self.sim, self.app, f))

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