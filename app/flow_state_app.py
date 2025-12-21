import pygame
pygame.mixer.pre_init(44100, -16, 1, 1024)
pygame.init()

import config
import utils
import file_io
import time
import sys
import subprocess
from tkinter import filedialog, Tk

from simulation import Simulation
from session import Session
from ui_widgets import InputField
from geometry import Line, Circle
from renderer import Renderer
from tools import SelectTool, BrushTool, LineTool, RectTool, CircleTool, PointTool
from ui_manager import UIManager
from input_handler import InputHandler
from sound_manager import SoundManager
from app_controller import AppController

class FlowStateApp:
    def __init__(self, start_mode=config.MODE_SIM):
        try: self.root_tk = Tk(); self.root_tk.withdraw()
        except: self.root_tk = None

        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE)
        
        title_suffix = "Simulation" if start_mode == config.MODE_SIM else "Model Builder"
        pygame.display.set_caption(f"Flow State - {title_suffix}")
        
        self.font = pygame.font.SysFont("segoeui", 15)
        self.big_font = pygame.font.SysFont("segoeui", 22)
        
        self.renderer = Renderer(self.screen, self.font, self.big_font)
        self.clock = pygame.time.Clock()
        self.running = True

        self.sound_manager = SoundManager.get()
        if start_mode == config.MODE_SIM:
            self.sound_manager.play_music('Whimsical')
        else:
            self.sound_manager.play_music('Jungle')

        self.session = Session()
        self.session.mode = start_mode 
        
        is_editor = (start_mode == config.MODE_EDITOR)
        self.sim = Simulation(skip_warmup=is_editor)
        self.sketch = self.sim.sketch
        self.session.input_world = InputField(0, 0, 0, 0)
        self.session.zoom = 0.9
        
        self.session.editor_paused = False 
        self.session.show_constraints = True
        self.session.geo_time = 0.0
        self.last_time = time.time()

        w, h = self.screen.get_size()
        self.init_layout(w, h)
        
        self.ui = UIManager(self.layout, self.session.input_world, mode=self.session.mode)
        
        # --- NEW CONTROLLER ---
        self.actions = AppController(self)
        self.input_handler = InputHandler(self) 
        
        self.init_tools() 
        if self.session.mode == config.MODE_EDITOR:
            self.change_tool(config.TOOL_SELECT) 
        else:
            self.change_tool(config.TOOL_BRUSH) 

    @property
    def walls(self): return self.sim.walls
    @property
    def constraints(self): return self.sim.constraints
    @property
    def geo(self): return self.sim.geo

    def init_layout(self, w, h):
        self.layout = {
            'W': w, 'H': h,
            'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
            'RIGHT_W': config.PANEL_RIGHT_WIDTH, 'RIGHT_X': w - config.PANEL_RIGHT_WIDTH,
            'MID_X': config.PANEL_LEFT_WIDTH, 'MID_W': w - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
            'MID_H': h - config.TOP_MENU_H
        }

    def init_tools(self):
        tool_registry = [
            (config.TOOL_SELECT, SelectTool, None), (config.TOOL_BRUSH, BrushTool, None),
            (config.TOOL_LINE, LineTool, None), (config.TOOL_RECT, RectTool, None),
            (config.TOOL_CIRCLE, CircleTool, None), (config.TOOL_POINT, PointTool, None),
            (config.TOOL_REF, LineTool, "Ref Line"), 
        ]
        for tid, cls, name in tool_registry:
            self.session.tools[tid] = cls(self) 
            if name: self.session.tools[tid].name = name

    def handle_resize(self, w, h):
        old_mid_w = self.layout['MID_W']
        old_mid_h = self.layout['MID_H']
        self.init_layout(w, h)
        if old_mid_w > 0 and old_mid_h > 0:
            new_mid_w = self.layout['MID_W']
            new_mid_h = self.layout['MID_H']
            correction = (old_mid_w / old_mid_h) / (new_mid_w / new_mid_h)
            if abs(correction - 1.0) > 0.001:
                self.session.zoom *= correction
        self.ui = UIManager(self.layout, self.session.input_world)
        self.input_handler = InputHandler(self)
        self.ui.menu.resize(w)

    def run(self):
        while self.running:
            self.input_handler.handle_input()
            self.update_physics()
            self.render()
            self.clock.tick()
        if self.root_tk: self.root_tk.destroy()
        pygame.quit()

    def update_physics(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        self.ui.update(dt)
        self.actions.update(dt) # Delegate dialog updates
        
        if not self.session.editor_paused:
            self.session.geo_time += dt

        self.sim.update_constraint_drivers(self.session.geo_time)
        self.sim.apply_constraints()

        if self.session.mode == config.MODE_SIM:
            self.sim.paused = not self.ui.buttons['play'].active
            self.sim.gravity = self.ui.sliders['gravity'].val
            self.sim.target_temp = self.ui.sliders['temp'].val
            self.sim.damping = self.ui.sliders['damping'].val
            self.sim.dt = self.ui.sliders['dt'].val
            self.sim.sigma = self.ui.sliders['sigma'].val
            self.sim.epsilon = self.ui.sliders['epsilon'].val
            self.sim.skin_distance = self.ui.sliders['skin'].val
            self.sim.use_thermostat = self.ui.buttons['thermostat'].active
            self.sim.use_boundaries = self.ui.buttons['boundaries'].active
            
            if not self.sim.paused:
                self.sim.step(int(self.ui.sliders['speed'].val))
        
        if self.session.current_tool:
            if self.session.current_tool.name == "Brush":
                if 'brush_size' in self.ui.sliders:
                    self.session.current_tool.brush_radius = self.ui.sliders['brush_size'].val
            self.session.current_tool.update(dt, self.layout)

    def render(self):
        self.renderer.draw_app(self, self.layout, [])
        self.ui.draw(self.screen, self.font, self.session.mode)
        self.actions.draw_overlays(self.screen, self.font) # Delegate overlay drawing
        pygame.display.flip()

    def change_tool(self, tool_id):
        self.session.change_tool(tool_id) 
        if hasattr(self, 'input_handler'):
            for btn, tid in self.input_handler.tool_btn_map.items(): 
                btn.active = (tid == tool_id)
            self.sound_manager.play_sound('tool_select')

    def exit_editor_mode(self, backup_state=None):
        self.sim.reset_simulation()
        self.sim.sketch.clear()
        self.session.set_status("Reset/Discarded")
        self.sound_manager.play_sound('click')
    
    def save_geo_dialog(self):
        if self.root_tk:
            f = filedialog.asksaveasfilename(defaultextension=".geom", filetypes=[("Geometry Files", "*.geom")])
            if f: 
                self.session.current_geom_filepath = f
                self.session.set_status(file_io.save_geometry_file(self.sim, self.session, f))
    
    def _execute_menu(self, selection):
        self.sound_manager.play_sound('click')
        if selection == "New Simulation":
            subprocess.Popen([sys.executable, "run_instance.py", "sim"])
        elif selection == "New Model":
            subprocess.Popen([sys.executable, "run_instance.py", "editor"])
        elif selection == "New": 
            self.sim.reset_simulation()
            self.sim.sketch.clear()
            self.session.input_world.set_value(config.DEFAULT_WORLD_SIZE)
            self.session.current_sim_filepath = None
            self.session.set_status("New Project Created")
        elif selection == "Import Geometry":
            if self.root_tk:
                f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.geom"), ("All Files", "*.*")])
                if f:
                    data, _ = file_io.load_geometry_file(f)
                    if data: 
                        if hasattr(self.sim.geo, 'place_geometry'):
                            self.session.placing_geo_data = data
                            self.session.set_status("Place Model")
        elif self.root_tk:
            if selection == "Save As..." or (selection == "Save"):
                is_save_as = (selection == "Save As...")
                if is_save_as or not self.session.current_sim_filepath:
                    ext = ".sim" if self.session.mode == config.MODE_SIM else ".mdl"
                    ft = [("Simulation Files", "*.sim")] if ext == ".sim" else [("Model Files", "*.mdl")]
                    f = filedialog.asksaveasfilename(defaultextension=ext, filetypes=ft)
                    if f: self.session.current_sim_filepath = f
                if self.session.current_sim_filepath:
                    self.session.set_status(file_io.save_file(self.sim, self.session, self.session.current_sim_filepath))
            elif selection == "Open...":
                ext = "*.sim" if self.session.mode == config.MODE_SIM else "*.mdl"
                f = filedialog.askopenfilename(filetypes=[("Project Files", ext)])
                if f:
                    self.session.current_sim_filepath = f
                    success, msg, view_state = file_io.load_file(self.sim, f)
                    self.session.set_status(msg)
                    if success: 
                        self.session.input_world.set_value(self.sim.world_size)
                        if view_state:
                            self.session.zoom = view_state['zoom']
                            self.session.pan_x = view_state['pan_x']
                            self.session.pan_y = view_state['pan_y']
        pygame.event.pump()