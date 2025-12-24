import pygame
pygame.mixer.pre_init(44100, -16, 1, 1024)
pygame.init()

import core.config as config

import core.utils as utils

import core.file_io as file_io

import time
import sys
import subprocess
from tkinter import filedialog, Tk

# Phase 4: Import Scene instead of Simulation directly
from core.scene import Scene
from core.session import Session
from ui.ui_widgets import InputField
# REMOVED: from model.geometry import Line, Circle  # Unused imports (SoC violation)
from ui.renderer import Renderer
from ui.tools import SelectTool, BrushTool, LineTool, RectTool, CircleTool, PointTool
from ui.ui_manager import UIManager
from ui.input_handler import InputHandler
from core.sound_manager import SoundManager
from app.app_controller import AppController

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
        
        # =====================================================================
        # Phase 4: Scene Pattern Integration
        # =====================================================================
        # Create Scene as the document container
        is_editor = (start_mode == config.MODE_EDITOR)
        self.scene = Scene(skip_warmup=is_editor)
        
        # Create aliases for backward compatibility
        # All existing code uses self.sim and self.sketch - these aliases
        # ensure everything continues to work without modification
        self.sim = self.scene.simulation
        self.sketch = self.scene.sketch
        # =====================================================================
        
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

    # =========================================================================
    # Property Aliases (Backward Compatibility)
    # =========================================================================
    
    @property
    def walls(self):
        return self.sim.walls
    
    @property
    def constraints(self):
        return self.sim.constraints
    
    @property
    def geo(self):
        # Phase 6.5b: GeometryManager is now owned by Scene
        return self.scene.geo

    # =========================================================================
    # Initialization
    # =========================================================================

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

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        while self.running:
            self.input_handler.handle_input()
            self.update_physics()
            self.render()
            self.clock.tick()
        if self.root_tk: self.root_tk.destroy()
        pygame.quit()

    def update_physics(self):
        """
        Main frame update - delegates to Scene.update() for proper orchestration.
        
        Phase 7 Fix: Now uses the Orchestrator pattern via scene.update()
        instead of manually calling individual methods.
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Update UI widgets
        self.ui.update(dt)
        self.actions.update(dt)
        
        # Update geometry animation time (unless paused)
        if not self.session.editor_paused:
            self.session.geo_time += dt

        # =====================================================================
        # Phase 7: Use Orchestrator Pattern
        # =====================================================================
        # Update simulation parameters from UI BEFORE orchestration
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
            
            physics_steps = int(self.ui.sliders['speed'].val)
        else:
            # Editor mode: no physics, but still update geometry/constraints
            physics_steps = 0
        
        # Delegate to Scene orchestrator for correct update ordering:
        # 1. Update constraint drivers (animations)
        # 2. Solve constraints (geometry moves)
        # 3. Rebuild static atoms if geometry changed
        # 4. Run physics step (if enabled)
        run_physics = (self.session.mode == config.MODE_SIM)
        self.scene.update(dt, self.session.geo_time, run_physics, physics_steps)
        # =====================================================================
        
        # Update tool state
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

    # =========================================================================
    # Tool Management
    # =========================================================================

    def change_tool(self, tool_id):
        self.session.change_tool(tool_id) 
        if hasattr(self, 'input_handler'):
            for btn, tid in self.input_handler.tool_btn_map.items(): 
                btn.active = (tid == tool_id)
            self.sound_manager.play_sound('tool_select')

    # =========================================================================
    # Mode/State Management
    # =========================================================================

    def exit_editor_mode(self, backup_state=None):
        # Phase 4: Use scene.new() for clean reset
        self.scene.new()
        self.session.set_status("Reset/Discarded")
        self.sound_manager.play_sound('click')
    
    # =========================================================================
    # File I/O Dialogs
    # =========================================================================

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
            # Phase 4: Use scene.new() for clean reset
            self.scene.new()
            self.session.input_world.set_value(config.DEFAULT_WORLD_SIZE)
            self.session.current_sim_filepath = None
            self.session.set_status("New Project Created")
        elif selection == "Import Geometry":
            if self.root_tk:
                f = filedialog.askopenfilename(filetypes=[("Geometry Files", "*.geom"), ("All Files", "*.*")])
                if f:
                    data, _ = file_io.load_geometry_file(f)
                    if data: 
                        # Phase 6.5b: Use self.geo (now from scene.geo)
                        if self.geo and hasattr(self.geo, 'place_geometry'):
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