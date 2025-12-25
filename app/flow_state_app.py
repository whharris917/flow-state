"""
FlowStateApp - The Main Application

This is the application shell that:
- Initializes all systems (Scene, Session, UI, Renderer)
- Runs the main game loop
- Delegates physics updates to Scene.update()
- Handles mode switching and file I/O dialogs
"""

import pygame
pygame.mixer.pre_init(44100, -16, 1, 1024)
pygame.init()

import core.config as config
import core.file_io as file_io
import time
import sys
import subprocess
from tkinter import filedialog, Tk

from core.scene import Scene
from core.session import Session
from ui.ui_widgets import InputField
from ui.renderer import Renderer
from ui.tools import SelectTool, BrushTool, LineTool, RectTool, CircleTool, PointTool
from ui.ui_manager import UIManager
from ui.input_handler import InputHandler
from core.sound_manager import SoundManager
from app.app_controller import AppController

# Import SourceTool for ProcessObject creation
try:
    from ui.source_tool import SourceTool
    HAS_SOURCE_TOOL = True
except ImportError:
    HAS_SOURCE_TOOL = False


class FlowStateApp:
    def __init__(self, start_mode=config.MODE_SIM):
        try:
            self.root_tk = Tk()
            self.root_tk.withdraw()
        except:
            self.root_tk = None

        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), 
            pygame.RESIZABLE
        )
        
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
        # Scene Pattern: Scene owns Sketch, Simulation, Compiler
        # =====================================================================
        is_editor = (start_mode == config.MODE_EDITOR)
        self.scene = Scene(skip_warmup=is_editor)
        
        # Alias for physics parameters access (used in update_physics)
        self.sim = self.scene.simulation
        
        # =====================================================================
        # UI & Input Setup
        # =====================================================================
        
        self.session.input_world = InputField(0, 0, 0, 0)
        self.session.camera.zoom = 0.9
        
        self.session.editor_paused = False 
        self.session.show_constraints = True
        self.session.geo_time = 0.0
        self.last_time = time.time()

        w, h = self.screen.get_size()
        self.init_layout(w, h)
        
        self.ui = UIManager(self.layout, self.session.input_world, mode=self.session.mode)
        
        # Controller handles actions and dialogs
        self.actions = AppController(self)
        self.input_handler = InputHandler(self) 
        
        self.init_tools() 
        if self.session.mode == config.MODE_EDITOR:
            self.change_tool(config.TOOL_SELECT) 
        else:
            self.change_tool(config.TOOL_BRUSH) 

    # =========================================================================
    # Property Accessors (Clean SoC - single source of truth)
    # =========================================================================
    
    @property
    def sketch(self):
        """Access to CAD domain via Scene."""
        return self.scene.sketch
    
    @property
    def geo(self):
        """Access to GeometryManager via Scene."""
        return self.scene.geo

    # =========================================================================
    # Initialization
    # =========================================================================

    def init_layout(self, w, h):
        self.layout = {
            'W': w, 'H': h,
            'LEFT_X': 0, 'LEFT_W': config.PANEL_LEFT_WIDTH,
            'RIGHT_W': config.PANEL_RIGHT_WIDTH, 
            'RIGHT_X': w - config.PANEL_RIGHT_WIDTH,
            'MID_X': config.PANEL_LEFT_WIDTH, 
            'MID_W': w - config.PANEL_LEFT_WIDTH - config.PANEL_RIGHT_WIDTH,
            'MID_H': h - config.TOP_MENU_H
        }

    def init_tools(self):
        """Initialize all available tools."""
        tool_registry = [
            (config.TOOL_SELECT, SelectTool, None),
            (config.TOOL_BRUSH, BrushTool, None),
            (config.TOOL_LINE, LineTool, None),
            (config.TOOL_RECT, RectTool, None),
            (config.TOOL_CIRCLE, CircleTool, None),
            (config.TOOL_POINT, PointTool, None),
            (config.TOOL_REF, LineTool, "Ref Line"), 
        ]
        
        # Add SourceTool if available
        if HAS_SOURCE_TOOL:
            tool_registry.append((config.TOOL_SOURCE, SourceTool, None))
        
        for tid, cls, name in tool_registry:
            self.session.tools[tid] = cls(self) 
            if name:
                self.session.tools[tid].name = name

    def handle_resize(self, w, h):
        old_mid_w = self.layout['MID_W']
        old_mid_h = self.layout['MID_H']
        self.init_layout(w, h)
        if old_mid_w > 0 and old_mid_h > 0:
            new_mid_w = self.layout['MID_W']
            new_mid_h = self.layout['MID_H']
            correction = (old_mid_w / old_mid_h) / (new_mid_w / new_mid_h)
            if abs(correction - 1.0) > 0.001:
                self.session.camera.zoom *= correction
        
        self.ui = UIManager(self.layout, self.session.input_world, mode=self.session.mode)
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
        if self.root_tk:
            self.root_tk.destroy()
        pygame.quit()

    def update_physics(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Update UI widgets
        self.ui.update(dt)
        self.actions.update(dt)
        
        # Update geometry time (for animations)
        if not self.session.editor_paused:
            self.session.geo_time += dt

        # Update physics parameters from UI sliders (Sim mode only)
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
            physics_steps = 1
        
        # === SCENE ORCHESTRATOR ===
        # Scene.update() handles: drivers → solve → rebuild → process_objects → physics
        run_physics = (self.session.mode == config.MODE_SIM and not self.sim.paused)
        self.scene.update(dt, self.session.geo_time, run_physics, physics_steps)
        
        # Update current tool
        if self.session.current_tool:
            if self.session.current_tool.name == "Brush":
                if 'brush_size' in self.ui.sliders:
                    self.session.current_tool.brush_radius = self.ui.sliders['brush_size'].val
            self.session.current_tool.update(dt, self.layout)

    def render(self):
        self.renderer.draw_app(self, self.layout, [])
        self.ui.draw(self.screen, self.font, self.session.mode)
        self.actions.draw_overlays(self.screen, self.font)
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

    def switch_mode(self, new_mode):
        """Switch between SIM and EDITOR modes."""
        if self.session.mode == new_mode:
            return
        
        old_mode = self.session.mode
        
        # Store camera for old mode
        self.session.store_camera_for_mode(old_mode)
        
        # Clear transient state
        self.session.clear_interaction_state()
        
        # Update mode
        self.session.mode = new_mode
        
        # Restore camera for new mode
        self.session.restore_camera_for_mode(new_mode)
        
        # Rebuild UI for new mode
        w, h = self.screen.get_size()
        self.ui = UIManager(self.layout, self.session.input_world, mode=new_mode)
        self.input_handler = InputHandler(self)
        
        # Switch to appropriate default tool
        if new_mode == config.MODE_SIM:
            self.change_tool(config.TOOL_BRUSH)
            pygame.display.set_caption("Flow State - Simulation")
            self.sound_manager.play_music('Whimsical')
        else:
            self.change_tool(config.TOOL_SELECT)
            pygame.display.set_caption("Flow State - Model Builder")
            self.sound_manager.play_music('Jungle')
        
        self.session.status.set(f"Mode: {'Simulation' if new_mode == config.MODE_SIM else 'Editor'}")

    def enter_editor_mode(self):
        """Enter editor mode, optionally backing up simulation state."""
        if self.session.mode == config.MODE_EDITOR:
            return
        
        # Backup current simulation state
        self.session.sim_backup_state = self.sim.snapshot_state()
        self.session.was_sim_running = not self.sim.paused
        self.sim.paused = True
        
        self.switch_mode(config.MODE_EDITOR)

    def exit_editor_mode(self, restore_sim=True):
        """Exit editor mode, optionally restoring simulation state."""
        if self.session.mode == config.MODE_SIM:
            return
        
        if restore_sim and self.session.sim_backup_state:
            self.sim.restore_state(self.session.sim_backup_state)
            self.sim.paused = not self.session.was_sim_running
        
        self.session.sim_backup_state = None
        self.switch_mode(config.MODE_SIM)

    # =========================================================================
    # File Operations
    # =========================================================================

    def new_scene(self):
        """Create a new empty scene."""
        self.scene = Scene(skip_warmup=(self.session.mode == config.MODE_EDITOR))
        self.sim = self.scene.simulation
        self.session.current_sim_filepath = None
        self.session.current_geom_filepath = None
        self.session.status.set("New scene created")

    def save_scene(self, filepath=None):
        """Save the current scene."""
        if not filepath:
            if self.root_tk:
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".scn",
                    filetypes=[("Scene files", "*.scn"), ("All files", "*.*")]
                )
            if not filepath:
                return
        
        view_state = {
            'zoom': self.session.camera.zoom,
            'pan_x': self.session.camera.pan_x,
            'pan_y': self.session.camera.pan_y
        }
        
        msg = self.scene.save_scene(filepath, view_state)
        self.session.current_sim_filepath = filepath
        self.session.status.set(msg)

    def load_scene(self, filepath=None):
        """Load a scene from file."""
        if not filepath:
            if self.root_tk:
                filepath = filedialog.askopenfilename(
                    filetypes=[("Scene files", "*.scn"), ("All files", "*.*")]
                )
            if not filepath:
                return
        
        new_scene, view_state, msg = Scene.load_scene(
            filepath, 
            skip_warmup=(self.session.mode == config.MODE_EDITOR)
        )
        
        if new_scene:
            self.scene = new_scene
            self.sim = self.scene.simulation
            
            if view_state:
                self.session.camera.zoom = view_state.get('zoom', 1.0)
                self.session.camera.pan_x = view_state.get('pan_x', 0.0)
                self.session.camera.pan_y = view_state.get('pan_y', 0.0)
            
            self.session.current_sim_filepath = filepath
        
        self.session.status.set(msg)

    def save_geometry(self, filepath=None):
        """Save just the geometry (model)."""
        if not filepath:
            if self.root_tk:
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".mdl",
                    filetypes=[("Model files", "*.mdl"), ("All files", "*.*")]
                )
            if not filepath:
                return
        
        msg = self.scene.save_model(filepath)
        self.session.current_geom_filepath = filepath
        self.session.status.set(msg)

    def load_geometry(self, filepath=None):
        """Load geometry from file."""
        if not filepath:
            if self.root_tk:
                filepath = filedialog.askopenfilename(
                    filetypes=[("Model files", "*.mdl"), ("Geometry files", "*.geom"), ("All files", "*.*")]
                )
            if not filepath:
                return
        
        msg = self.scene.load_model(filepath)
        self.session.current_geom_filepath = filepath
        self.session.status.set(msg)

    def import_geometry(self, filepath=None):
        """Import geometry (append to current scene)."""
        if not filepath:
            if self.root_tk:
                filepath = filedialog.askopenfilename(
                    filetypes=[("Model files", "*.mdl"), ("Geometry files", "*.geom"), ("All files", "*.*")]
                )
            if not filepath:
                return
        
        msg = self.scene.import_model(filepath)
        self.session.status.set(msg)
