"""
UIManager - UI Element Organization

Separates the definition of UI widgets from application logic.
Handles layout, element creation, update, and draw cycles.
"""

import core.config as config

from ui.ui_widgets import SmartSlider, Button, InputField, MenuBar
from ui import icons


class UIManager:
    """
    Helper to organize UI elements.
    Separates the definition of UI widgets from the application logic.
    """
    def __init__(self, layout, app_input_world=None, mode=config.MODE_SIM):
        self.sliders = {}
        self.buttons = {}
        self.tools = {}
        self.inputs = {}
        self.mode = mode
        self.menu = MenuBar(layout['W'], config.TOP_MENU_H)
        
        self.menu.items["File"] = [
            "New Simulation", 
            "New Model", 
            "---",
            "Open...", 
            "Save", 
            "Save As...", 
            "---", 
            "Import Geometry"
        ] 
        
        self._init_elements(layout)
        
        if app_input_world:
            self.inputs['world'] = app_input_world
        else:
            # Fallback if input not provided
            rp_x = layout['RIGHT_X'] + 15
            base_y = config.TOP_MENU_H + 20
            rp_y_for_input = base_y + 800 if self.mode == config.MODE_SIM else base_y + 700
            self.inputs['world'] = InputField(rp_x + 80, rp_y_for_input, 60, 25, str(config.DEFAULT_WORLD_SIZE))

    def _init_elements(self, layout):
        # --- LEFT PANEL: Physics Controls ---
        # ONLY initialize these if we are in Simulation Mode
        if self.mode == config.MODE_SIM:
            lp_x = layout['LEFT_X'] + 15 
            lp_w = layout['LEFT_W'] - 30
            lp_y = config.TOP_MENU_H + 20
            
            self.buttons['play'] = Button(lp_x, lp_y, lp_w, 35, "Play/Pause", active=False, color_active=config.COLOR_SUCCESS, color_inactive=config.COLOR_DANGER); lp_y += 50
            self.buttons['clear'] = Button(lp_x, lp_y, lp_w, 35, "Clear Particles", active=False, toggle=False); lp_y += 50
            self.buttons['reset'] = Button(lp_x, lp_y, lp_w, 35, "Reset All", active=False, toggle=False); lp_y += 50
            
            lp_y += 10
            
            self.sliders['gravity'] = SmartSlider(lp_x, lp_y, lp_w, 0.0, 50.0, config.DEFAULT_GRAVITY, "Gravity", hard_min=0.0); lp_y += 60
            self.sliders['temp'] = SmartSlider(lp_x, lp_y, lp_w, 0.0, 5.0, 0.5, "Temperature", hard_min=0.0); lp_y += 60
            self.sliders['damping'] = SmartSlider(lp_x, lp_y, lp_w, 0.90, 1.0, config.DEFAULT_DAMPING, "Damping", hard_min=0.0, hard_max=1.0); lp_y += 60
            self.sliders['dt'] = SmartSlider(lp_x, lp_y, lp_w, 0.0001, 0.01, config.DEFAULT_DT, "Time Step (dt)", hard_min=0.00001); lp_y += 60
            
            self.sliders['speed'] = SmartSlider(lp_x, lp_y, lp_w, 1.0, 100.0, float(config.DEFAULT_DRAW_M), "Speed (Steps/Frame)", hard_min=1.0); lp_y += 60
            
            self.sliders['sigma'] = SmartSlider(lp_x, lp_y, lp_w, 0.5, 2.0, config.ATOM_SIGMA, "Sigma (Size)", hard_min=0.1); lp_y += 60
            self.sliders['epsilon'] = SmartSlider(lp_x, lp_y, lp_w, 0.1, 5.0, config.ATOM_EPSILON, "Epsilon (Strength)", hard_min=0.0); lp_y += 60
            self.sliders['skin'] = SmartSlider(lp_x, lp_y, lp_w, 0.1, 2.0, config.DEFAULT_SKIN_DISTANCE, "Skin Distance", hard_min=0.05); lp_y += 60
            
            btn_half_left = (lp_w - 10) // 2
            self.buttons['thermostat'] = Button(lp_x, lp_y, btn_half_left, 30, "Thermostat", active=False)
            self.buttons['boundaries'] = Button(lp_x + btn_half_left + 10, lp_y, btn_half_left, 30, "Bounds", active=False); lp_y += 45

            self.buttons['undo'] = Button(lp_x, lp_y, btn_half_left, 30, "Undo", active=False, toggle=False)
            self.buttons['redo'] = Button(lp_x + btn_half_left + 10, lp_y, btn_half_left, 30, "Redo", active=False, toggle=False)

        # --- RIGHT PANEL: Editor Tools (Always Needed) ---
        rp_x = layout['RIGHT_X'] + 15
        rp_w = layout['RIGHT_W'] - 30
        rp_y = config.TOP_MENU_H + 20
        
        # Mode toggle - keep as text button (changes text dynamically)
        self.buttons['mode_ghost'] = Button(rp_x, rp_y, rp_w, 35, "Mode: Physical", active=False, color_active=config.COLOR_ACCENT, color_inactive=config.COLOR_SUCCESS); rp_y += 45
        
        # Atomize - icon button (fixed 32x32 icon centered in wide button)
        self.buttons['atomize'] = Button(rp_x, rp_y, rp_w, 35, 
                                         icon=icons.get_icon_fixed_size('atomize', 32), 
                                         tooltip="Atomize Selected",
                                         active=False, toggle=False); rp_y += 45

        # --- TOOL ICONS (2-column grid) ---
        btn_size = 38  # Square icon buttons
        spacing = 6
        
        # Row 1: Brush, Select
        self.tools['brush'] = Button(rp_x, rp_y, btn_size, btn_size, 
                                     icon=icons.get_icon('brush'), 
                                     tooltip="Brush (B)",
                                     active=True, toggle=False)
        self.tools['select'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                      icon=icons.get_icon('select'),
                                      tooltip="Select (V)",
                                      active=False, toggle=False)
        rp_y += btn_size + spacing
        
        # Row 2: Line, Rectangle
        self.tools['line'] = Button(rp_x, rp_y, btn_size, btn_size,
                                    icon=icons.get_icon('line'),
                                    tooltip="Line (L)",
                                    active=False, toggle=False)
        self.tools['rect'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                    icon=icons.get_icon('rect'),
                                    tooltip="Rectangle (R)",
                                    active=False, toggle=False)
        rp_y += btn_size + spacing
        
        # Row 3: Circle, Point
        self.tools['circle'] = Button(rp_x, rp_y, btn_size, btn_size,
                                      icon=icons.get_icon('circle'),
                                      tooltip="Circle (C)",
                                      active=False, toggle=False)
        self.tools['point'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                     icon=icons.get_icon('point'),
                                     tooltip="Point (P)",
                                     active=False, toggle=False)
        rp_y += btn_size + spacing
        
        # Row 4: Ref Line (single button)
        self.tools['ref'] = Button(rp_x, rp_y, btn_size, btn_size,
                                   icon=icons.get_icon('ref_line'),
                                   tooltip="Reference Line",
                                   active=False, toggle=False)
        rp_y += btn_size + 10
        
        # Brush size slider
        self.sliders['brush_size'] = SmartSlider(rp_x, rp_y, rp_w, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5); rp_y += 60
        
        # Animation/Constraint visibility controls (icon buttons)
        self.buttons['editor_play'] = Button(rp_x, rp_y, btn_size, btn_size,
                                             icon=icons.get_icon('anim_pause'),
                                             tooltip="Pause Animation",
                                             active=False, toggle=False)
        self.buttons['show_const'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                            icon=icons.get_icon('hide'),
                                            tooltip="Hide Constraints",
                                            active=False, toggle=False)
        rp_y += btn_size + spacing
        
        # Extend button - icon (fixed 32x32 icon centered in wide button)
        self.buttons['extend'] = Button(rp_x, rp_y, rp_w, 35,
                                        icon=icons.get_icon_fixed_size('extend', 32),
                                        tooltip="Extend Infinite Line",
                                        toggle=False); rp_y += 45

        # --- CONSTRAINT ICONS (grid layout) ---
        # Row 1: Coincident, Collinear
        self.buttons['const_coincident'] = Button(rp_x, rp_y, btn_size, btn_size,
                                                  icon=icons.get_icon('coincident'),
                                                  tooltip="Coincident",
                                                  toggle=False)
        self.buttons['const_collinear'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                                 icon=icons.get_icon('collinear'),
                                                 tooltip="Collinear",
                                                 toggle=False)
        rp_y += btn_size + spacing
        
        # Row 2: Midpoint, Length
        self.buttons['const_midpoint'] = Button(rp_x, rp_y, btn_size, btn_size,
                                                icon=icons.get_icon('midpoint'),
                                                tooltip="Midpoint",
                                                toggle=False)
        self.buttons['const_length'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                              icon=icons.get_icon('length'),
                                              tooltip="Fix Length",
                                              toggle=False)
        rp_y += btn_size + spacing
        
        # Row 3: Equal, Parallel
        self.buttons['const_equal'] = Button(rp_x, rp_y, btn_size, btn_size,
                                             icon=icons.get_icon('equal'),
                                             tooltip="Equal Length",
                                             toggle=False)
        self.buttons['const_parallel'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                                icon=icons.get_icon('parallel'),
                                                tooltip="Parallel",
                                                toggle=False)
        rp_y += btn_size + spacing
        
        # Row 4: Perpendicular, Angle
        self.buttons['const_perp'] = Button(rp_x, rp_y, btn_size, btn_size,
                                            icon=icons.get_icon('perpendicular'),
                                            tooltip="Perpendicular",
                                            toggle=False)
        self.buttons['const_angle'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                             icon=icons.get_icon('angle'),
                                             tooltip="Angle",
                                             toggle=False)
        rp_y += btn_size + spacing
        
        # Row 5: Horizontal, Vertical
        self.buttons['const_horiz'] = Button(rp_x, rp_y, btn_size, btn_size,
                                             icon=icons.get_icon('horizontal'),
                                             tooltip="Horizontal",
                                             toggle=False)
        self.buttons['const_vert'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                            icon=icons.get_icon('vertical'),
                                            tooltip="Vertical",
                                            toggle=False)
        rp_y += btn_size + 15

        # --- SAVE/EXIT BUTTONS (icons) ---
        self.buttons['save_geo'] = Button(rp_x, rp_y, btn_size, btn_size,
                                          icon=icons.get_icon('save'),
                                          tooltip="Save",
                                          active=False, toggle=False, 
                                          color_inactive=config.COLOR_SUCCESS)
        self.buttons['discard_geo'] = Button(rp_x + btn_size + spacing, rp_y, btn_size, btn_size,
                                             icon=icons.get_icon('exit'),
                                             tooltip="Exit Editor",
                                             active=False, toggle=False, 
                                             color_inactive=config.COLOR_DANGER)
        rp_y += btn_size + 10
        
        # Resize world (keep as text - has input field)
        self.buttons['resize'] = Button(rp_x, rp_y, rp_w - 70, 25, "Resize World", active=False, toggle=False)

    def update(self, dt):
        """
        Propagate time step to all widgets for smooth animations.
        """
        for b in self.buttons.values():
            if b: b.update(dt)
        for t in self.tools.values():
            if t: t.update(dt)
        for s in self.sliders.values():
            if s: s.update(dt)
        for i in self.inputs.values():
            if i: i.update(dt)

    def draw(self, screen, font, mode):
        # Update text for Ghost/Physical toggle
        if self.buttons.get('mode_ghost'):
            if self.buttons['mode_ghost'].active:
                self.buttons['mode_ghost'].text = "Mode: Ghost (Blueprint)"
            else:
                self.buttons['mode_ghost'].text = "Mode: Physical (Live)"
            self.buttons['mode_ghost'].cached_surf = None

        physics_elements = []
        if self.mode == config.MODE_SIM:
            physics_elements = [
                self.buttons.get('play'), self.buttons.get('clear'), self.buttons.get('reset'),
                self.buttons.get('undo'), self.buttons.get('redo'),
                self.buttons.get('thermostat'), self.buttons.get('boundaries'),
                self.sliders.get('gravity'), self.sliders.get('temp'), self.sliders.get('damping'),
                self.sliders.get('dt'), self.sliders.get('speed'), 
                self.sliders.get('sigma'), self.sliders.get('epsilon'), self.sliders.get('skin')
            ]
        
        editor_elements = [
            self.buttons.get('mode_ghost'), self.buttons.get('atomize'),
            self.buttons.get('save_geo'), self.buttons.get('discard_geo'), 
            self.buttons.get('editor_play'), self.buttons.get('show_const'),
            *self.tools.values(), 
            self.sliders.get('brush_size'),
            self.buttons.get('const_coincident'), self.buttons.get('const_collinear'), self.buttons.get('const_midpoint'),
            self.buttons.get('const_length'), self.buttons.get('const_equal'), self.buttons.get('const_parallel'),
            self.buttons.get('const_perp'), self.buttons.get('const_horiz'), self.buttons.get('const_vert'),
            self.buttons.get('const_angle'), self.buttons.get('extend'),
            self.buttons.get('resize'), self.inputs.get('world')
        ]
        
        # Filter None and Combine
        active_list = [el for el in physics_elements + editor_elements if el is not None]
        
        for el in active_list:
            if el == self.inputs.get('world') and el:
                screen.blit(font.render("Size:", True, config.COLOR_TEXT), 
                             (el.rect.x - 40, el.rect.y + 4))
            el.draw(screen, font)

        self.menu.draw(screen, font)