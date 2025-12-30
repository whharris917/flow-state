"""
UIManager - UI Element Organization (Phase IV: Final Integration)

Constructs the Hierarchical Container Tree.
Root -> [Menu Bar, Middle Region, Status Bar]
"""

import pygame
import core.config as config
from ui.ui_widgets import UIContainer, UIElement, Button, SmartSlider, InputField, MenuBar, StatusBar, ScrollableContainer, MaterialPropertyWidget
from ui import icons
from core.session import InteractionState

class SceneViewport(UIElement):
    """
    A UIElement wrapper for the Scene.
    """
    def __init__(self, x, y, w, h, controller):
        super().__init__(x, y, w, h)
        self.controller = controller
        self.session = controller.session

    def handle_event(self, event):
        if not self.visible: return False
        
        mouse_pos = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            
        if mouse_pos:
            if not self.rect.collidepoint(mouse_pos):
                return False

        # Tool Delegation
        if self.session.current_tool:
            if self.session.current_tool.handle_event(event, self.controller.layout):
                return True 

        # Camera Panning
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            self.session.state = InteractionState.PANNING
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            if self.session.state == InteractionState.PANNING:
                self.session.state = InteractionState.IDLE
            return True
        elif event.type == pygame.MOUSEMOTION:
            if self.session.state == InteractionState.PANNING:
                self.session.camera.apply_pan(event.rel[0], event.rel[1])
                return True
        
        # Zooming
        elif event.type == pygame.MOUSEWHEEL:
            self.session.camera.apply_zoom(event.y)
            return True
        
        # Context Menu
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.controller.actions.spawn_context_menu(event.pos)
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            return True

        return False


class UIManager:
    """
    Orchestrates the UI Layout Tree.
    """
    def __init__(self, layout, app_input_world=None, mode=config.MODE_SIM, controller=None):
        self.layout = layout
        self.mode = mode
        self.controller = controller
        
        # Scale Helper
        s = config.scale
        
        # --- FLAT REFERENCES ---
        self.sliders = {}
        self.buttons = {}
        self.tools = {}
        self.inputs = {}
        
        # --- HIERARCHICAL TREE CONSTRUCTION ---
        
        # Level 0: Root
        self.root = UIContainer(0, 0, layout['W'], layout['H'], layout_type='free')
        
        # Level 1: Vertical Stack (Menu, Middle, Status)
        
        # 1. Top Menu
        self.menu = MenuBar(layout['W'], config.TOP_MENU_H)
        self.menu.items["File"] = [
            "New Simulation", "New Model", "---",
            "Open...", "Save", "Save As...", "---", 
            "Import Geometry"
        ]
        self.root.add_child(self.menu)
        
        # 2. Middle Region (Workspace)
        # Calculates available height between Top Menu and Bottom Status Bar
        mid_y = config.TOP_MENU_H
        status_bar_h = s(30)
        mid_h = layout['H'] - mid_y - status_bar_h
        self.middle_region = UIContainer(0, mid_y, layout['W'], mid_h, layout_type='free')
        self.root.add_child(self.middle_region)
        
        # 3. Status Bar (Footer)
        if self.controller:
            self.status_bar = StatusBar(0, layout['H'] - status_bar_h, layout['W'], status_bar_h, self.controller.session)
            self.root.add_child(self.status_bar)
        
        # --- LEVEL 2: MIDDLE REGION SUBDIVISION ---
        
        # A. Left Panel
        self.left_panel = ScrollableContainer(
            layout['LEFT_X'], mid_y, layout['LEFT_W'], mid_h,
            layout_type='vertical', padding=s(15), spacing=s(15),
            bg_color=config.PANEL_BG_COLOR, border_color=config.PANEL_BORDER_COLOR
        )
        self.middle_region.add_child(self.left_panel)
            
        # B. Scene Viewport
        scene_x = layout.get('MID_X', 0)
        scene_w = layout.get('MID_W', layout['W'])
        
        if self.controller:
            self.scene_viewport = SceneViewport(scene_x, mid_y, scene_w, mid_h, self.controller)
            self.middle_region.add_child(self.scene_viewport)
            
        # C. Right Panel
        self.right_panel = ScrollableContainer(
            layout['RIGHT_X'], mid_y, layout['RIGHT_W'], mid_h,
            layout_type='vertical', padding=s(15), spacing=s(15),
            bg_color=config.PANEL_BG_COLOR, border_color=config.PANEL_BORDER_COLOR
        )
        self.middle_region.add_child(self.right_panel)

        # --- POPULATE WIDGETS ---
        self._init_elements(app_input_world)

    def _init_elements(self, app_input_world):
        """Populate the panels with widgets."""
        s = config.scale
        
        # --- LEFT PANEL: Physics Controls ---
        if self.mode == config.MODE_SIM:
            def add_sim_btn(key, text, **kwargs):
                btn = Button(0, 0, self.left_panel.rect.w - s(30), s(35), text, **kwargs)
                self.left_panel.add_child(btn)
                self.buttons[key] = btn
                return btn

            add_sim_btn('play', "Play/Pause", active=False, color_active=config.COLOR_SUCCESS, color_inactive=config.COLOR_DANGER)
            add_sim_btn('clear', "Clear Particles", active=False, toggle=False)
            add_sim_btn('reset', "Reset All", active=False, toggle=False)

            def add_slider(key, label, min_v, max_v, init_v, **kwargs):
                sld = SmartSlider(0, 0, self.left_panel.rect.w - s(30), min_v, max_v, init_v, label, **kwargs)
                self.left_panel.add_child(sld)
                self.sliders[key] = sld
                return sld

            add_slider('gravity', "Gravity", 0.0, 50.0, config.DEFAULT_GRAVITY, hard_min=0.0)
            add_slider('temp', "Temperature", 0.0, 5.0, 0.5, hard_min=0.0)
            add_slider('damping', "Damping", 0.90, 1.0, config.DEFAULT_DAMPING, hard_min=0.0, hard_max=1.0)
            add_slider('dt', "Time Step (dt)", 0.0001, 0.01, config.DEFAULT_DT, hard_min=0.00001)
            add_slider('speed', "Speed (Steps/Frame)", 1.0, 100.0, float(config.DEFAULT_DRAW_M), hard_min=1.0)
            add_slider('skin', "Skin Distance", 0.1, 2.0, config.DEFAULT_SKIN_DISTANCE, hard_min=0.05)

            row_container = UIContainer(0, 0, self.left_panel.rect.w - s(30), s(30), layout_type='horizontal', padding=0, spacing=s(10))
            self.left_panel.add_child(row_container)
            
            btn_w = (row_container.rect.w - s(10)) // 2
            btn_therm = Button(0, 0, btn_w, s(30), "Thermostat", active=False)
            btn_bound = Button(0, 0, btn_w, s(30), "Bounds", active=False)
            row_container.add_child(btn_therm)
            row_container.add_child(btn_bound)
            self.buttons['thermostat'] = btn_therm
            self.buttons['boundaries'] = btn_bound
            
            row_undo = UIContainer(0, 0, self.left_panel.rect.w - s(30), s(30), layout_type='horizontal', padding=0, spacing=s(10))
            self.left_panel.add_child(row_undo)
            
            btn_undo = Button(0, 0, btn_w, s(30), "Undo", active=False, toggle=False)
            btn_redo = Button(0, 0, btn_w, s(30), "Redo", active=False, toggle=False)
            row_undo.add_child(btn_undo)
            row_undo.add_child(btn_redo)
            self.buttons['undo'] = btn_undo
            self.buttons['redo'] = btn_redo

        # --- RIGHT PANEL: Editor Tools ---
        rp = self.right_panel
        rp_w = rp.rect.w - s(30)

        btn_mode = Button(0, 0, rp_w, s(35), "Mode: Physical", active=False, color_active=config.COLOR_ACCENT, color_inactive=config.COLOR_SUCCESS)
        rp.add_child(btn_mode)
        self.buttons['mode_ghost'] = btn_mode
        
        btn_atom = Button(0, 0, rp_w, s(35), icon=icons.get_icon_fixed_size('atomize', s(32)), tooltip="Atomize Selected", active=False, toggle=False)
        rp.add_child(btn_atom)
        self.buttons['atomize'] = btn_atom

        btn_size = s(38)
        spacing = s(6)
        
        def add_tool_row(t1_key, t1_icon, t1_tip, t2_key, t2_icon, t2_tip):
            row = UIContainer(0, 0, rp_w, btn_size, layout_type='horizontal', padding=0, spacing=spacing)
            rp.add_child(row)
            b1 = Button(0, 0, btn_size, btn_size, icon=icons.get_icon(t1_icon), tooltip=t1_tip, active=(t1_key=='brush'), toggle=False)
            b2 = Button(0, 0, btn_size, btn_size, icon=icons.get_icon(t2_icon), tooltip=t2_tip, active=False, toggle=False)
            row.add_child(b1)
            row.add_child(b2)
            self.tools[t1_key] = b1
            self.tools[t2_key] = b2
        
        add_tool_row('brush', 'brush', "Brush (B)", 'select', 'select', "Select (V)")
        add_tool_row('line', 'line', "Line (L)", 'rect', 'rect', "Rectangle (R)")
        add_tool_row('circle', 'circle', "Circle (C)", 'point', 'point', "Point (P)")
        add_tool_row('ref', 'ref_line', "Reference Line", 'source', 'source', "Source (Emitter)")

        sld_brush = SmartSlider(0, 0, rp_w, 1.0, 10.0, 2.0, "Brush Radius", hard_min=0.5)
        rp.add_child(sld_brush)
        self.sliders['brush_size'] = sld_brush

        # Material Property Widget (dropdown + sliders for sigma, epsilon, mass)
        if self.controller:
            self.material_widget = MaterialPropertyWidget(0, 0, rp_w, self.controller.session)
            rp.add_child(self.material_widget)

        row_vis = UIContainer(0, 0, rp_w, btn_size, layout_type='horizontal', padding=0, spacing=spacing)
        rp.add_child(row_vis)
        btn_play = Button(0, 0, btn_size, btn_size, icon=icons.get_icon('anim_pause'), tooltip="Pause Animation", active=False, toggle=False)
        btn_hide = Button(0, 0, btn_size, btn_size, icon=icons.get_icon('hide'), tooltip="Hide Constraints", active=False, toggle=False)
        row_vis.add_child(btn_play)
        row_vis.add_child(btn_hide)
        self.buttons['editor_play'] = btn_play
        self.buttons['show_const'] = btn_hide
        
        btn_ext = Button(0, 0, rp_w, s(35), icon=icons.get_icon_fixed_size('extend', s(32)), tooltip="Extend Infinite Line", toggle=False)
        rp.add_child(btn_ext)
        self.buttons['extend'] = btn_ext
        
        def add_const_row(k1, i1, tip1, k2, i2, tip2):
            row = UIContainer(0, 0, rp_w, btn_size, layout_type='horizontal', padding=0, spacing=spacing)
            rp.add_child(row)
            b1 = Button(0, 0, btn_size, btn_size, icon=icons.get_icon(i1), tooltip=tip1, toggle=False)
            b2 = Button(0, 0, btn_size, btn_size, icon=icons.get_icon(i2), tooltip=tip2, toggle=False)
            row.add_child(b1)
            row.add_child(b2)
            self.buttons[k1] = b1
            self.buttons[k2] = b2

        add_const_row('const_coincident', 'coincident', "Coincident", 'const_collinear', 'collinear', "Collinear")
        add_const_row('const_midpoint', 'midpoint', "Midpoint", 'const_length', 'length', "Fix Dimension")
        add_const_row('const_equal', 'equal', "Equal Length", 'const_parallel', 'parallel', "Parallel")
        add_const_row('const_perp', 'perpendicular', "Perpendicular", 'const_angle', 'angle', "Angle")
        add_const_row('const_horiz', 'horizontal', "Horizontal", 'const_vert', 'vertical', "Vertical")

        row_io = UIContainer(0, 0, rp_w, btn_size, layout_type='horizontal', padding=0, spacing=spacing)
        rp.add_child(row_io)
        btn_save = Button(0, 0, btn_size, btn_size, icon=icons.get_icon('save'), tooltip="Save", active=False, toggle=False, color_inactive=config.COLOR_SUCCESS)
        btn_exit = Button(0, 0, btn_size, btn_size, icon=icons.get_icon('exit'), tooltip="Exit Editor", active=False, toggle=False, color_inactive=config.COLOR_DANGER)
        row_io.add_child(btn_save)
        row_io.add_child(btn_exit)
        self.buttons['save_geo'] = btn_save
        self.buttons['discard_geo'] = btn_exit
        
        if app_input_world:
            self.inputs['world'] = app_input_world
        else:
            self.inputs['world'] = InputField(0, 0, s(60), s(25), str(config.DEFAULT_WORLD_SIZE))
        
        row_resize = UIContainer(0, 0, rp_w, s(25), layout_type='horizontal', padding=0, spacing=s(10))
        rp.add_child(row_resize)
        
        btn_resize = Button(0, 0, rp_w - s(70), s(25), "Resize World", active=False, toggle=False)
        row_resize.add_child(btn_resize)
        self.buttons['resize'] = btn_resize
        row_resize.add_child(self.inputs['world'])


    def update(self, dt):
        """
        Propagate time step to all widgets via the Tree.
        """
        self.root.update(dt)

    def draw(self, screen, font, mode):
        """
        Draw the UI Tree.
        """
        self.root.draw(screen, font)

        # Draw floating overlays (dropdown expanded lists) on top of everything
        self._draw_overlays(screen, font)

    def _draw_overlays(self, screen, font):
        """Draw elements that should appear above everything else (dropdowns, etc.)."""
        # Draw MaterialPropertyWidget's dropdown overlay if expanded
        if hasattr(self, 'material_widget') and self.material_widget.dropdown.expanded:
            dropdown = self.material_widget.dropdown
            if dropdown.options:
                # Draw the expanded list manually (bypasses parent clip)
                expanded_rect = dropdown.get_expanded_rect()

                # Shadow
                shadow_rect = expanded_rect.copy()
                shadow_rect.x += 3
                shadow_rect.y += 3
                pygame.draw.rect(screen, (0, 0, 0, 80), shadow_rect, border_radius=4)

                # Background
                pygame.draw.rect(screen, config.PANEL_BG_COLOR, expanded_rect, border_radius=4)
                pygame.draw.rect(screen, config.COLOR_ACCENT, expanded_rect, 1, border_radius=4)

                # Options
                for i, option in enumerate(dropdown.options):
                    option_rect = pygame.Rect(
                        expanded_rect.x,
                        expanded_rect.y + i * dropdown.option_height,
                        expanded_rect.width,
                        dropdown.option_height
                    )
                    # Highlight hovered option
                    if i == dropdown.hovered_option:
                        pygame.draw.rect(screen, config.COLOR_ACCENT, option_rect)
                    elif i == dropdown.selected_index:
                        pygame.draw.rect(screen, (60, 60, 70), option_rect)

                    opt_surf = font.render(option, True, config.COLOR_TEXT)
                    screen.blit(opt_surf, (option_rect.x + 5, option_rect.centery - opt_surf.get_height() // 2))