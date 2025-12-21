import pygame
import math
import core.config as config

from model.properties import Material
from core.sound_manager import SoundManager

# --- ANIMATION HELPER ---
def lerp(start, end, t):
    return start + (end - start) * t

class AnimVar:
    """Simple spring-like animation value."""
    def __init__(self, value, speed=15.0): 
        self.value = value
        self.target = value
        self.speed = speed
        
    def update(self, dt): 
        # Exponential smoothing approach
        self.value += (self.target - self.value) * min(self.speed * dt, 1.0)
        
    def set(self, target): 
        self.target = target

# --- BASE WIDGETS ---

class Widget:
    def update(self, dt): pass
    def handle_event(self, event): return False
    def draw(self, screen, font): pass

class Button(Widget):
    def __init__(self, x, y, w, h, text, active=False, toggle=True, color_active=config.COLOR_ACCENT, color_inactive=(60, 60, 65)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.active = active
        self.toggle = toggle
        self.clicked = False
        self.disabled = False
        
        # Colors
        self.c_active = color_active
        self.c_inactive = color_inactive
        
        # Animation State
        self.hovered = False
        self.anim_hover = AnimVar(0.0)
        self.anim_click = AnimVar(0.0, speed=20.0)
        self.ripples = [] # List of dicts {x, y, r, alpha}
        
        self.cached_surf = None

    def update(self, dt):
        if self.disabled: 
            self.anim_hover.target = 0.0
        else:
            self.anim_hover.target = 1.0 if self.hovered else 0.0
            
        self.anim_hover.update(dt)
        self.anim_click.update(dt)
        
        # Ripple physics
        for r in self.ripples[:]:
            r['r'] += 200 * dt
            r['a'] -= 2.5 * dt
            if r['a'] <= 0: self.ripples.remove(r)

    def handle_event(self, event):
        if self.disabled: return False
        
        action = False
        sounds = SoundManager.get()
        
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos)
            if self.hovered and not was_hovered:
                sounds.play_sound('hover')

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos) and event.button == 1:
                self.clicked = True
                self.anim_click.value = 1.0 # Flash effect
                self.ripples.append({'x': event.pos[0]-self.rect.x, 'y': event.pos[1]-self.rect.y, 'r': 5, 'a': 1.0})
                
                # SOUND LOGIC ADJUSTMENT:
                # If it's a toggle button that is currently INACTIVE (about to turn ON/Resume),
                # we suppress the click sound here so we only hear the Bloop on release.
                # If it's ACTIVE (about to turn OFF/Pause), we play the click/clink here.
                should_play_click = True
                if self.toggle and not self.active:
                    should_play_click = False
                
                if should_play_click:
                    sounds.play_sound('click')

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.clicked:
                if self.rect.collidepoint(event.pos) and event.button == 1:
                    if self.toggle:
                        self.active = not self.active
                        # If we just turned ON (Resume), play the Bloop (snap)
                        if self.active: 
                            sounds.play_sound('snap')
                    action = True
                self.clicked = False
        
        return action

    def draw(self, screen, font):
        # 1. Color Blending
        base = self.c_active if self.active else self.c_inactive
        if self.disabled: base = (40, 40, 40)
        
        # Highlight is lighter version of base
        highlight = [min(255, c + 35) for c in base]
        
        # Lerp based on hover
        t = self.anim_hover.value
        col = [int(lerp(base[i], highlight[i], t)) for i in range(3)]
        
        # Click flash (white tint)
        if self.anim_click.value > 0.01:
            flash_int = int(50 * self.anim_click.value)
            col = [min(255, c + flash_int) for c in col]

        # 2. Draw Shadow (Pseudo-3D)
        if not self.active:
            shadow_rect = self.rect.copy()
            shadow_rect.y += 3
            pygame.draw.rect(screen, (0,0,0,60), shadow_rect, border_radius=6)

        # 3. Draw Body
        draw_rect = self.rect.copy()
        if self.active: draw_rect.y += 1 # Press down slightly
        
        pygame.draw.rect(screen, col, draw_rect, border_radius=6)
        
        # 4. Draw Border (Active glow)
        if self.active or self.hovered:
            border_col = [min(255, c + 60) for c in col]
            pygame.draw.rect(screen, border_col, draw_rect, 1, border_radius=6)
        else:
            pygame.draw.rect(screen, config.PANEL_BORDER_COLOR, draw_rect, 1, border_radius=6)

        # 5. Draw Ripples (Clipped)
        if self.ripples:
            # Create a subsurface for clipping
            surf = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
            for r in self.ripples:
                pygame.draw.circle(surf, (255, 255, 255, int(r['a']*50)), (int(r['x']), int(r['y'])), int(r['r']))
            screen.blit(surf, draw_rect.topleft)

        # 6. Text
        txt_col = config.COLOR_TEXT
        if self.disabled: txt_col = (100, 100, 100)
        elif not self.active and not self.hovered: txt_col = config.COLOR_TEXT_DIM
        
        ts = font.render(self.text, True, txt_col)
        ts_rect = ts.get_rect(center=draw_rect.center)
        screen.blit(ts, ts_rect)

class InputField(Widget):
    def __init__(self, x, y, w, h, initial_text="", text_color=config.COLOR_TEXT):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = str(initial_text)
        self.active = False
        self.text_color = text_color
        self.last_text = None
        self.cached_surf = None
        
        self.hovered = False

    def move(self, dx, dy):
        self.rect.x += dx; self.rect.y += dy

    def update(self, dt):
        pass 

    def handle_event(self, event):
        changed = False
        
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True; changed = True
                SoundManager.get().play_sound('click')
            elif self.active:
                self.active = False; changed = True
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN: 
                self.active = False
                SoundManager.get().play_sound('snap')
            elif event.key == pygame.K_BACKSPACE: 
                self.text = self.text[:-1]
            else: 
                # Basic filter
                if len(event.unicode) > 0 and (event.unicode.isprintable()):
                    self.text += event.unicode
            changed = True
        return changed

    def get_value(self, default=0.0):
        try: return float(self.text)
        except: return default

    def get_text(self): return self.text

    def set_value(self, val):
        if not self.active:
            self.text = f"{val:.2f}" if isinstance(val, float) else str(val)

    def draw(self, screen, font):
        bg = config.COLOR_INPUT_ACTIVE if self.active else config.COLOR_INPUT_BG
        if self.hovered and not self.active:
            bg = (min(255, bg[0]+10), min(255, bg[1]+10), min(255, bg[2]+10))
            
        pygame.draw.rect(screen, bg, self.rect, border_radius=4)
        
        border = config.COLOR_ACCENT if self.active else config.PANEL_BORDER_COLOR
        pygame.draw.rect(screen, border, self.rect, 1, border_radius=4)
        
        ts = font.render(self.text, True, self.text_color)
        
        # Clip text if too long
        screen.set_clip(self.rect.inflate(-4, -4))
        screen.blit(ts, (self.rect.x+5, self.rect.centery - ts.get_height()//2))
        screen.set_clip(None)

class SmartSlider(Widget):
    def __init__(self, x, y, w, min_val, max_val, initial_val, label, hard_min=None, hard_max=None):
        self.x=x; self.y=y; self.w=w; self.val=initial_val; self.label=label
        self.min_val=min_val; self.max_val=max_val; self.hard_min=hard_min; self.hard_max=hard_max
        self.dragging = False
        
        # Sub-widgets
        self.in_val = InputField(x+w-50, y, 50, 24, str(initial_val))
        self.rect_track = pygame.Rect(x+5, y+35, w-10, 6)
        
        self.hovered = False
        self.anim_hover = AnimVar(0.0)

    def update(self, dt):
        self.anim_hover.target = 1.0 if self.hovered or self.dragging else 0.0
        self.anim_hover.update(dt)

    def handle_event(self, event):
        changed = False
        if self.in_val.handle_event(event):
            self.val = self.in_val.get_value(self.val)
            changed = True
        
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect_track.inflate(0, 14).collidepoint(event.pos)
            if self.dragging:
                rel = (event.pos[0] - self.rect_track.x) / self.rect_track.w
                self.val = self.min_val + max(0, min(1, rel)) * (self.max_val - self.min_val)
                changed = True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.dragging = True
                changed = True
                SoundManager.get().play_sound('click')
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            
        if changed and not self.in_val.active: 
            self.in_val.set_value(self.val)
        return changed

    def draw(self, screen, font):
        # Label
        screen.blit(font.render(self.label, True, config.COLOR_TEXT_DIM), (self.x, self.y+2))
        self.in_val.draw(screen, font)
        
        # Track Background
        pygame.draw.rect(screen, (20, 20, 22), self.rect_track, border_radius=3)
        
        # Filled part
        if self.max_val > self.min_val:
            pct = (self.val - self.min_val) / (self.max_val - self.min_val)
        else:
            pct = 0
        pct = max(0, min(1, pct))
        
        fill_rect = self.rect_track.copy()
        fill_rect.width = max(6, fill_rect.width * pct)
        
        fill_col = config.COLOR_ACCENT
        if self.dragging: 
            fill_col = (min(255, fill_col[0]+40), min(255, fill_col[1]+40), min(255, fill_col[2]+40))
            
        pygame.draw.rect(screen, fill_col, fill_rect, border_radius=3)
        
        # Handle (Scales on hover)
        handle_x = self.rect_track.x + self.rect_track.w * pct
        h_rad = 6 + 2 * self.anim_hover.value
        pygame.draw.circle(screen, (240, 240, 240), (int(handle_x), self.rect_track.centery), int(h_rad))

class MenuBar(Widget):
    def __init__(self, w, h=30):
        self.rect = pygame.Rect(0, 0, w, h)
        self.items = {"File": [], "Tools": [], "Help": []} # Items populated externally
        self.active_menu = None 
        self.dropdown_rect = None
        self.hover_item_idx = -1 
        
        self.item_rects = {}
        curr_x = 15
        for key in self.items:
            pass
            
    def resize(self, w):
        self.rect.width = w

    def handle_event(self, event):
        # Re-calc rects on fly for hit testing
        curr_x = 15
        self.item_rects = {}
        for key in self.items:
            width = 60 # approx
            self.item_rects[key] = pygame.Rect(curr_x, 0, width, self.rect.height)
            curr_x += width + 5

        if event.type == pygame.MOUSEMOTION:
            if self.active_menu and self.dropdown_rect:
                if self.dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - self.dropdown_rect.y - 5
                    if rel_y >= 0:
                        idx = rel_y // 30
                        if 0 <= idx < len(self.items[self.active_menu]):
                            if self.items[self.active_menu][idx] != "---":
                                self.hover_item_idx = idx
                            else: self.hover_item_idx = -1
                        else: self.hover_item_idx = -1
                else: self.hover_item_idx = -1
            else: self.hover_item_idx = -1
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Top Bar Click
            for key, r in self.item_rects.items():
                if r.collidepoint(event.pos):
                    if self.active_menu == key:
                        self.active_menu = None
                    else:
                        self.active_menu = key
                        SoundManager.get().play_sound('click')
                    return True
                    
            # Dropdown Click
            if self.active_menu and self.dropdown_rect and self.hover_item_idx != -1:
                item = self.items[self.active_menu][self.hover_item_idx]
                SoundManager.get().play_sound('snap')
                self.active_menu = None
                return item # Return action string
                
        return False

    def draw(self, screen, font):
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect)
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (0, self.rect.bottom-1), (self.rect.width, self.rect.bottom-1))
        
        curr_x = 15
        for key in self.items:
            ts = font.render(key, True, config.COLOR_TEXT)
            width = ts.get_width() + 20
            r = pygame.Rect(curr_x, 0, width, self.rect.height)
            self.item_rects[key] = r
            
            if self.active_menu == key:
                pygame.draw.rect(screen, (50, 50, 55), r)
            
            screen.blit(ts, (curr_x + 10, r.centery - ts.get_height()//2))
            curr_x += width + 5
            
        if self.active_menu:
            opts = self.items[self.active_menu]
            r = self.item_rects[self.active_menu]
            
            dd_w = 180
            dd_h = len(opts) * 30 + 10
            self.dropdown_rect = pygame.Rect(r.x, r.height + 2, dd_w, dd_h)
            
            # Shadow
            s = self.dropdown_rect.copy(); s.x+=4; s.y+=4
            s_surf = pygame.Surface((s.width, s.height), pygame.SRCALPHA)
            pygame.draw.rect(s_surf, (0, 0, 0, 80), s_surf.get_rect(), border_radius=4)
            screen.blit(s_surf, s)
            
            # Body
            pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.dropdown_rect, border_radius=4)
            pygame.draw.rect(screen, config.PANEL_BORDER_COLOR, self.dropdown_rect, 1, border_radius=4)
            
            for i, opt in enumerate(opts):
                if i == self.hover_item_idx and opt != "---":
                    h_rect = pygame.Rect(self.dropdown_rect.x + 2, self.dropdown_rect.y + 5 + i*30, self.dropdown_rect.width - 4, 30)
                    pygame.draw.rect(screen, (60, 60, 65), h_rect, border_radius=2)

                if opt == "---":
                    y = self.dropdown_rect.y + 5 + i*30 + 15
                    pygame.draw.line(screen, (60, 60, 60), (self.dropdown_rect.x + 10, y), (self.dropdown_rect.right - 10, y))
                else:
                    col = config.COLOR_TEXT
                    if i == self.hover_item_idx: col = (255, 255, 255)
                    otxt = font.render(opt, True, col)
                    screen.blit(otxt, (self.dropdown_rect.x + 15, self.dropdown_rect.y + 5 + i*30 + 5))

class ContextMenu(Widget):
    def __init__(self, x, y, options):
        self.x = x
        self.y = y
        self.options = options
        self.width = 160
        self.height = len(options) * 30 + 10
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.selected_idx = -1
        self.action = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - (self.y + 5)
                if rel_y >= 0: self.selected_idx = rel_y // 30
                else: self.selected_idx = -1
            else: self.selected_idx = -1
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - (self.y + 5)
                idx = rel_y // 30
                if 0 <= idx < len(self.options):
                    self.action = self.options[idx]
                    SoundManager.get().play_sound('snap')
                    return True
            else:
                self.action = "CLOSE"
                return True
        return False

    def draw(self, screen, font):
        shadow = self.rect.copy(); shadow.x += 4; shadow.y += 4
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=4)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=4)
        pygame.draw.rect(screen, config.PANEL_BORDER_COLOR, self.rect, 1, border_radius=4)
        
        for i, opt in enumerate(self.options):
            bg_col = config.COLOR_ACCENT if i == self.selected_idx else config.PANEL_BG_COLOR
            item_rect = pygame.Rect(self.x + 2, self.y + 5 + i*30, self.width - 4, 28)
            
            if i == self.selected_idx:
                pygame.draw.rect(screen, bg_col, item_rect, border_radius=3)
            
            col = config.COLOR_TEXT if i != self.selected_idx else (255, 255, 255)
            txt = font.render(opt, True, col)
            screen.blit(txt, (self.x + 10, self.y + 5 + i*30 + 5))

class MaterialDialog:
    def __init__(self, x, y, sketch, current_material_id):
        self.rect = pygame.Rect(x, y, 300, 280)
        self.sketch = sketch
        self.done = False
        self.apply = False
        
        mat = sketch.get_material(current_material_id)
        
        self.in_id = InputField(x + 120, y + 45, 150, 25, mat.name)
        self.in_sigma = InputField(x + 120, y + 80, 150, 25, str(mat.sigma))
        self.in_epsilon = InputField(x + 120, y + 115, 150, 25, str(mat.epsilon))
        self.in_spacing = InputField(x + 120, y + 150, 150, 25, str(mat.spacing))
        
        self.btn_phys = Button(x + 120, y + 185, 80, 25, "Solid" if mat.physical else "Ghost", 
                               active=mat.physical, toggle=True, color_active=config.COLOR_SUCCESS)

        self.btn_apply = Button(x + 20, y + 230, 80, 30, "Apply", toggle=False)
        self.btn_ok = Button(x + 180, y + 230, 80, 30, "OK", toggle=False)

    def handle_event(self, event):
        if self.in_id.handle_event(event):
            name = self.in_id.get_text()
            if name in self.sketch.materials:
                m = self.sketch.materials[name]
                self.in_sigma.set_value(m.sigma)
                self.in_epsilon.set_value(m.epsilon)
                self.in_spacing.set_value(m.spacing)
                self.btn_phys.active = m.physical
                self.btn_phys.text = "Solid" if m.physical else "Ghost"
                self.btn_phys.cached_surf = None
            return True
            
        if self.in_sigma.handle_event(event): return True
        if self.in_epsilon.handle_event(event): return True
        if self.in_spacing.handle_event(event): return True
        
        if self.btn_phys.handle_event(event):
            self.btn_phys.text = "Solid" if self.btn_phys.active else "Ghost"
            self.btn_phys.cached_surf = None
            return True

        if self.btn_apply.handle_event(event):
            self.apply = True; return True
        if self.btn_ok.handle_event(event):
            self.apply = True; self.done = True; return True
        return False
        
    def update(self, dt):
        self.btn_phys.update(dt)
        self.btn_apply.update(dt)
        self.btn_ok.update(dt)

    def get_result(self):
        name = self.in_id.get_text()
        if not name: name = "Default"
        m = Material(
            name,
            sigma=self.in_sigma.get_value(1.0),
            epsilon=self.in_epsilon.get_value(1.0),
            spacing=self.in_spacing.get_value(0.7),
            physical=self.btn_phys.active
        )
        return m

    def draw(self, screen, font):
        shadow = self.rect.copy(); shadow.x += 5; shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)
        
        title = font.render("Material Editor", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 10))
        
        labels = ["Material ID:", "Sigma:", "Epsilon:", "Spacing:", "Physics:"]
        ys = [50, 85, 120, 155, 190]
        for l, y in zip(labels, ys):
            screen.blit(font.render(l, True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + y))
        
        self.in_id.draw(screen, font)
        self.in_sigma.draw(screen, font)
        self.in_epsilon.draw(screen, font)
        self.in_spacing.draw(screen, font)
        self.btn_phys.draw(screen, font)
        self.btn_apply.draw(screen, font)
        self.btn_ok.draw(screen, font)

class RotationDialog:
    def __init__(self, x, y, anim_data):
        self.rect = pygame.Rect(x, y, 250, 210)
        self.done = False
        self.apply = False
        
        speed = 0.0
        pivot = "center"
        if anim_data:
            speed = anim_data.get('speed', 0.0)
            pivot = anim_data.get('pivot', 'center')

        self.in_speed = InputField(x + 100, y + 60, 100, 25, str(speed))
        
        self.pivots = ["center", "start", "end"]
        try: self.pivot_idx = self.pivots.index(pivot)
        except: self.pivot_idx = 0
            
        self.btn_pivot = Button(x + 100, y + 100, 100, 25, f"{self.pivots[self.pivot_idx].title()}", toggle=False, color_inactive=(60, 60, 70))
        self.btn_apply = Button(x + 20, y + 160, 80, 30, "Apply", toggle=False)
        self.btn_ok = Button(x + 150, y + 160, 80, 30, "OK", toggle=False)

    def handle_event(self, event):
        if self.in_speed.handle_event(event): return True
        if self.btn_pivot.handle_event(event):
            self.pivot_idx = (self.pivot_idx + 1) % len(self.pivots)
            self.btn_pivot.text = f"{self.pivots[self.pivot_idx].title()}"
            self.btn_pivot.cached_surf = None
            return True
        if self.btn_apply.handle_event(event):
            self.apply = True; return True
        if self.btn_ok.handle_event(event):
            self.apply = True; self.done = True; return True
        return False
    
    def update(self, dt):
        self.btn_pivot.update(dt)
        self.btn_apply.update(dt)
        self.btn_ok.update(dt)

    def get_values(self):
        return { 'speed': self.in_speed.get_value(0.0), 'pivot': self.pivots[self.pivot_idx] }

    def draw(self, screen, font):
        shadow = self.rect.copy(); shadow.x += 5; shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)
        
        title = font.render("Rotation Animation", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 10))
        
        screen.blit(font.render("Speed:", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 65))
        self.in_speed.draw(screen, font)
        
        screen.blit(font.render("Axis:", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 105))
        self.btn_pivot.draw(screen, font)
        
        self.btn_apply.draw(screen, font)
        self.btn_ok.draw(screen, font)

class AnimationDialog:
    def __init__(self, x, y, driver_data):
        self.rect = pygame.Rect(x, y, 300, 280)
        self.driver = driver_data if driver_data else {'type': 'sin', 'amp': 15.0, 'freq': 0.5, 'phase': 0.0, 'rate': 10.0}
        self.done = False
        self.apply = False
        self.current_tab = self.driver.get('type', 'sin')
        
        self.btn_stop = Button(x + 20, y + 230, 100, 30, "Stop/Clear", toggle=False, color_inactive=config.COLOR_DANGER)
        self.btn_ok = Button(x + 180, y + 230, 100, 30, "Update", toggle=False, color_inactive=config.COLOR_SUCCESS)
        
        self.btn_tab_sin = Button(x + 20, y + 50, 120, 25, "Sinusoidal", toggle=False, active=(self.current_tab == 'sin'))
        self.btn_tab_lin = Button(x + 160, y + 50, 120, 25, "Linear", toggle=False, active=(self.current_tab == 'lin'))

        self.in_amp = InputField(x + 150, y + 90, 100, 25, str(self.driver.get('amp', 15.0)))
        self.in_freq = InputField(x + 150, y + 130, 100, 25, str(self.driver.get('freq', 0.5)))
        self.in_phase = InputField(x + 150, y + 170, 100, 25, str(self.driver.get('phase', 0.0)))
        
        self.in_rate = InputField(x + 150, y + 90, 100, 25, str(self.driver.get('rate', 10.0)))

    def handle_event(self, event):
        if self.btn_tab_sin.handle_event(event):
            self.current_tab = 'sin'
            self.btn_tab_sin.active = True; self.btn_tab_lin.active = False
            return True
        if self.btn_tab_lin.handle_event(event):
            self.current_tab = 'lin'
            self.btn_tab_sin.active = False; self.btn_tab_lin.active = True
            return True

        if self.current_tab == 'sin':
            if self.in_amp.handle_event(event): return True
            if self.in_freq.handle_event(event): return True
            if self.in_phase.handle_event(event): return True
        else:
            if self.in_rate.handle_event(event): return True
        
        if self.btn_stop.handle_event(event):
            self.driver = None; self.apply = True; self.done = True
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
                self.driver = {'type': 'lin', 'rate': self.in_rate.get_value(0.0)}
            self.apply = True; self.done = True
            return True
        return False
    
    def update(self, dt):
        self.btn_tab_sin.update(dt)
        self.btn_tab_lin.update(dt)
        self.btn_stop.update(dt)
        self.btn_ok.update(dt)

    def get_values(self): return self.driver

    def draw(self, screen, font):
        shadow = self.rect.copy(); shadow.x += 5; shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)
        
        screen.blit(font.render("Drive Constraint", True, (255, 255, 255)), (self.rect.x + 15, self.rect.y + 10))
        
        self.btn_tab_sin.draw(screen, font)
        self.btn_tab_lin.draw(screen, font)
        
        if self.current_tab == 'sin':
            screen.blit(font.render("Amplitude:", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 95))
            screen.blit(font.render("Freq (Hz):", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 135))
            screen.blit(font.render("Phase (deg):", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 175))
            self.in_amp.draw(screen, font)
            self.in_freq.draw(screen, font)
            self.in_phase.draw(screen, font)
        else:
            screen.blit(font.render("Rate (deg/s):", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 95))
            self.in_rate.draw(screen, font)
            
        self.btn_stop.draw(screen, font)
        self.btn_ok.draw(screen, font)