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


# =============================================================================
# OVERLAY PROVIDER PROTOCOL
# =============================================================================

class OverlayProvider:
    """
    Protocol for widgets that need to render floating overlays above other UI elements.

    Examples: Dropdowns (expanded list), Tooltips, Popovers

    Widgets implementing this protocol should:
    1. Call ui_manager.register_overlay(self) when overlay should be visible
    2. Call ui_manager.unregister_overlay(self) when overlay should hide
    3. Implement get_overlay_rect() and draw_overlay() methods
    """

    # Reference to UIManager for registration (set by UIManager during init)
    _ui_manager = None

    @classmethod
    def set_ui_manager(cls, ui_manager):
        """Set the UIManager instance for all OverlayProviders."""
        cls._ui_manager = ui_manager

    def get_overlay_rect(self):
        """Return the screen-space rect of the overlay for hit-testing."""
        return pygame.Rect(0, 0, 0, 0)

    def draw_overlay(self, screen, font):
        """Draw the floating overlay element."""
        pass

    def _register_overlay(self):
        """Register this widget's overlay with the UIManager."""
        if OverlayProvider._ui_manager:
            OverlayProvider._ui_manager.register_overlay(self)

    def _unregister_overlay(self):
        """Unregister this widget's overlay from the UIManager."""
        if OverlayProvider._ui_manager:
            OverlayProvider._ui_manager.unregister_overlay(self)


# =============================================================================
# PHASE I: HIERARCHICAL STRUCTURAL FOUNDATIONS
# =============================================================================

class UIElement:
    """
    Abstract Base Class for all visual components.
    Defines the contract for the 'Layer Cake' hierarchy.
    """
    def __init__(self, x=0, y=0, w=0, h=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.visible = True
        self.active = False
        self.disabled = False
        self.hovered = False
        self.parent = None
        self.tooltip = None

        # Relative positioning (for Containers)
        self.rel_x = 0
        self.rel_y = 0
        self.rel_layout_managed = False  # If True, container controls position

    def update(self, dt):
        """Update logic/animation."""
        pass

    def handle_event(self, event):
        """
        Process input event.
        Returns True if the event was 'absorbed' (consumed), False otherwise.
        """
        return False

    def draw(self, screen, font):
        """Draw to screen."""
        pass

    def set_position(self, x, y):
        """Update absolute position."""
        self.rect.x = x
        self.rect.y = y

    def get_absolute_rect(self):
        """
        Calculate absolute screen rect based on parent hierarchy.
        (Future-proofing for deep nesting in Phase II/III)
        """
        if self.parent:
            p_rect = self.parent.get_absolute_rect()
            return pygame.Rect(p_rect.x + self.rel_x, p_rect.y + self.rel_y, self.rect.w, self.rect.h)
        return self.rect

    def on_focus_lost(self):
        """
        Called when this element loses focus (user clicked elsewhere).

        Override in subclasses to handle focus-loss behavior like:
        - Clearing preview changes
        - Reverting unsaved modifications
        - Closing expanded dropdowns

        The InputHandler manages focus and calls this method when appropriate.
        """
        pass

    def wants_focus(self):
        """
        Return True if this element should acquire focus when clicked.

        Override in subclasses that need focus tracking (e.g., MaterialPropertyWidget).
        Default is False - most elements don't need focus tracking.
        """
        return False

    def reset_interaction_state(self):
        """
        Reset any in-progress interaction state (e.g., button clicks, drag operations).

        Called when a modal dialog opens to prevent stale interaction state from
        causing spurious triggers when the modal closes. For example, if a button
        is clicked and that click opens a modal, the button's 'clicked' state
        should be cleared so the eventual MOUSEBUTTONUP doesn't re-trigger it.

        Override in subclasses that maintain interaction state.
        """
        pass


class UIContainer(UIElement):
    """
    Composite Node in the UI Tree.
    Manages layout and delegates events to children.
    """
    def __init__(self, x, y, w, h, layout_type='free', padding=5, spacing=5, bg_color=None, border_color=None):
        super().__init__(x, y, w, h)
        self.children = []
        self.layout_type = layout_type  # 'vertical', 'horizontal', 'free'
        self.padding = padding
        self.spacing = spacing
        self.bg_color = bg_color
        self.border_color = border_color

    def add_child(self, child):
        """Add a child element and recalculate layout."""
        child.parent = self
        self.children.append(child)
        if self.layout_type != 'free':
            child.rel_layout_managed = True
            self.recalculate_layout()
    
    def clear_children(self):
        self.children = []

    def set_position(self, x, y):
        """
        Override to propagate position changes to children.
        This ensures that when a container is moved (e.g. by scrolling),
        all nested elements move with it.
        """
        dx = x - self.rect.x
        dy = y - self.rect.y
        
        # Update self
        super().set_position(x, y)
        
        # Propagate delta to all children
        for child in self.children:
            child.set_position(child.rect.x + dx, child.rect.y + dy)

    def recalculate_layout(self):
        """
        Automatic Stacking: Increments cursor position based on padding/spacing.
        Updates children's positions.
        """
        if self.layout_type == 'free':
            return

        cursor_x = self.padding
        cursor_y = self.padding

        for child in self.children:
            if not child.visible:
                continue

            if self.layout_type == 'vertical':
                # For Phase I mixed mode: We update the child's absolute rect directly
                child.set_position(self.rect.x + self.padding, self.rect.y + cursor_y)
                cursor_y += child.rect.height + self.spacing
            
            elif self.layout_type == 'horizontal':
                child.set_position(self.rect.x + cursor_x, self.rect.y + self.padding)
                cursor_x += child.rect.width + self.spacing

    def update(self, dt):
        """Propagate updates to children."""
        for child in self.children:
            if child.visible:
                child.update(dt)

    def handle_event(self, event):
        """
        The Chain of Responsibility:
        1. Check if visible.
        2. Delegate to children (Top-down or Z-order).
        3. Check self.
        """
        if not self.visible:
            return False

        # Delegate to children in reverse order (top-most first)
        for child in reversed(self.children):
            if child.handle_event(event):
                return True  # Child absorbed event

        # Handle container-specific events (e.g. background click)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True  # Absorb click on panel background

        return False

    def draw(self, screen, font):
        """Draw background then children."""
        if not self.visible:
            return

        # Draw Background
        if self.bg_color:
            pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=4)
        if self.border_color:
            pygame.draw.rect(screen, self.border_color, self.rect, 1, border_radius=4)

        # Draw Children
        for child in self.children:
            if child.visible:
                child.draw(screen, font)

    def reset_interaction_state(self):
        """Propagate interaction state reset to all children."""
        for child in self.children:
            child.reset_interaction_state()


class ScrollableContainer(UIContainer):
    """
    A UIContainer that allows vertical scrolling if content exceeds height.
    """
    def __init__(self, x, y, w, h, **kwargs):
        super().__init__(x, y, w, h, **kwargs)
        self.scroll_y = 0
        self.content_height = 0
        self.surface = None # For clipping/scrolling optimization if needed

    def handle_event(self, event):
        if not self.visible: return False

        # 1. Handle Scrolling
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                # Scroll speed (pixels per click)
                scroll_speed = 30
                
                # Only scroll if content is larger than container
                max_scroll = max(0, self.content_height - self.rect.height + self.padding * 2)
                
                if max_scroll > 0:
                    self.scroll_y += event.y * scroll_speed
                    # Clamp scroll
                    self.scroll_y = min(0, max(-max_scroll, self.scroll_y))
                    self.recalculate_layout()
                    return True # Consume the wheel event

        # 2. Clip interactions to the container rect
        # (Prevents clicking buttons that are scrolled out of view)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
             if not self.rect.collidepoint(event.pos):
                 return False

        return super().handle_event(event)

    def recalculate_layout(self):
        """
        Standard layout calculation but applies self.scroll_y offset.
        """
        if self.layout_type == 'free': return

        cursor_x = self.padding
        cursor_y = self.padding + self.scroll_y # Apply scroll offset

        # Reset content height calculation
        total_h = self.padding 

        for child in self.children:
            if not child.visible: continue

            # Position child
            if self.layout_type == 'vertical':
                child.set_position(self.rect.x + self.padding, self.rect.y + cursor_y)
                cursor_y += child.rect.height + self.spacing
                total_h += child.rect.height + self.spacing
            
            elif self.layout_type == 'horizontal':
                child.set_position(self.rect.x + cursor_x, self.rect.y + self.padding + self.scroll_y)
                cursor_x += child.rect.width + self.spacing
                # Height tracking for horizontal is just the tallest item
                total_h = max(total_h, child.rect.height + self.padding*2)
        
        self.content_height = total_h

    def draw(self, screen, font):
        if not self.visible: return

        # 1. Draw Background
        if self.bg_color:
            pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=4)
        
        # 2. Clip Content
        # We set the clipping region to the container's rect so children
        # don't draw outside the bounds when scrolling.
        prev_clip = screen.get_clip()
        clip_rect = self.rect.clip(prev_clip) if prev_clip else self.rect
        screen.set_clip(clip_rect)

        # 3. Draw Children
        for child in self.children:
            if child.visible:
                # Optimization: Only draw if roughly in view
                if child.rect.bottom > self.rect.top and child.rect.top < self.rect.bottom:
                    child.draw(screen, font)

        # 4. Restore Clip
        screen.set_clip(prev_clip)
        
        # 5. Draw Border (on top of clipped content)
        if self.border_color:
            pygame.draw.rect(screen, self.border_color, self.rect, 1, border_radius=4)
            
        # 6. Simple Scrollbar Indicator
        if self.content_height > self.rect.height:
            bar_h = max(20, (self.rect.height / self.content_height) * self.rect.height)
            scroll_pct = abs(self.scroll_y) / (self.content_height - self.rect.height)
            bar_y = self.rect.y + scroll_pct * (self.rect.height - bar_h)
            
            scrollbar_rect = pygame.Rect(self.rect.right - 6, bar_y, 4, bar_h)
            pygame.draw.rect(screen, (100, 100, 100), scrollbar_rect, border_radius=2)


# =============================================================================
# WIDGET IMPLEMENTATIONS
# =============================================================================

class Button(UIElement):
    def __init__(self, x, y, w, h, text="", active=False, toggle=True, 
                 color_active=config.COLOR_ACCENT, color_inactive=(60, 60, 65),
                 icon=None, tooltip=None, callback=None):
        super().__init__(x, y, w, h)
        self.text = text
        self.icon = icon  # Icon drawing function from icons.py
        self.tooltip = tooltip
        self.active = active
        self.toggle = toggle
        self.clicked = False
        self.callback = callback
        
        # Colors
        self.c_active = color_active
        self.c_inactive = color_inactive
        
        # Animation State
        self.anim_hover = AnimVar(0.0)
        self.anim_click = AnimVar(0.0, speed=20.0)
        self.ripples = [] # List of dicts {x, y, r, alpha}
        
        # Tooltip state
        self.hover_time = 0.0
        self.show_tooltip = False

    def update(self, dt):
        if self.disabled: 
            self.anim_hover.target = 0.0
        else:
            self.anim_hover.target = 1.0 if self.hovered else 0.0
            
        self.anim_hover.update(dt)
        self.anim_click.update(dt)
        
        # Tooltip timing
        if self.hovered and self.tooltip:
            self.hover_time += dt
            if self.hover_time > 0.5:  # Show tooltip after 0.5 seconds
                self.show_tooltip = True
        else:
            self.hover_time = 0.0
            self.show_tooltip = False
        
        # Ripple physics
        for r in self.ripples[:]:
            r['r'] += 200 * dt
            r['a'] -= 2.5 * dt
            if r['a'] <= 0: self.ripples.remove(r)

    def handle_event(self, event):
        if self.disabled or not self.visible: return False
        
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
                
                should_play_click = True
                if self.toggle and not self.active:
                    should_play_click = False
                
                if should_play_click:
                    sounds.play_sound('click')
                return True # Absorb the click

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.clicked:
                if self.rect.collidepoint(event.pos) and event.button == 1:
                    if self.toggle:
                        self.active = not self.active
                        if self.active: 
                            sounds.play_sound('snap')
                    if self.callback:
                        self.callback()
                    action = True
                self.clicked = False
                return True 
        
        return action

    def reset_interaction_state(self):
        """Clear clicked state to prevent stale triggers after modal dialogs."""
        self.clicked = False

    def draw(self, screen, font):
        if not self.visible: return

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

        # 6. Content (Icon or Text)
        content_col = config.COLOR_TEXT
        if self.disabled: content_col = (100, 100, 100)
        elif not self.active and not self.hovered: content_col = config.COLOR_TEXT_DIM
        
        if self.icon:
            # Draw icon centered in button with padding
            icon_rect = draw_rect.inflate(-6, -6)
            self.icon(screen, icon_rect, content_col)
        elif self.text:
            # Draw text centered
            ts = font.render(self.text, True, content_col)
            ts_rect = ts.get_rect(center=draw_rect.center)
            screen.blit(ts, ts_rect)
        
        # 7. Tooltip (drawn last, on top)
        if self.show_tooltip and self.tooltip:
            self._draw_tooltip(screen, font)
    
    def _draw_tooltip(self, screen, font):
        """Draw tooltip near the button."""
        tip_surf = font.render(self.tooltip, True, config.COLOR_TEXT)
        tip_w = tip_surf.get_width() + 12
        tip_h = tip_surf.get_height() + 8
        
        # Position tooltip above the button
        tip_x = self.rect.centerx - tip_w // 2
        tip_y = self.rect.y - tip_h - 5
        
        # Keep tooltip on screen
        screen_w = screen.get_width()
        if tip_x < 5: tip_x = 5
        if tip_x + tip_w > screen_w - 5: tip_x = screen_w - tip_w - 5
        if tip_y < 5: tip_y = self.rect.bottom + 5  # Show below if no room above
        
        tip_rect = pygame.Rect(tip_x, tip_y, tip_w, tip_h)
        
        # Shadow
        shadow_rect = tip_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        shadow_surf = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 100), shadow_surf.get_rect(), border_radius=4)
        screen.blit(shadow_surf, shadow_rect.topleft)
        
        # Background
        pygame.draw.rect(screen, (50, 52, 58), tip_rect, border_radius=4)
        pygame.draw.rect(screen, config.COLOR_ACCENT, tip_rect, 1, border_radius=4)
        
        # Text
        screen.blit(tip_surf, (tip_x + 6, tip_y + 4))


class InputField(UIElement):
    def __init__(self, x, y, w, h, initial_text="", text_color=config.COLOR_TEXT):
        super().__init__(x, y, w, h)
        self.text = str(initial_text)
        self.text_color = text_color
        self.last_text = None
        self.hovered = False
        # Cursor support
        self.cursor_pos = len(self.text)  # Position in text (0 = before first char)
        self.cursor_blink_timer = 0.0
        self.cursor_visible = True
        self.CURSOR_BLINK_RATE = 0.5  # seconds per blink cycle

    def move(self, dx, dy):
        """Manual move helper (Legacy support)."""
        self.rect.x += dx; self.rect.y += dy

    def update(self, dt):
        """Update cursor blink state."""
        if self.active:
            self.cursor_blink_timer += dt
            if self.cursor_blink_timer >= self.CURSOR_BLINK_RATE:
                self.cursor_blink_timer = 0.0
                self.cursor_visible = not self.cursor_visible
        else:
            self.cursor_visible = True
            self.cursor_blink_timer = 0.0

    def handle_event(self, event):
        if not self.visible: return False
        changed = False
        consumed = False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                was_active = self.active
                self.active = True
                # Reset cursor blink on click
                self.cursor_blink_timer = 0.0
                self.cursor_visible = True
                # Move cursor to end when clicking to activate
                if not was_active:
                    self.cursor_pos = len(self.text)
                changed = True
                SoundManager.get().play_sound('click')
                consumed = True
            elif self.active:
                self.active = False; changed = True
                # Don't consume here to allow click-off to register elsewhere

        elif event.type == pygame.KEYDOWN and self.active:
            # Reset cursor visibility on any keypress
            self.cursor_blink_timer = 0.0
            self.cursor_visible = True

            if event.key == pygame.K_RETURN:
                self.active = False
                SoundManager.get().play_sound('snap')
            elif event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            elif event.key == pygame.K_LEFT:
                if self.cursor_pos > 0:
                    self.cursor_pos -= 1
            elif event.key == pygame.K_RIGHT:
                if self.cursor_pos < len(self.text):
                    self.cursor_pos += 1
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            else:
                # Basic filter
                if len(event.unicode) > 0 and (event.unicode.isprintable()):
                    self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                    self.cursor_pos += 1
            changed = True
            consumed = True # Consume keyboard events if active

        if changed:
            return True
        return consumed

    def get_value(self, default=0.0):
        try: return float(self.text)
        except: return default

    def get_text(self): return self.text

    def set_value(self, val):
        if not self.active:
            self.text = f"{val:.2f}" if isinstance(val, float) else str(val)
            self.cursor_pos = len(self.text)

    def draw(self, screen, font):
        if not self.visible: return

        bg = config.COLOR_INPUT_ACTIVE if self.active else config.COLOR_INPUT_BG
        if self.hovered and not self.active:
            bg = (min(255, bg[0]+10), min(255, bg[1]+10), min(255, bg[2]+10))

        pygame.draw.rect(screen, bg, self.rect, border_radius=4)

        border = config.COLOR_ACCENT if self.active else config.PANEL_BORDER_COLOR
        pygame.draw.rect(screen, border, self.rect, 1, border_radius=4)

        ts = font.render(self.text, True, self.text_color)
        text_x = self.rect.x + 5
        text_y = self.rect.centery - ts.get_height() // 2

        # Clip text if too long
        screen.set_clip(self.rect.inflate(-4, -4))
        screen.blit(ts, (text_x, text_y))

        # Draw blinking cursor when active
        if self.active and self.cursor_visible:
            # Calculate cursor x position based on text before cursor
            text_before_cursor = self.text[:self.cursor_pos]
            cursor_x_offset = font.size(text_before_cursor)[0] if text_before_cursor else 0
            cursor_x = text_x + cursor_x_offset
            cursor_y1 = self.rect.centery - ts.get_height() // 2 + 2
            cursor_y2 = self.rect.centery + ts.get_height() // 2 - 2
            pygame.draw.line(screen, self.text_color, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)

        screen.set_clip(None)


class Dropdown(UIElement, OverlayProvider):
    """A dropdown selector widget with a list of options."""

    def __init__(self, x, y, w, h, options, selected_index=0, text_color=config.COLOR_TEXT):
        super().__init__(x, y, w, h)
        self.options = options  # List of string options
        self.selected_index = selected_index
        self.text_color = text_color
        self.hovered = False
        self._expanded = False  # Is the dropdown list showing?
        self.hovered_option = -1  # Which option is being hovered
        self.option_height = h  # Height of each option in the list
        self.on_change = None  # Callback when selection changes

    @property
    def expanded(self):
        """Get expanded state."""
        return self._expanded

    @expanded.setter
    def expanded(self, value):
        """Set expanded state and register/unregister overlay."""
        if self._expanded != value:
            self._expanded = value
            if value:
                self._register_overlay()
            else:
                self._unregister_overlay()

    def get_selected(self):
        """Get the currently selected option string."""
        if 0 <= self.selected_index < len(self.options):
            return self.options[self.selected_index]
        return ""

    def get_selected_index(self):
        return self.selected_index

    def set_options(self, options, selected_index=0):
        """Update the list of options."""
        self.options = options
        self.selected_index = min(selected_index, len(options) - 1) if options else 0

    def get_expanded_rect(self):
        """Get the rectangle covering the expanded dropdown list."""
        if not self.options:
            return pygame.Rect(0, 0, 0, 0)
        list_height = len(self.options) * self.option_height
        return pygame.Rect(self.rect.x, self.rect.bottom, self.rect.width, list_height)

    def get_overlay_rect(self):
        """OverlayProvider: Return the screen-space rect of the overlay."""
        return self.get_expanded_rect()

    def draw_overlay(self, screen, font):
        """OverlayProvider: Draw the expanded dropdown list as a floating overlay."""
        if not self.expanded or not self.options:
            return

        expanded_rect = self.get_expanded_rect()

        # Shadow
        shadow_rect = expanded_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(screen, (0, 0, 0, 80), shadow_rect, border_radius=4)

        # Background
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, expanded_rect, border_radius=4)
        pygame.draw.rect(screen, config.COLOR_ACCENT, expanded_rect, 1, border_radius=4)

        # Options
        for i, option in enumerate(self.options):
            option_rect = pygame.Rect(
                expanded_rect.x,
                expanded_rect.y + i * self.option_height,
                expanded_rect.width,
                self.option_height
            )
            # Highlight hovered option
            if i == self.hovered_option:
                pygame.draw.rect(screen, config.COLOR_ACCENT, option_rect)
            elif i == self.selected_index:
                pygame.draw.rect(screen, (60, 60, 70), option_rect)

            opt_surf = font.render(option, True, self.text_color)
            screen.blit(opt_surf, (option_rect.x + 5, option_rect.centery - opt_surf.get_height() // 2))

    def handle_event(self, event):
        if not self.visible:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            if self.expanded:
                expanded_rect = self.get_expanded_rect()
                if expanded_rect.collidepoint(event.pos):
                    # Determine which option is hovered
                    rel_y = event.pos[1] - expanded_rect.y
                    self.hovered_option = int(rel_y // self.option_height)
                else:
                    self.hovered_option = -1

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                # Toggle expanded state
                self.expanded = not self.expanded
                self.hovered_option = -1
                SoundManager.get().play_sound('click')
                return True
            elif self.expanded:
                expanded_rect = self.get_expanded_rect()
                if expanded_rect.collidepoint(event.pos):
                    # Select the clicked option
                    rel_y = event.pos[1] - expanded_rect.y
                    clicked_index = int(rel_y // self.option_height)
                    if 0 <= clicked_index < len(self.options):
                        old_index = self.selected_index
                        self.selected_index = clicked_index
                        self.expanded = False
                        SoundManager.get().play_sound('snap')
                        if self.on_change and old_index != clicked_index:
                            self.on_change(clicked_index, self.options[clicked_index])
                        return True
                else:
                    # Click outside - close dropdown
                    self.expanded = False
                    return True

        return False

    def draw(self, screen, font):
        if not self.visible:
            return

        # Draw main button
        bg = config.COLOR_INPUT_ACTIVE if self.expanded else config.COLOR_INPUT_BG
        if self.hovered and not self.expanded:
            bg = (min(255, bg[0] + 10), min(255, bg[1] + 10), min(255, bg[2] + 10))

        pygame.draw.rect(screen, bg, self.rect, border_radius=4)
        border = config.COLOR_ACCENT if self.expanded else config.PANEL_BORDER_COLOR
        pygame.draw.rect(screen, border, self.rect, 1, border_radius=4)

        # Draw selected text
        selected_text = self.get_selected()
        ts = font.render(selected_text, True, self.text_color)
        text_x = self.rect.x + 5
        text_y = self.rect.centery - ts.get_height() // 2
        # Clip text area: leave room for arrow on right (20px), minimal padding on left (2px)
        clip_rect = pygame.Rect(self.rect.x + 2, self.rect.y + 2, self.rect.width - 22, self.rect.height - 4)
        screen.set_clip(clip_rect)
        screen.blit(ts, (text_x, text_y))
        screen.set_clip(None)

        # Draw dropdown arrow
        arrow_x = self.rect.right - 15
        arrow_y = self.rect.centery
        arrow_size = 4
        if self.expanded:
            # Up arrow
            points = [(arrow_x, arrow_y + 2), (arrow_x - arrow_size, arrow_y + arrow_size + 2),
                      (arrow_x + arrow_size, arrow_y + arrow_size + 2)]
        else:
            # Down arrow
            points = [(arrow_x, arrow_y + arrow_size), (arrow_x - arrow_size, arrow_y),
                      (arrow_x + arrow_size, arrow_y)]
        pygame.draw.polygon(screen, self.text_color, points)

        # Note: Expanded list is drawn by UIManager via OverlayProvider.draw_overlay()


class SmartSlider(UIElement):
    def __init__(self, x, y, w, min_val, max_val, initial_val, label, hard_min=None, hard_max=None):
        h = config.scale(60) # Scaled height
        super().__init__(x, y, w, h)
        self.val=initial_val; self.label=label
        self.min_val=min_val; self.max_val=max_val; self.hard_min=hard_min; self.hard_max=hard_max
        self.dragging = False
        
        # Sub-widgets (Scaled)
        input_w = config.scale(50)
        input_h = config.scale(24)
        track_y_offset = config.scale(35)
        
        self.in_val = InputField(x+w-input_w, y, input_w, input_h, str(initial_val))
        self.rect_track = pygame.Rect(x+5, y+track_y_offset, w-10, config.scale(6))
        
        self.anim_hover = AnimVar(0.0)

    def set_position(self, x, y):
        """Override to also move internal sub-widgets."""
        super().set_position(x, y)
        if hasattr(self, 'in_val'):
            input_w = self.in_val.rect.w
            self.in_val.set_position(x + self.rect.w - input_w, y)
        if hasattr(self, 'rect_track'):
            self.rect_track.x = x + 5
            self.rect_track.y = y + config.scale(35)

    def update(self, dt):
        self.anim_hover.target = 1.0 if self.hovered or self.dragging else 0.0
        self.anim_hover.update(dt)
        # Update input field for cursor blinking
        self.in_val.update(dt)

    def handle_event(self, event):
        if not self.visible: return False
        
        changed = False
        
        # Pass event to input field first
        if self.in_val.handle_event(event):
            self.val = self.in_val.get_value(self.val)
            changed = True
            return True
        
        if event.type == pygame.MOUSEMOTION:
            # Expand hit area for easier grabbing
            self.hovered = self.rect_track.inflate(0, 14).collidepoint(event.pos)
            if self.dragging:
                rel = (event.pos[0] - self.rect_track.x) / self.rect_track.w
                self.val = self.min_val + max(0, min(1, rel)) * (self.max_val - self.min_val)
                changed = True
                return True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.dragging = True
                changed = True
                SoundManager.get().play_sound('click')
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True
            
        if changed and not self.in_val.active: 
            self.in_val.set_value(self.val)
            
        return False

    def draw(self, screen, font):
        if not self.visible: return
        
        # Label
        screen.blit(font.render(self.label, True, config.COLOR_TEXT_DIM), (self.rect.x, self.rect.y+2))
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

class MenuBar(UIElement):
    def __init__(self, w, h=30):
        super().__init__(0, 0, w, h)
        self.items = {"File": [], "Tools": [], "Help": []} 
        self.active_menu = None 
        self.dropdown_rect = None
        self.hover_item_idx = -1 
        self.item_rects = {}

    def resize(self, w):
        self.rect.width = w

    def handle_event(self, event):
        if not self.visible: return False
        
        curr_x = self.rect.x + 15
        self.item_rects = {}
        for key in self.items:
            width = 60 # approx
            self.item_rects[key] = pygame.Rect(curr_x, self.rect.y, width, self.rect.height)
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
                    return False 
                else: 
                    self.hover_item_idx = -1
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
                return item
            
            # Click outside closes menu
            if self.active_menu and self.dropdown_rect:
                 if not self.dropdown_rect.collidepoint(event.pos):
                     self.active_menu = None
                     return True
                
        return False

    def draw(self, screen, font):
        if not self.visible: return
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect)
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (0, self.rect.bottom-1), (self.rect.width, self.rect.bottom-1))
        
        curr_x = self.rect.x + 15
        for key in self.items:
            ts = font.render(key, True, config.COLOR_TEXT)
            width = ts.get_width() + 20
            r = pygame.Rect(curr_x, self.rect.y, width, self.rect.height)
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

class StatusBar(UIElement):
    """
    Footer bar that displays status messages.
    Integrates the core.StatusBar logic into the UI Tree.
    """
    def __init__(self, x, y, w, h, session):
        super().__init__(x, y, w, h)
        self.session = session

    def draw(self, screen, font):
        # Draw background bar (Footer)
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect)
        pygame.draw.line(screen, config.PANEL_BORDER_COLOR, (0, self.rect.top), (self.rect.width, self.rect.top))
        
        # Draw message if visible
        if self.session.status.is_visible:
            msg = self.session.status.message
            col = (100, 255, 100) # Green text like old renderer
            ts = font.render(msg, True, col)
            screen.blit(ts, (self.rect.x + 15, self.rect.centery - ts.get_height()//2))

class ContextMenu(UIElement):
    def __init__(self, x, y, options):
        width = 160
        height = len(options) * 30 + 10
        super().__init__(x, y, width, height)
        self.options = options
        self.selected_idx = -1
        self.action = None

    def handle_event(self, event):
        if not self.visible: return False

        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - (self.rect.y + 5)
                if rel_y >= 0: self.selected_idx = rel_y // 30
                else: self.selected_idx = -1
            else: self.selected_idx = -1
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                # Click inside menu - try to select an option
                rel_y = event.pos[1] - (self.rect.y + 5)
                idx = rel_y // 30
                if 0 <= idx < len(self.options):
                    self.action = self.options[idx]
                    SoundManager.get().play_sound('snap')
                # If click is inside but not on valid option, just consume it
            else:
                # Click outside menu - dismiss it
                self.action = "CLOSE"
            # CRITICAL: Always consume MOUSEBUTTONDOWN to prevent click-through
            return True
        return False

    def draw(self, screen, font):
        if not self.visible: return
        
        shadow = self.rect.copy(); shadow.x += 4; shadow.y += 4
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=4)
        screen.blit(s_surf, shadow)
        
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=4)
        pygame.draw.rect(screen, config.PANEL_BORDER_COLOR, self.rect, 1, border_radius=4)
        
        for i, opt in enumerate(self.options):
            bg_col = config.COLOR_ACCENT if i == self.selected_idx else config.PANEL_BG_COLOR
            item_rect = pygame.Rect(self.rect.x + 2, self.rect.y + 5 + i*30, self.rect.width - 4, 28)
            
            if i == self.selected_idx:
                pygame.draw.rect(screen, bg_col, item_rect, border_radius=3)
            
            col = config.COLOR_TEXT if i != self.selected_idx else (255, 255, 255)
            txt = font.render(opt, True, col)
            screen.blit(txt, (self.rect.x + 10, self.rect.y + 5 + i*30 + 5))

# =============================================================================
# DIALOGS (Floats)
# =============================================================================

class MaterialDialog:
    CREATE_NEW_OPTION = "[ Create New ]"

    # Color palette for cycling (same as MaterialPropertyWidget)
    COLOR_PALETTE = [
        (50, 150, 255),   # Blue (Water)
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (255, 200, 50),   # Yellow/Gold
        (180, 100, 255),  # Purple
        (255, 150, 50),   # Orange
        (100, 200, 200),  # Cyan
        (180, 180, 190),  # Silver
        (100, 100, 120),  # Gray (Wall)
    ]

    def __init__(self, x, y, sketch, current_material_id):
        self.rect = pygame.Rect(x, y, 300, 350)  # Taller to fit mass and color
        self.sketch = sketch
        self.done = False
        self.apply = False
        self.visible = True

        # Build dropdown options: "Create New" first, then existing materials
        self.material_names = list(sketch.materials.keys())
        dropdown_options = [self.CREATE_NEW_OPTION] + self.material_names

        # Determine initial selection
        if current_material_id in self.material_names:
            selected_idx = self.material_names.index(current_material_id) + 1  # +1 for Create New
        else:
            selected_idx = 0  # Create New

        mat = sketch.get_material(current_material_id)

        # Dropdown for selecting existing materials or creating new
        self.dropdown = Dropdown(x + 120, y + 45, 150, 25, dropdown_options, selected_idx)
        self.dropdown.on_change = self._on_material_selected

        # Material ID input (only visible when "Create New" is selected)
        self.in_id = InputField(x + 120, y + 80, 150, 25, mat.name if selected_idx == 0 else "")
        self.in_id.visible = (selected_idx == 0)

        self.in_sigma = InputField(x + 120, y + 115, 150, 25, str(mat.sigma))
        self.in_epsilon = InputField(x + 120, y + 150, 150, 25, str(mat.epsilon))
        self.in_mass = InputField(x + 120, y + 185, 150, 25, str(mat.mass))
        self.in_spacing = InputField(x + 120, y + 220, 150, 25, str(mat.spacing))

        # Color swatch
        self.color = mat.color
        self.color_rect = pygame.Rect(x + 120, y + 255, 30, 25)
        self._color_index = self._find_color_index(mat.color)

        self.btn_apply = Button(x + 20, y + 300, 80, 30, "Apply", toggle=False)
        self.btn_ok = Button(x + 180, y + 300, 80, 30, "OK", toggle=False)

        # Track mode
        self.is_create_new = (selected_idx == 0)

    def _find_color_index(self, color):
        """Find index of color in palette, or 0 if not found."""
        if color in self.COLOR_PALETTE:
            return self.COLOR_PALETTE.index(color)
        return 0

    def _on_material_selected(self, index, option):
        """Called when dropdown selection changes."""
        if option == self.CREATE_NEW_OPTION:
            # Create New mode - show editable name field
            self.is_create_new = True
            self.in_id.visible = True
            self.in_id.text = ""
            self.in_id.cursor_pos = 0
            # Reset to defaults
            self.in_sigma.set_value(1.0)
            self.in_epsilon.set_value(1.0)
            self.in_mass.set_value(1.0)
            self.in_spacing.set_value(0.7)
            self.color = (200, 200, 200)
            self._color_index = 0
        else:
            # Existing material selected
            self.is_create_new = False
            self.in_id.visible = False
            m = self.sketch.materials.get(option)
            if m:
                self.in_sigma.set_value(m.sigma)
                self.in_epsilon.set_value(m.epsilon)
                self.in_mass.set_value(m.mass)
                self.in_spacing.set_value(m.spacing)
                self.color = m.color
                self._color_index = self._find_color_index(m.color)

    def handle_event(self, event):
        if not self.visible:
            return False

        # Handle dropdown first (it may be expanded and overlay other widgets)
        if self.dropdown.handle_event(event):
            return True

        # Handle other inputs only if dropdown is not expanded
        if not self.dropdown.expanded:
            if self.in_id.visible and self.in_id.handle_event(event):
                return True

            if self.in_sigma.handle_event(event):
                return True
            if self.in_epsilon.handle_event(event):
                return True
            if self.in_mass.handle_event(event):
                return True
            if self.in_spacing.handle_event(event):
                return True

            # Handle color swatch click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.color_rect.collidepoint(event.pos):
                    self._color_index = (self._color_index + 1) % len(self.COLOR_PALETTE)
                    self.color = self.COLOR_PALETTE[self._color_index]
                    return True

            if self.btn_apply.handle_event(event):
                self.apply = True
                return True
            if self.btn_ok.handle_event(event):
                self.apply = True
                self.done = True
                return True

        # Absorb all mouse events inside dialog (including MOUSEBUTTONUP)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        # Absorb mouse motion to prevent hover effects on elements behind dialog
        if event.type == pygame.MOUSEMOTION:
            return True

        return False

    def update(self, dt):
        # Update input fields for cursor blinking
        self.in_id.update(dt)
        self.in_sigma.update(dt)
        self.in_epsilon.update(dt)
        self.in_mass.update(dt)
        self.in_spacing.update(dt)
        # Update buttons
        self.btn_apply.update(dt)
        self.btn_ok.update(dt)

    def get_result(self):
        if self.is_create_new:
            name = self.in_id.get_text()
            if not name:
                name = "Custom"
        else:
            # Use the selected material name from dropdown
            name = self.dropdown.get_selected()

        m = Material(
            name,
            sigma=self.in_sigma.get_value(1.0),
            epsilon=self.in_epsilon.get_value(1.0),
            mass=self.in_mass.get_value(1.0),
            spacing=self.in_spacing.get_value(0.7),
            color=self.color,
            physical=True
        )
        return m

    def draw(self, screen, font):
        if not self.visible:
            return

        shadow = self.rect.copy()
        shadow.x += 5
        shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)

        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)

        title = font.render("Material Editor", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 10))

        # Labels - adjust based on whether we're in create new mode
        screen.blit(font.render("Material:", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 50))

        if self.is_create_new:
            screen.blit(font.render("New Name:", True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + 85))

        labels_with_ys = [("Sigma:", 120), ("Epsilon:", 155), ("Mass:", 190), ("Spacing:", 225), ("Color:", 260)]
        for label, y in labels_with_ys:
            screen.blit(font.render(label, True, config.COLOR_TEXT), (self.rect.x + 20, self.rect.y + y))

        # Draw widgets (dropdown last so its expanded list appears on top)
        if self.in_id.visible:
            self.in_id.draw(screen, font)
        self.in_sigma.draw(screen, font)
        self.in_epsilon.draw(screen, font)
        self.in_mass.draw(screen, font)
        self.in_spacing.draw(screen, font)

        # Draw color swatch
        pygame.draw.rect(screen, self.color, self.color_rect)
        pygame.draw.rect(screen, config.COLOR_TEXT, self.color_rect, 1)

        self.btn_apply.draw(screen, font)
        self.btn_ok.draw(screen, font)

        # Draw dropdown last so expanded list renders on top
        self.dropdown.draw(screen, font)

class RotationDialog:
    def __init__(self, x, y, anim_data):
        self.rect = pygame.Rect(x, y, 250, 210)
        self.done = False
        self.apply = False
        self.visible = True
        
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
        if not self.visible: return False
        
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

        # Absorb all mouse events inside dialog (including MOUSEBUTTONUP)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True
        if event.type == pygame.MOUSEMOTION:
            return True
        return False

    def update(self, dt):
        self.in_speed.update(dt)
        self.btn_pivot.update(dt)
        self.btn_apply.update(dt)
        self.btn_ok.update(dt)

    def get_values(self):
        return { 'speed': self.in_speed.get_value(0.0), 'pivot': self.pivots[self.pivot_idx] }

    def draw(self, screen, font):
        if not self.visible: return
        
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

class SaveAsNewDialog:
    """Simple dialog for naming a new material when using 'Save as New'."""

    def __init__(self, x, y, suggested_name="Custom", existing_names=None):
        self.rect = pygame.Rect(x, y, 280, 150)
        self.done = False
        self.cancelled = False
        self.visible = True
        self.existing_names = existing_names or set()

        # Input field for material name (positioned after "Name:" label)
        # Dialog is 280 wide, label ~50px + padding, input ~190 wide
        self.in_name = InputField(x + 70, y + 45, 190, 28, suggested_name)
        self.in_name.active = True  # Start with input focused
        self.in_name.cursor_pos = len(suggested_name)

        # Buttons (positioned at bottom of dialog)
        btn_w = 100
        self.btn_cancel = Button(x + 20, y + 105, btn_w, 30, "Cancel",
                                 toggle=False, color_inactive=(80, 80, 90))
        self.btn_ok = Button(x + 160, y + 105, btn_w, 30, "OK",
                             toggle=False, color_inactive=config.COLOR_SUCCESS)

        self.error_message = None

    def has_active_input(self):
        """Check if any input field in this dialog is active (for hotkey blocking)."""
        return self.in_name.active

    def handle_event(self, event):
        if not self.visible:
            return False

        # Handle mouse clicks - check buttons FIRST so they work even when input is focused
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check Cancel button
            if self.btn_cancel.rect.collidepoint(event.pos):
                self.btn_cancel.handle_event(event)
                self.cancelled = True
                self.done = True
                return True

            # Check OK button
            if self.btn_ok.rect.collidepoint(event.pos):
                self.btn_ok.handle_event(event)
                return self._try_accept()

        # Handle keyboard BEFORE input field so Enter/Escape work correctly
        if event.type == pygame.KEYDOWN:
            # Enter to accept (intercept before InputField consumes it)
            if event.key == pygame.K_RETURN:
                return self._try_accept()
            # Escape to cancel
            elif event.key == pygame.K_ESCAPE:
                self.cancelled = True
                self.done = True
                return True

        # Handle input field for text editing and focus changes
        if self.in_name.handle_event(event):
            self.error_message = None  # Clear error on edit
            return True

        # Block ALL other keyboard events from reaching global hotkeys
        if event.type == pygame.KEYDOWN:
            return True

        # Absorb all mouse events inside dialog (including MOUSEBUTTONUP)
        # This prevents clicks from falling through to widgets behind the dialog
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True

        # Absorb mouse motion to prevent hover effects on elements behind dialog
        if event.type == pygame.MOUSEMOTION:
            return True

        return False

    def _try_accept(self):
        """Validate and accept the name."""
        name = self.in_name.get_text().strip()

        if not name:
            self.error_message = "Name cannot be empty"
            return True

        if name in self.existing_names:
            self.error_message = "Name already exists"
            return True

        self.error_message = None
        self.done = True
        return True

    def update(self, dt):
        self.in_name.update(dt)
        self.btn_cancel.update(dt)
        self.btn_ok.update(dt)

    def get_name(self):
        """Get the entered name (only valid if done and not cancelled)."""
        if self.cancelled:
            return None
        return self.in_name.get_text().strip()

    def draw(self, screen, font):
        if not self.visible:
            return

        # Shadow
        shadow = self.rect.copy()
        shadow.x += 5
        shadow.y += 5
        s_surf = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(s_surf, (0, 0, 0, 100), s_surf.get_rect(), border_radius=6)
        screen.blit(s_surf, shadow)

        # Background
        pygame.draw.rect(screen, config.PANEL_BG_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(screen, config.COLOR_ACCENT, self.rect, 1, border_radius=6)

        # Title
        title = font.render("Save as New Material", True, (255, 255, 255))
        screen.blit(title, (self.rect.x + 15, self.rect.y + 12))

        # Label
        label = font.render("Name:", True, config.COLOR_TEXT)
        screen.blit(label, (self.rect.x + 20, self.rect.y + 50))

        # Input field (position is set in __init__, just draw it)
        self.in_name.draw(screen, font)

        # Error message (below input field)
        if self.error_message:
            err_surf = font.render(self.error_message, True, config.COLOR_DANGER)
            screen.blit(err_surf, (self.rect.x + 70, self.rect.y + 78))

        # Buttons
        self.btn_cancel.draw(screen, font)
        self.btn_ok.draw(screen, font)


class AnimationDialog:
    def __init__(self, x, y, driver_data):
        self.rect = pygame.Rect(x, y, 300, 280)
        self.driver = driver_data if driver_data else {'type': 'sin', 'amp': 15.0, 'freq': 0.5, 'phase': 0.0, 'rate': 10.0}
        self.done = False
        self.apply = False
        self.visible = True
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
        if not self.visible: return False
        
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

        # Absorb all mouse events inside dialog (including MOUSEBUTTONUP)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                return True
        if event.type == pygame.MOUSEMOTION:
            return True
        return False

    def update(self, dt):
        # Update input fields for cursor blinking
        self.in_amp.update(dt)
        self.in_freq.update(dt)
        self.in_phase.update(dt)
        self.in_rate.update(dt)
        # Update buttons
        self.btn_tab_sin.update(dt)
        self.btn_tab_lin.update(dt)
        self.btn_stop.update(dt)
        self.btn_ok.update(dt)

    def get_values(self): return self.driver

    def draw(self, screen, font):
        if not self.visible: return
        
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


# =============================================================================
# MATERIAL PROPERTY WIDGET (Embedded Panel)
# =============================================================================

class MaterialPropertyWidget(UIContainer):
    """
    Embedded widget for viewing and editing material properties.

    Features:
    - Dropdown to select from material library
    - Sliders for sigma, epsilon, mass
    - Color swatch (clickable to cycle colors)
    - Save button to add current settings as new preset

    Context-Aware Behavior:
    - When entity is selected: edits selected entity's material directly
    - When nothing is selected: edits session.active_material (brush default)

    Live Physics:
    - Slider changes trigger immediate compiler.rebuild() for real-time updates
    """

    # Color palette for cycling through on swatch click
    COLOR_PALETTE = [
        (50, 150, 255),   # Blue (Water)
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (255, 200, 50),   # Yellow/Gold
        (180, 100, 255),  # Purple
        (255, 150, 50),   # Orange
        (100, 200, 200),  # Cyan
        (180, 180, 190),  # Silver
        (100, 100, 120),  # Gray (Wall)
    ]

    def __init__(self, x, y, w, session, controller=None):
        # Calculate total height for all sub-widgets
        s = config.scale
        slider_h = s(50)  # Compact slider height
        row_h = s(28)
        total_h = row_h + slider_h * 4 + row_h + s(10)  # dropdown + 4 sliders + save row + padding

        super().__init__(x, y, w, total_h, layout_type='vertical', padding=0, spacing=s(4))
        self.session = session
        self.controller = controller  # For accessing scene, sketch, compiler
        self._width = w
        self._color_index = 0  # For cycling through palette
        self._last_selection_hash = None  # Track selection changes
        self._last_material_id = None  # Track material for revert on deselect
        self._last_entity_idx = None  # Track entity index for clearing previews
        # Track which library material is selected in dropdown (for no-entity-selected mode)
        self._dropdown_material_id = None
        # Track if current material has unsaved modifications (for display only)
        self._is_current_modified = False
        # Callback for requesting "Save as New" dialog (set by controller)
        self.on_save_as_new_request = None
        # Pending save data (stored while dialog is open)
        self._pending_save_as_new = None

        # Material dropdown (from library)
        self._build_dropdown()

        # Property sliders (compact version)
        self.slider_sigma = self._create_mini_slider("Sigma", 0.5, 2.0, 1.0)
        self.slider_epsilon = self._create_mini_slider("Epsilon", 0.1, 3.0, 1.0)
        self.slider_mass = self._create_mini_slider("Mass", 0.1, 20.0, 1.0)
        self.slider_spacing = self._create_mini_slider("Spacing", 0.3, 2.0, 0.7)

        self.add_child(self.slider_sigma)
        self.add_child(self.slider_epsilon)
        self.add_child(self.slider_mass)
        self.add_child(self.slider_spacing)

        # Save buttons row
        self._build_save_row()

        # Sync initial values
        self._sync_from_material(session.active_material)

    def _build_dropdown(self):
        """Build the material library dropdown."""
        s = config.scale
        options = self._get_material_names()
        self.dropdown = Dropdown(0, 0, self._width, s(25), options, selected_index=0)
        self.dropdown.on_change = self._on_material_selected
        self.add_child(self.dropdown)
        # Initialize dropdown material ID to first material
        if options:
            self._dropdown_material_id = self._strip_modified_suffix(options[0])

    def _get_material_names(self):
        """Get material names from sketch (if available) or session library."""
        if self.controller and hasattr(self.controller, 'sketch'):
            return list(self.controller.sketch.materials.keys())
        return list(self.session.material_library.keys())

    def _strip_modified_suffix(self, name):
        """Strip ' (modified)' suffix from material name if present."""
        if name and name.endswith(' (modified)'):
            return name[:-11]  # len(' (modified)') == 11
        return name

    def _get_material_library(self):
        """Get material library from sketch (if available) or session."""
        if self.controller and hasattr(self.controller, 'sketch'):
            return self.controller.sketch.materials
        return self.session.material_library

    def _get_material_manager(self):
        """Get the MaterialManager from scene (if available)."""
        if self.controller and hasattr(self.controller, 'scene'):
            return getattr(self.controller.scene, 'material_manager', None)
        return None

    def _get_selected_entity(self):
        """Get the first selected entity, or None if nothing selected."""
        if not self.controller:
            return None
        selection = self.session.selection
        if selection.has_entities:
            entity_idx = next(iter(selection.entities))
            sketch = self.controller.sketch
            if 0 <= entity_idx < len(sketch.entities):
                return sketch.entities[entity_idx]
        return None

    def _get_selected_entity_index(self):
        """Get the index of the first selected entity, or None if nothing selected."""
        if not self.controller:
            return None
        selection = self.session.selection
        if selection.has_entities:
            entity_idx = next(iter(selection.entities))
            sketch = self.controller.sketch
            if 0 <= entity_idx < len(sketch.entities):
                return entity_idx
        return None

    def _get_target_material(self):
        """
        Get the material to edit based on context.
        Returns (material, is_entity_material) tuple.

        When entity is selected: Returns entity's library material
        When no entity selected: Returns library material selected in dropdown
        """
        entity = self._get_selected_entity()
        library = self._get_material_library()

        if entity:
            # Entity selected - edit its material from library
            mat_id = getattr(entity, 'material_id', 'Wall')
            if mat_id in library:
                return library[mat_id], True
            # Fallback
            return self.session.active_material, False

        # No entity selected - edit the library material selected in dropdown
        if self._dropdown_material_id and self._dropdown_material_id in library:
            return library[self._dropdown_material_id], False

        # Fallback to first material in library or session active
        if library:
            first_mat_id = next(iter(library.keys()))
            self._dropdown_material_id = first_mat_id
            return library[first_mat_id], False

        return self.session.active_material, False

    def _trigger_rebuild(self):
        """Trigger physics recompilation for live updates."""
        if self.controller and hasattr(self.controller, 'scene'):
            self.controller.scene.rebuild()

    def _build_save_row(self):
        """Build the save/color row with Save and Save as New buttons."""
        s = config.scale
        row = UIContainer(0, 0, self._width, s(28), layout_type='horizontal', padding=0, spacing=s(4))

        # Color swatch (simple colored rectangle)
        self.color_swatch = ColorSwatch(0, 0, s(28), s(28), (50, 150, 255))
        row.add_child(self.color_swatch)

        # Calculate button widths (equal split of remaining space)
        remaining_w = self._width - s(32) - s(4)  # swatch + spacing
        btn_w = (remaining_w - s(4)) // 2  # Two buttons with spacing

        # Save button - saves changes to current material
        self.btn_save = Button(0, 0, btn_w, s(28), "Save",
                               toggle=False, color_inactive=config.COLOR_SUCCESS)
        row.add_child(self.btn_save)

        # Save as New button - opens dialog to save with new name
        self.btn_save_new = Button(0, 0, btn_w, s(28), "Save as",
                                   toggle=False, color_inactive=(80, 80, 90))
        row.add_child(self.btn_save_new)

        self.add_child(row)

    def _create_mini_slider(self, label, min_v, max_v, init_v):
        """Create a compact slider for material properties."""
        s = config.scale
        return MiniSlider(0, 0, self._width, min_v, max_v, init_v, label)

    def _on_material_selected(self, index, name):
        """Called when user selects a material from the dropdown."""
        # Strip "(modified)" suffix if present (for display names)
        clean_name = self._strip_modified_suffix(name)
        library = self._get_material_library()
        mat_mgr = self._get_material_manager()

        if clean_name in library:
            mat = library[clean_name]
            entity = self._get_selected_entity()

            # Clear any pending global preview for the OLD material
            old_mat_id = self._dropdown_material_id
            if mat_mgr and old_mat_id and old_mat_id != clean_name:
                if mat_mgr.has_global_override(old_mat_id):
                    mat_mgr.clear_global_preview(old_mat_id)
                    self._trigger_rebuild()

            # Track which material is selected in dropdown
            self._dropdown_material_id = clean_name

            if entity:
                # Apply to selected entity
                entity.material_id = clean_name
                self._trigger_rebuild()
            else:
                # No entity selected - also update brush default
                self.session.active_material = mat.copy()

            self._sync_from_material(mat)
            self._update_dropdown_display()

    def _sync_from_material(self, mat):
        """Update sliders and swatch to match material."""
        self.slider_sigma.set_value(mat.sigma)
        self.slider_epsilon.set_value(mat.epsilon)
        self.slider_mass.set_value(mat.mass)
        self.slider_spacing.set_value(mat.spacing)
        self.color_swatch.color = mat.color
        # Update color index to match
        if mat.color in self.COLOR_PALETTE:
            self._color_index = self.COLOR_PALETTE.index(mat.color)

    def _sync_to_material(self):
        """
        Update target material from slider values and trigger rebuild.

        Selection-aware behavior:
        - Entity selected: Update per-entity preview override (only affects selected entity)
        - No entity selected: Update global preview override (affects all entities with that material)
        """
        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()
        mat_mgr = self._get_material_manager()

        if entity is not None and entity_idx is not None and mat_mgr:
            # Entity selected: Use per-entity preview override
            material_id = getattr(entity, 'material_id', 'Wall')

            # Ensure preview is started
            if not mat_mgr.has_entity_override(entity_idx):
                mat_mgr.begin_entity_preview(entity_idx, material_id)

            # Update preview values
            mat_mgr.update_entity_preview(
                entity_idx,
                sigma=self.slider_sigma.val,
                epsilon=self.slider_epsilon.val,
                mass=self.slider_mass.val,
                spacing=self.slider_spacing.val,
                color=self.color_swatch.color
            )

            # Trigger rebuild to show preview (only affects this entity)
            self._trigger_rebuild()
            # Update dropdown to show "(modified)"
            self._update_dropdown_display()

        elif mat_mgr and self._dropdown_material_id:
            # No entity selected: Use global preview override
            material_id = self._dropdown_material_id

            # Ensure global preview is started
            if not mat_mgr.has_global_override(material_id):
                mat_mgr.begin_global_preview(material_id)

            # Update global preview values
            mat_mgr.update_global_preview(
                material_id,
                sigma=self.slider_sigma.val,
                epsilon=self.slider_epsilon.val,
                mass=self.slider_mass.val,
                spacing=self.slider_spacing.val,
                color=self.color_swatch.color
            )

            # Also update session active material for brush (as a convenience copy)
            override = mat_mgr.get_global_override(material_id)
            if override:
                self.session.active_material = override.copy()

            # Trigger rebuild for global preview
            self._trigger_rebuild()
            self._update_dropdown_display()

    def handle_event(self, event):
        """Handle events and sync material on slider changes."""
        # Handle dropdown first (may be expanded)
        if self.dropdown.handle_event(event):
            return True

        if self.dropdown.expanded:
            return False  # Don't process other widgets when dropdown is open

        # Handle sliders
        for slider in [self.slider_sigma, self.slider_epsilon, self.slider_mass, self.slider_spacing]:
            if slider.handle_event(event):
                self._sync_to_material()
                return True

        # Handle save button - saves to current material
        if self.btn_save.handle_event(event):
            self._save_material()
            return True

        # Handle save as new button - prompts for new name
        if self.btn_save_new.handle_event(event):
            self._save_as_new_material()
            return True

        # Handle color swatch click - cycle through colors
        if event.type == pygame.MOUSEBUTTONDOWN and self.color_swatch.rect.collidepoint(event.pos):
            self._cycle_color()
            return True

        if self.color_swatch.handle_event(event):
            return True

        return super().handle_event(event)

    def _cycle_color(self):
        """Cycle to the next color in the palette."""
        self._color_index = (self._color_index + 1) % len(self.COLOR_PALETTE)
        new_color = self.COLOR_PALETTE[self._color_index]
        self.color_swatch.color = new_color

        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()
        mat_mgr = self._get_material_manager()

        if entity is not None and entity_idx is not None and mat_mgr:
            # Entity selected: Update per-entity preview override
            material_id = getattr(entity, 'material_id', 'Wall')

            if not mat_mgr.has_entity_override(entity_idx):
                mat_mgr.begin_entity_preview(entity_idx, material_id)

            mat_mgr.update_entity_preview(entity_idx, color=new_color)
            self._trigger_rebuild()
            self._update_dropdown_display()
        elif mat_mgr and self._dropdown_material_id:
            # No entity selected: Use global preview override
            material_id = self._dropdown_material_id

            if not mat_mgr.has_global_override(material_id):
                mat_mgr.begin_global_preview(material_id)

            mat_mgr.update_global_preview(material_id, color=new_color)

            # Update session active material for brush
            override = mat_mgr.get_global_override(material_id)
            if override:
                self.session.active_material.color = new_color

            self._trigger_rebuild()
            self._update_dropdown_display()

        SoundManager.get().play_sound('snap')

    def _save_material(self):
        """
        Save current slider values to the selected material in the library.

        Selection-aware behavior:
        - Entity selected: Commit per-entity preview to library (affects all entities with that material)
        - No entity selected: Commit global preview to library (affects all entities with that material)
        """
        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()
        mat_mgr = self._get_material_manager()

        if entity is not None and entity_idx is not None and mat_mgr:
            # Entity selected: Commit the per-entity preview to the library
            material_id = getattr(entity, 'material_id', 'Wall')

            if mat_mgr.has_entity_override(entity_idx):
                # Commit preview values to library material
                mat_mgr.commit_entity_preview(entity_idx, material_id)
                # Trigger rebuild so all entities with this material update
                self._trigger_rebuild()
                self._update_dropdown_display()
                SoundManager.get().play_sound('snap')
                self.session.status.set(f"Saved: {material_id}")
            else:
                # No changes to commit
                SoundManager.get().play_sound('snap')
        elif mat_mgr and self._dropdown_material_id:
            # No entity selected: Commit global preview to library
            mat_id = self._dropdown_material_id

            if mat_mgr.has_global_override(mat_id):
                mat_mgr.commit_global_preview(mat_id)
                self._trigger_rebuild()

            self._update_dropdown_display()
            SoundManager.get().play_sound('snap')
            self.session.status.set(f"Saved: {mat_id}")

    def _save_as_new_material(self):
        """
        Request the Save as New dialog to get a name for the new material.

        Stores pending save data and calls the on_save_as_new_request callback
        to show the dialog. Actual save is done in complete_save_as_new().
        """
        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()
        library = self._get_material_library()

        # Generate a suggested name
        base_name = "Custom"
        counter = 1
        suggested_name = base_name
        while suggested_name in library:
            suggested_name = f"{base_name} {counter}"
            counter += 1

        # Store pending save data
        self._pending_save_as_new = {
            'entity_idx': entity_idx,
            'old_mat_id': getattr(entity, 'material_id', 'Wall') if entity else self._dropdown_material_id,
            'sigma': self.slider_sigma.val,
            'epsilon': self.slider_epsilon.val,
            'mass': self.slider_mass.val,
            'spacing': self.slider_spacing.val,
            'color': self.color_swatch.color,
            'suggested_name': suggested_name,
        }

        # Request dialog via callback
        if self.on_save_as_new_request:
            self.on_save_as_new_request(suggested_name, set(library.keys()))
        else:
            # Fallback if no callback set - use suggested name directly
            self.complete_save_as_new(suggested_name)

    def complete_save_as_new(self, new_name):
        """
        Complete the Save as New operation with the provided name.

        Called by the controller after the dialog is completed.

        Args:
            new_name: The name for the new material (None if cancelled)
        """
        if not self._pending_save_as_new:
            return

        pending = self._pending_save_as_new
        self._pending_save_as_new = None

        if not new_name:
            # Cancelled - do nothing
            return

        entity_idx = pending['entity_idx']
        old_mat_id = pending['old_mat_id']
        mat_mgr = self._get_material_manager()
        library = self._get_material_library()

        # Create new material with pending values
        new_mat = Material(
            new_name,
            sigma=pending['sigma'],
            epsilon=pending['epsilon'],
            mass=pending['mass'],
            spacing=pending['spacing'],
            color=pending['color'],
            physical=True
        )
        library[new_name] = new_mat

        if entity_idx is not None:
            # Entity was selected: Assign new material to entity
            entity = self._get_selected_entity()
            if entity:
                entity.material_id = new_name
                self._dropdown_material_id = new_name
                self._last_material_id = new_name

            # Clear any entity preview
            if mat_mgr and mat_mgr.has_entity_override(entity_idx):
                mat_mgr.clear_entity_preview(entity_idx)

            self._trigger_rebuild()
            self._update_dropdown_display()
            SoundManager.get().play_sound('snap')
            self.session.status.set(f"Created: {new_name}")
        else:
            # No entity selected: Create as preset, clear global preview
            if mat_mgr and old_mat_id:
                mat_mgr.clear_global_preview(old_mat_id)

            self._trigger_rebuild()
            self._update_dropdown_display()

            # Sync sliders to show the library values (global preview was cleared)
            if old_mat_id and old_mat_id in library:
                self._sync_from_material(library[old_mat_id])

            SoundManager.get().play_sound('snap')
            self.session.status.set(f"Created preset: {new_name}")

    def update(self, dt):
        """Update child widgets and check for selection changes."""
        # Check if selection changed
        self._check_selection_change()

        self.dropdown.hovered = False  # Reset hover state
        for child in self.children:
            if hasattr(child, 'update'):
                child.update(dt)

    def wants_focus(self):
        """MaterialPropertyWidget wants focus for global preview tracking."""
        return True

    def on_focus_lost(self):
        """
        Clear global preview when focus is lost (user clicked outside widget).

        Called by InputHandler when a click occurs outside this widget.
        Uses borrowed focus: if a modal is open, focus is not truly lost.
        """
        # Don't clear preview while a modal dialog is open (borrowed focus)
        if self.controller:
            actions = getattr(self.controller, 'actions', None)
            if actions and actions.is_modal_active():
                return

        # Only clear global preview (not entity-specific preview)
        entity = self._get_selected_entity()
        if entity is not None:
            return  # Entity selected, not using global preview

        mat_mgr = self._get_material_manager()
        if mat_mgr and self._dropdown_material_id:
            if mat_mgr.has_global_override(self._dropdown_material_id):
                mat_mgr.clear_global_preview(self._dropdown_material_id)
                self._trigger_rebuild()
                self._update_dropdown_display()
                # Sync sliders back to library values
                library = self._get_material_library()
                if self._dropdown_material_id in library:
                    self._sync_from_material(library[self._dropdown_material_id])

    def _check_selection_change(self):
        """Check if selection changed and update widget accordingly."""
        # Create a hash of current selection state
        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()

        if entity:
            current_hash = (id(entity), getattr(entity, 'material_id', None))
            current_mat_id = getattr(entity, 'material_id', None)
        else:
            current_hash = None
            # Track dropdown material for no-entity case
            current_mat_id = self._dropdown_material_id

        if current_hash != self._last_selection_hash:
            # Selection changed - clear any pending previews
            mat_mgr = self._get_material_manager()

            if mat_mgr and self._last_entity_idx is not None:
                # Clear the entity preview for the previously selected entity
                if mat_mgr.has_entity_override(self._last_entity_idx):
                    mat_mgr.clear_entity_preview(self._last_entity_idx)
                    self._trigger_rebuild()  # Rebuild to show reverted values

            # Also clear global preview if we were editing without selection
            if mat_mgr and self._last_material_id and self._last_entity_idx is None:
                if mat_mgr.has_global_override(self._last_material_id):
                    mat_mgr.clear_global_preview(self._last_material_id)
                    self._trigger_rebuild()

            self._last_selection_hash = current_hash
            self._last_material_id = current_mat_id
            self._last_entity_idx = entity_idx
            self._on_selection_changed()

    def _on_selection_changed(self):
        """Called when selection changes - sync widget to new target."""
        entity = self._get_selected_entity()
        library = self._get_material_library()

        if entity:
            # Entity selected - sync dropdown to entity's material
            mat_id = getattr(entity, 'material_id', 'Wall')
            self._dropdown_material_id = mat_id
        # else: Keep current dropdown selection when deselecting

        # Get the material to display (library material, not any override)
        mat, _ = self._get_target_material()
        self._sync_from_material(mat)

        # Update dropdown to show selected material
        self._update_dropdown_display()

    def _update_dropdown_display(self):
        """Update dropdown options and track modification state.

        Dropdown options always contain clean material IDs (never with suffix).
        The "(modified)" indicator is tracked separately and displayed in draw().
        """
        mat_mgr = self._get_material_manager()
        library = self._get_material_library()

        # Get current entity and material id
        entity = self._get_selected_entity()
        entity_idx = self._get_selected_entity_index()

        # Determine which material should be selected in dropdown
        if entity:
            target_mat_id = getattr(entity, 'material_id', None)
        else:
            target_mat_id = self._dropdown_material_id

        # Build clean options list (no suffix!)
        new_options = list(library.keys())
        selected_idx = 0

        # Find selected index
        for i, name in enumerate(new_options):
            if name == target_mat_id:
                selected_idx = i
                break

        # Check if current material is modified (for display purposes)
        self._is_current_modified = False
        if mat_mgr and target_mat_id:
            if entity_idx is not None:
                # Entity selected: Check entity-specific modification
                self._is_current_modified = mat_mgr.is_entity_modified(entity_idx, target_mat_id)
            else:
                # No entity selected: Check global override modification
                self._is_current_modified = mat_mgr.is_global_modified(target_mat_id)

        # Update dropdown with clean options
        if hasattr(self.dropdown, 'options'):
            self.dropdown.options = new_options
            self.dropdown.selected_index = selected_idx
            self.dropdown.cached_surf = None  # Force re-render

    def refresh_from_selection(self):
        """Public method to force refresh from current selection."""
        self._on_selection_changed()

    def draw(self, screen, font):
        """Draw the widget."""
        if not self.visible:
            return

        # Draw section header (context-aware)
        entity = self._get_selected_entity()
        if entity:
            header_text = "Selected Material"
            header_color = config.COLOR_ACCENT
        else:
            header_text = "Brush Material"
            header_color = config.COLOR_TEXT_DIM

        # Add "(modified)" to header if material has unsaved changes
        if self._is_current_modified:
            header_text += " (modified)"

        header = font.render(header_text, True, header_color)
        screen.blit(header, (self.rect.x, self.rect.y - config.scale(18)))

        # Draw children
        super().draw(screen, font)


class MiniSlider(UIElement):
    """Compact slider for property editing (single row)."""

    def __init__(self, x, y, w, min_val, max_val, initial_val, label):
        s = config.scale
        h = s(50)
        super().__init__(x, y, w, h)
        self.val = initial_val
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.dragging = False
        self.hovered = False

        # Input field for direct value entry
        input_w = s(45)
        input_h = s(20)
        self.in_val = InputField(x + w - input_w, y, input_w, input_h, f"{initial_val:.2f}")

        # Track rect (calculated in draw)
        track_y = s(30)
        self.rect_track = pygame.Rect(x + 5, y + track_y, w - 10, s(6))

        self.anim_hover = AnimVar(0.0)

    def set_value(self, val):
        """Set the slider value programmatically."""
        self.val = max(self.min_val, min(self.max_val, val))
        self.in_val.set_value(self.val)

    def set_position(self, x, y):
        """Override to move internal widgets."""
        super().set_position(x, y)
        s = config.scale
        input_w = self.in_val.rect.w
        self.in_val.set_position(x + self.rect.w - input_w, y)
        self.rect_track.x = x + 5
        self.rect_track.y = y + s(30)

    def update(self, dt):
        self.anim_hover.target = 1.0 if self.hovered or self.dragging else 0.0
        self.anim_hover.update(dt)
        self.in_val.update(dt)

    def handle_event(self, event):
        if not self.visible:
            return False

        # Input field first
        if self.in_val.handle_event(event):
            self.val = self.in_val.get_value(self.val)
            self.val = max(self.min_val, min(self.max_val, self.val))
            return True

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect_track.inflate(0, 14).collidepoint(event.pos)
            if self.dragging:
                rel = (event.pos[0] - self.rect_track.x) / self.rect_track.w
                self.val = self.min_val + max(0, min(1, rel)) * (self.max_val - self.min_val)
                if not self.in_val.active:
                    self.in_val.set_value(self.val)
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.dragging = True
                SoundManager.get().play_sound('click')
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True

        return False

    def draw(self, screen, font):
        if not self.visible:
            return

        # Label
        label_surf = font.render(self.label, True, config.COLOR_TEXT_DIM)
        screen.blit(label_surf, (self.rect.x, self.rect.y + 2))

        # Input field
        self.in_val.draw(screen, font)

        # Track background
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
            fill_col = tuple(min(255, c + 40) for c in fill_col)

        pygame.draw.rect(screen, fill_col, fill_rect, border_radius=3)

        # Handle
        handle_x = self.rect_track.x + self.rect_track.w * pct
        h_rad = 5 + 2 * self.anim_hover.value
        pygame.draw.circle(screen, (240, 240, 240), (int(handle_x), self.rect_track.centery), int(h_rad))


class ColorSwatch(UIElement):
    """Simple colored rectangle that displays a color."""

    def __init__(self, x, y, w, h, color=(200, 200, 200)):
        super().__init__(x, y, w, h)
        self.color = color
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                # Could open color picker in future
                return True
        return False

    def draw(self, screen, font):
        if not self.visible:
            return

        # Draw color fill
        pygame.draw.rect(screen, self.color, self.rect, border_radius=4)

        # Draw border
        border_col = config.COLOR_ACCENT if self.hovered else config.PANEL_BORDER_COLOR
        pygame.draw.rect(screen, border_col, self.rect, 1, border_radius=4)