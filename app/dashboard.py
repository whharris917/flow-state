import pygame
import sys
import subprocess
import os

# Try to import config to respect global settings if available
try:
    import core.config as config

except ImportError:
    config = None

class MenuButton:
    def __init__(self, x, y, w, h, text, action_code):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action_code = action_code
        self.color = (60, 60, 60)
        self.hover_color = (80, 80, 90)
    
    def draw(self, screen, font):
        mouse_pos = pygame.mouse.get_pos()
        col = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(screen, col, self.rect, border_radius=8)
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 1, border_radius=8)
        
        txt_surf = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class Dashboard:
    def __init__(self):
        # Initialize mixer before pygame for high quality audio
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        
        # --- Environment Setup ---
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Background Image Setup ---
        self.bg_image = None
        self.width = 800  # Default width
        self.height = 600 # Default height
        
        bg_path = os.path.join(self.base_dir, "..", "art", "colorful_vessels.jpg")
        if os.path.exists(bg_path):
            try:
                self.bg_image = pygame.image.load(bg_path)
                bg_w, bg_h = self.bg_image.get_size()
                # Ensure window is at least as large as the background image
                self.width = max(self.width, bg_w)
                self.height = max(self.height, bg_h)
                print(f"Loaded background: {bg_path} ({bg_w}x{bg_h})")
            except Exception as e:
                print(f"Failed to load background image: {e}")
        else:
            print(f"Background image not found: {bg_path}")
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flow State - Dashboard")
        
        # Fonts
        self.font = pygame.font.SysFont("segoeui", 24)
        self.title_font = pygame.font.SysFont("segoeui", 48, bold=True)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # --- Background Music ---
        self._start_music()
        
        # Layout
        cx = self.width // 2
        cy = self.height // 2
        
        # Shift buttons down to create breathing room
        self.buttons = [
            MenuButton(cx - 150, cy - 20, 300, 50, "New Simulation", "sim"),
            MenuButton(cx - 150, cy + 50, 300, 50, "New Model Builder", "builder"),
            MenuButton(cx - 150, cy + 120, 300, 50, "Open File...", "sim"), # Currently maps to sim
            MenuButton(cx - 150, cy + 190, 300, 50, "Exit", "EXIT"),
        ]

    def _start_music(self):
        # Look for the music file in the 'music' subdirectory
        music_path = os.path.join(self.base_dir, "..", "music", "flow_dynamics.mp3")
        
        if os.path.exists(music_path):
            try:
                # Load and play continuously (-1 means loop forever)
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1, fade_ms=1000)
                print(f"Launcher playing: {music_path}")
            except Exception as e:
                print(f"Error loading music: {e}")
        else:
            print(f"Music file not found: {music_path}")

    def run(self):
        while self.running:
            if self.bg_image:
                # Center background if window is larger, or top-left if matching
                bg_x = (self.width - self.bg_image.get_width()) // 2
                bg_y = (self.height - self.bg_image.get_height()) // 2
                self.screen.blit(self.bg_image, (bg_x, bg_y))
            else:
                self.screen.fill((30, 30, 35))
            
            # Draw semi-transparent overlay behind text for readability if bg exists
            if self.bg_image:
                overlay_h = 500 
                overlay = pygame.Surface((400, overlay_h), pygame.SRCALPHA)
                overlay.fill((30, 30, 35, 200)) # Dark semi-transparent box
                # Center vertically over the content area
                self.screen.blit(overlay, (self.width//2 - 200, self.height//2 - 220))

            title = self.title_font.render("Flow State", True, (0, 122, 204))
            sub = self.font.render("Engineering Sandbox", True, (150, 150, 150))
            
            # Adjusted Y positions to center nicely over the button stack
            title_y = (self.height // 2) - 180
            sub_y = (self.height // 2) - 120
            
            # Draw Title
            self.screen.blit(title, (self.width//2 - title.get_width()//2, title_y))
            
            # Draw Subtitle with Drop Shadow
            sub_pos = (self.width//2 - sub.get_width()//2, sub_y)
            shadow = self.font.render("Engineering Sandbox", True, (0, 0, 0))
            self.screen.blit(shadow, (sub_pos[0] + 2, sub_pos[1] + 2))
            self.screen.blit(sub, sub_pos)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                for btn in self.buttons:
                    if btn.is_clicked(event):
                        self.handle_action(btn.action_code)

            for btn in self.buttons:
                btn.draw(self.screen, self.font)
                
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

    def handle_action(self, code):
        if code == "EXIT":
            self.running = False
        else:
            self.launch_app(code)

    def launch_app(self, mode):
        try:
            # Spawn the new independent instance using main.py
            # This respects the new entry point architecture
            script_path = os.path.join(self.base_dir, "..", "main.py")
            cmd = [sys.executable, script_path, "--mode", mode]
            
            print(f"Launching subprocess: {' '.join(cmd)}")
            subprocess.Popen(cmd)
        except Exception as e:
            print(f"Failed to launch: {e}")

def run():
    Dashboard().run()

if __name__ == "__main__":
    run()