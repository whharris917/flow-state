import pygame
import array
import math
import os
import random

# Try to import procedural tracks if they exist in the ui_folly package
try:
    import ui_folly.track_whimsical as track_whimsical
    import ui_folly.track_jungle as track_jungle
except ImportError:
    track_whimsical = None
    track_jungle = None

class SoundManager:
    _instance = None
    
    def __init__(self):
        self.sounds = {}
        self.tracks = {}
        self.current_music = None
        
        # Only generate if mixer is initialized
        if pygame.mixer.get_init():
            self._generate_library()
            self._load_tracks()

    @staticmethod
    def get():
        if SoundManager._instance is None:
            SoundManager._instance = SoundManager()
        return SoundManager._instance

    def play_sound(self, name):
        """Plays a sound effect by name."""
        if name in self.sounds:
            # Slight pitch randomization could go here for extra "juice"
            try:
                self.sounds[name].set_volume(0.4)
                self.sounds[name].play()
            except:
                pass

    def play_music(self, track_name):
        """Switches the background music track."""
        if track_name == self.current_music: return
        
        try:
            pygame.mixer.music.stop()
            if track_name in self.tracks:
                data = self.tracks[track_name]
                # Handle procedural buffer vs file path
                if isinstance(data, str) and os.path.exists(data):
                    pygame.mixer.music.load(data)
                    pygame.mixer.music.play(-1, fade_ms=1000)
                elif hasattr(data, 'buffer_info') or isinstance(data, (bytes, bytearray, array.array)): 
                     # For generated tracks in memory, we play them on a reserved channel
                     # Channel 7 is arbitrarily chosen for music
                     music_sound = pygame.mixer.Sound(buffer=data)
                     music_sound.set_volume(0.25)
                     pygame.mixer.Channel(7).play(music_sound, loops=-1, fade_ms=1000)
                self.current_music = track_name
        except Exception as e:
            print(f"Audio Error: {e}")

    def stop_music(self):
        pygame.mixer.music.fadeout(500)
        pygame.mixer.Channel(7).fadeout(500)
        self.current_music = None

    def _make_tone(self, freq, duration, decay=True, vol_scale=1.0):
        """Synthesizes a simple tone."""
        rate = 44100
        n = int(rate * duration)
        buf = array.array('h', [0]*n)
        amplitude = 3000 * vol_scale
        
        for i in range(n):
            t = i/rate
            # Sine wave
            val = math.sin(2 * math.pi * freq * t)
            # Envelope
            env = 1.0
            if decay: 
                # Quadratic decay
                env = ((1.0 - t/duration)**2)
            
            buf[i] = int(val * amplitude * env)
            
        return pygame.mixer.Sound(buffer=buf)

    def _make_bloop(self, start_freq, end_freq, duration, vol_scale=1.0):
        """Synthesizes a frequency sweep (bloop)."""
        rate = 44100
        n = int(rate * duration)
        buf = array.array('h', [0]*n)
        amplitude = 2500 * vol_scale
        
        for i in range(n):
            t = i/rate
            # Linear frequency sweep
            curr_freq = start_freq + (end_freq - start_freq) * (t / duration)
            phase = 2 * math.pi * curr_freq * t # Approx
            val = math.sin(phase)
            
            # Sine envelope for smooth start/stop
            env = math.sin(math.pi * (t / duration))
            
            buf[i] = int(val * amplitude * env)
            
        return pygame.mixer.Sound(buffer=buf)

    def _generate_library(self):
        # UI Sounds
        self.sounds['hover'] = self._make_tone(450, 0.05, vol_scale=0.5)
        self.sounds['click'] = self._make_tone(1200, 0.08, vol_scale=0.8)
        self.sounds['snap'] = self._make_bloop(220, 350, 0.1, vol_scale=0.8)
        self.sounds['error'] = self._make_tone(150, 0.3, vol_scale=1.2)
        self.sounds['success'] = self._make_tone(880, 0.2, vol_scale=0.8)
        
        # Tools
        self.sounds['tool_select'] = self._make_tone(600, 0.05)
        self.sounds['tool_draw'] = self._make_tone(400, 0.05)

    def _load_tracks(self):
        # Load procedural tracks if imported
        if track_whimsical: 
            try: self.tracks['Whimsical'] = track_whimsical.generate(44100)
            except: pass
        if track_jungle:
            try: self.tracks['Jungle'] = track_jungle.generate(44100)
            except: pass