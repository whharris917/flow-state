"""
StatusBar - Transient Status Message Display

Owns:
- Current message text
- Display timestamp
- Visibility logic

API:
- set(message): Set a new status message
- message: Get current message text
- visible_message: Get message only if still visible
- is_visible: Check if message should be displayed
- fade_alpha: Get alpha for fade effect (0.0-1.0)
"""

import time


class StatusBar:
    """
    Manages transient status messages shown to the user.
    
    Messages automatically fade after a configurable duration.
    """
    
    def __init__(self, display_duration: float = 3.0):
        """
        Initialize the status bar.
        
        Args:
            display_duration: How long messages stay visible (seconds)
        """
        self._message: str = ""
        self._timestamp: float = 0.0
        self._duration: float = display_duration
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def message(self) -> str:
        """Get the current message (even if expired)."""
        return self._message
    
    @property
    def timestamp(self) -> float:
        """Get the timestamp when message was set."""
        return self._timestamp
    
    @property
    def visible_message(self) -> str:
        """Get the message only if still visible, else empty string."""
        if self.is_visible:
            return self._message
        return ""
    
    @property
    def is_visible(self) -> bool:
        """Check if the message should still be displayed."""
        return time.time() - self._timestamp < self._duration
    
    @property
    def time_remaining(self) -> float:
        """Get seconds remaining before message fades."""
        remaining = self._duration - (time.time() - self._timestamp)
        return max(0.0, remaining)
    
    @property
    def fade_alpha(self) -> float:
        """
        Get alpha value for fade effect (1.0 = fully visible, 0.0 = invisible).
        
        Starts fading in the last 0.5 seconds.
        """
        remaining = self.time_remaining
        if remaining > 0.5:
            return 1.0
        elif remaining > 0:
            return remaining / 0.5
        return 0.0
    
    # =========================================================================
    # Operations
    # =========================================================================
    
    def set(self, message: str):
        """
        Set a new status message.
        
        Args:
            message: The message to display
        """
        self._message = message
        self._timestamp = time.time()
    
    def clear(self):
        """Clear the current message immediately."""
        self._message = ""
        self._timestamp = 0.0
    
    def extend(self, additional_seconds: float = None):
        """
        Extend the display time of the current message.
        
        Args:
            additional_seconds: Extra time to add (defaults to full duration)
        """
        if additional_seconds is None:
            additional_seconds = self._duration
        # Move timestamp forward to extend visibility
        self._timestamp = time.time() - (self._duration - additional_seconds)
