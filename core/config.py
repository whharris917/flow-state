"""
Configuration Settings - DEPRECATED LOCATION

This module re-exports from shared.config for backward compatibility.
New code should import from shared.config directly.

Migration: CR-2026-001
"""

# Re-export everything from shared.config for backward compatibility
from shared.config import *

# Also re-export functions explicitly
from shared.config import scale, set_ui_scale
