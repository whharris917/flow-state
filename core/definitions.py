"""
Constraint Definitions - DEPRECATED LOCATION

This module re-exports from model.constraint_factory for backward compatibility.
New code should import from model.constraint_factory directly.

Migration: CR-2026-001
"""

# Re-export for backward compatibility
from model.constraint_factory import CONSTRAINT_DEFS, get_l, get_r, get_angle
