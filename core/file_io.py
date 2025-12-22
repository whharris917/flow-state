"""
File I/O Module for Flow State

Handles saving and loading of:
- Full simulation state (.sim files)
- Model/geometry only (.mdl, .geom files)

All serialization uses JSON for portability and human-readability.
"""

import json
import os
import numpy as np
from model.geometry import Line, Circle, Point
from model.constraints import create_constraint


def save_file(sim, session, filepath):
    """
    Saves the complete simulation state (particles + geometry + view) to a JSON file.
    
    Args:
        sim: Simulation instance
        session: Session instance (for view state)
        filepath: Target file path (.sim recommended)
    
    Returns:
        Status message string
    """
    if not filepath:
        return "Cancelled"
    
    # Serialize geometry
    walls_dicts = [w.to_dict() for w in sim.walls]
    constraints_dicts = [c.to_dict() for c in sim.constraints]
    
    # Serialize materials from sketch
    materials_dicts = {}
    if hasattr(sim, 'sketch') and hasattr(sim.sketch, 'materials'):
        materials_dicts = {k: v.to_dict() for k, v in sim.sketch.materials.items()}
    
    # Capture view state from session
    view_state = {
        'zoom': session.zoom,
        'pan_x': session.pan_x,
        'pan_y': session.pan_y,
        'geo_time': getattr(session, 'geo_time', 0.0)
    }

    state = {
        'version': '1.1',
        'type': 'SIMULATION',
        
        # Physics arrays
        'count': sim.count,
        'pos_x': sim.pos_x[:sim.count].tolist(),
        'pos_y': sim.pos_y[:sim.count].tolist(),
        'vel_x': sim.vel_x[:sim.count].tolist(),
        'vel_y': sim.vel_y[:sim.count].tolist(),
        'is_static': sim.is_static[:sim.count].tolist(),
        'atom_sigma': sim.atom_sigma[:sim.count].tolist(),
        'atom_eps_sqrt': sim.atom_eps_sqrt[:sim.count].tolist(),
        
        # Geometry
        'walls': walls_dicts,
        'constraints': constraints_dicts,
        'materials': materials_dicts,
        
        # World config
        'world_size': sim.world_size,
        
        # View state
        'view_state': view_state
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        return f"Saved: {os.path.basename(filepath)}"
    except Exception as e:
        return f"Save Error: {e}"


def load_file(sim, filepath):
    """
    Loads a complete simulation state from a JSON file.
    
    Args:
        sim: Simulation instance to load into
        filepath: Source file path
    
    Returns:
        Tuple of (success: bool, message: str, view_state: dict or None)
    """
    if not filepath:
        return False, "Cancelled", None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 1. Restore Materials (before geometry, as geometry references materials)
        if 'materials' in data and hasattr(sim, 'sketch'):
            from model.properties import Material
            sim.sketch.materials = {}
            for k, v in data['materials'].items():
                sim.sketch.materials[k] = Material.from_dict(v)
            # Ensure defaults exist
            if "Default" not in sim.sketch.materials:
                sim.sketch.materials["Default"] = Material("Default")
            if "Ghost" not in sim.sketch.materials:
                sim.sketch.materials["Ghost"] = Material("Ghost", physical=False)
        
        # 2. Restore Geometry
        walls = []
        for w_data in data.get('walls', []):
            if w_data['type'] == 'line':
                walls.append(Line.from_dict(w_data))
            elif w_data['type'] == 'circle':
                walls.append(Circle.from_dict(w_data))
            elif w_data['type'] == 'point':
                walls.append(Point.from_dict(w_data))
        
        # 3. Restore Constraints
        constraints = []
        for c_data in data.get('constraints', []):
            c = create_constraint(c_data)
            if c:
                constraints.append(c)
        
        # 4. Build state dict for Simulation.restore_state()
        restored_state = {
            'count': data.get('count', 0),
            'world_size': data.get('world_size', 50.0),
            'walls': walls,
            'constraints': constraints,
        }
        
        # Convert arrays
        array_fields = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'atom_sigma', 'atom_eps_sqrt']
        for field in array_fields:
            if field in data:
                restored_state[field] = np.array(data[field], dtype=np.float32)
        
        if 'is_static' in data:
            restored_state['is_static'] = np.array(data['is_static'], dtype=np.int32)
        
        # Handle kinematic_props if present
        if 'kinematic_props' in data:
            restored_state['kinematic_props'] = np.array(data['kinematic_props'], dtype=np.float32)
        
        # 5. Apply to simulation
        sim.restore_state(restored_state)
        
        # 6. Extract view state
        view_state = data.get('view_state', None)
        
        return True, f"Loaded: {os.path.basename(filepath)}", view_state
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    except Exception as e:
        return False, f"Load Error: {e}", None


def save_geometry_file(sim, session, filepath):
    """
    Saves only geometry (entities, constraints, materials) without particle state.
    Useful for exporting reusable models/components.
    
    Args:
        sim: Simulation instance
        session: Session instance (for view state)
        filepath: Target file path (.geom or .mdl recommended)
    
    Returns:
        Status message string
    """
    if not filepath:
        return "Cancelled"
    
    # Check if there's any geometry to export
    if not sim.walls:
        return "No geometry to export"
    
    # Get geometry data from GeometryManager if available, otherwise build directly
    if hasattr(sim, 'geo') and hasattr(sim.geo, 'export_geometry_data'):
        geo_data = sim.geo.export_geometry_data()
    else:
        # Fallback: build directly from sim
        geo_data = {
            'walls': [w.to_dict() for w in sim.walls],
            'constraints': [c.to_dict() for c in sim.constraints],
        }
        
        # Include materials
        if hasattr(sim, 'sketch') and hasattr(sim.sketch, 'materials'):
            geo_data['materials'] = {k: v.to_dict() for k, v in sim.sketch.materials.items()}
    
    # Capture view state
    view_state = {
        'zoom': session.zoom,
        'pan_x': session.pan_x,
        'pan_y': session.pan_y,
        'geo_time': getattr(session, 'geo_time', 0.0)
    }
    
    wrapper = {
        'version': '1.1',
        'type': 'GEOMETRY',
        'data': geo_data,
        'view_state': view_state
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(wrapper, f, indent=2)
        return f"Geometry Saved: {os.path.basename(filepath)}"
    except Exception as e:
        return f"Export Error: {e}"


def load_geometry_file(filepath):
    """
    Loads geometry data for placement into an existing world.
    Does NOT directly modify simulation - returns data for caller to place.
    
    Args:
        filepath: Source file path
    
    Returns:
        Tuple of (data: dict or None, view_state: dict or None)
    """
    if not filepath:
        return None, None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both wrapped format and raw format
        if data.get('type') == 'GEOMETRY':
            return data.get('data', {}), data.get('view_state', None)
        elif 'walls' in data:
            # Raw geometry format (legacy or direct export)
            return data, None
        else:
            return None, None
            
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in geometry file: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading geometry: {e}")
        return None, None