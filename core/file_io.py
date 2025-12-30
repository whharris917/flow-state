"""
File I/O Module for Flow State

Handles saving and loading of:
- Full simulation state (.sim files)
- Model/geometry only (.mdl, .geom files)

All serialization uses JSON for portability and human-readability.

Architecture note: These functions receive Scene (the document container)
and access Sketch/Simulation through it.
"""

import json
import os
import numpy as np
from model.geometry import Line, Circle, Point
from model.constraints import create_constraint


def save_file(scene, session, filepath):
    """
    Saves the complete simulation state (particles + geometry + view) to a JSON file.
    
    Args:
        scene: Scene instance (owns Sketch and Simulation)
        session: Session instance (for view state)
        filepath: Target file path (.sim recommended)
    
    Returns:
        Status message string
    """
    if not filepath:
        return "Cancelled"
    
    sketch = scene.sketch
    simulation = scene.simulation
    
    # Serialize geometry
    walls_dicts = [e.to_dict() for e in sketch.entities]
    constraints_dicts = [c.to_dict() for c in sketch.constraints]
    materials_dicts = {k: v.to_dict() for k, v in sketch.materials.items()}
    
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
        'count': simulation.count,
        'pos_x': simulation.pos_x[:simulation.count].tolist(),
        'pos_y': simulation.pos_y[:simulation.count].tolist(),
        'vel_x': simulation.vel_x[:simulation.count].tolist(),
        'vel_y': simulation.vel_y[:simulation.count].tolist(),
        'is_static': simulation.is_static[:simulation.count].tolist(),
        'atom_sigma': simulation.atom_sigma[:simulation.count].tolist(),
        'atom_eps_sqrt': simulation.atom_eps_sqrt[:simulation.count].tolist(),
        
        # Geometry
        'walls': walls_dicts,
        'constraints': constraints_dicts,
        'materials': materials_dicts,
        
        # World config
        'world_size': simulation.world_size,
        
        # View state
        'view_state': view_state
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        return f"Saved: {os.path.basename(filepath)}"
    except Exception as e:
        return f"Save Error: {e}"


def load_file(scene, filepath):
    """
    Loads a complete simulation state from a JSON file.
    
    Args:
        scene: Scene instance to load into
        filepath: Source file path
    
    Returns:
        Tuple of (success: bool, message: str, view_state: dict or None)
    """
    if not filepath:
        return False, "Cancelled", None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        sketch = scene.sketch
        simulation = scene.simulation
        
        # 1. Restore Materials (before geometry, as geometry references materials)
        if 'materials' in data:
            from model.properties import Material
            sketch.materials = {}
            for k, v in data['materials'].items():
                sketch.materials[k] = Material.from_dict(v)
            # Ensure default exists
            if "Default" not in sketch.materials:
                sketch.materials["Default"] = Material("Default")
        
        # 2. Restore Geometry
        sketch.entities = []
        for w_data in data.get('walls', []):
            if w_data['type'] == 'line':
                sketch.entities.append(Line.from_dict(w_data))
            elif w_data['type'] == 'circle':
                sketch.entities.append(Circle.from_dict(w_data))
            elif w_data['type'] == 'point':
                sketch.entities.append(Point.from_dict(w_data))
        
        # 3. Restore Constraints
        sketch.constraints = []
        for c_data in data.get('constraints', []):
            c = create_constraint(c_data)
            if c:
                sketch.constraints.append(c)
        
        # 4. Restore Physics State
        physics_data = {
            'count': data.get('count', 0),
            'world_size': data.get('world_size', 50.0),
        }
        
        array_fields = ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'atom_sigma', 'atom_eps_sqrt']
        for field in array_fields:
            if field in data:
                physics_data[field] = np.array(data[field], dtype=np.float32)
        
        if 'is_static' in data:
            physics_data['is_static'] = np.array(data['is_static'], dtype=np.int32)

        simulation.restore(physics_data)
        
        # 5. Mark geometry dirty and rebuild
        scene.mark_dirty()
        
        # 6. Extract view state
        view_state = data.get('view_state', None)
        
        return True, f"Loaded: {os.path.basename(filepath)}", view_state
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    except Exception as e:
        return False, f"Load Error: {e}", None


def save_geometry_file(scene, session, filepath):
    """
    Saves only geometry (entities, constraints, materials) without particle state.
    Useful for exporting reusable models/components.
    
    Args:
        scene: Scene instance
        session: Session instance (for view state)
        filepath: Target file path (.geom or .mdl recommended)
    
    Returns:
        Status message string
    """
    if not filepath:
        return "Cancelled"
    
    sketch = scene.sketch
    
    # Check if there's any geometry to export
    if not sketch.entities:
        return "No geometry to export"
    
    # Get geometry data from GeometryManager if available
    if scene.geo and hasattr(scene.geo, 'export_geometry_data'):
        geo_data = scene.geo.export_geometry_data()
    else:
        # Fallback: build directly from sketch
        geo_data = {
            'walls': [e.to_dict() for e in sketch.entities],
            'constraints': [c.to_dict() for c in sketch.constraints],
            'materials': {k: v.to_dict() for k, v in sketch.materials.items()}
        }
    
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
    Does NOT directly modify scene - returns data for caller to place.
    
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