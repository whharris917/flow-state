import math
import numpy as np
import core.config as config

from model.geometry import Line, Circle, Point

# --- Coordinate Transformations ---

def sim_to_screen(x, y, zoom, pan_x, pan_y, world_size, layout):
    cx_world = world_size / 2.0
    cy_world = world_size / 2.0
    cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
    cy_screen = config.TOP_MENU_H + (layout['MID_H'] / 2.0)
    
    base_scale = (layout['MID_W'] - 50) / world_size
    final_scale = base_scale * zoom
    
    sx = cx_screen + (x - cx_world) * final_scale + pan_x
    sy = cy_screen + (y - cy_world) * final_scale + pan_y
    return int(sx), int(sy)

def screen_to_sim(sx, sy, zoom, pan_x, pan_y, world_size, layout):
    cx_world = world_size / 2.0
    cy_world = world_size / 2.0
    cx_screen = layout['MID_X'] + (layout['MID_W'] / 2.0)
    cy_screen = config.TOP_MENU_H + (layout['MID_H'] / 2.0)
    
    base_scale = (layout['MID_W'] - 50) / world_size
    final_scale = base_scale * zoom
    
    x = (sx - pan_x - cx_screen) / final_scale + cx_world
    y = (sy - pan_y - cy_screen) / final_scale + cy_world
    return x, y

# --- Grouping & Snapping Helpers ---

def get_connected_group(constraints, start_wall_idx):
    group = {start_wall_idx}
    queue = [start_wall_idx]
    
    adjacency = {}
    for c in constraints:
        if c.type == 'COINCIDENT':
            idx_list = c.indices
            w1, w2 = idx_list[0][0], idx_list[1][0]
            if w1 not in adjacency: adjacency[w1] = []
            if w2 not in adjacency: adjacency[w2] = []
            adjacency[w1].append(w2)
            adjacency[w2].append(w1)

    while queue:
        current = queue.pop(0)
        if current in adjacency:
            for neighbor in adjacency[current]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
    return group

def is_group_anchored(walls, group_indices):
    for idx in group_indices:
        if idx >= len(walls): continue
        w = walls[idx]
        if isinstance(w, Line):
            if w.anchored[0] or w.anchored[1]: return True
        elif isinstance(w, Circle):
            if w.anchored[0]: return True
    return False

def get_grouped_points(walls, zoom, pan_x, pan_y, world_size, layout):
    point_map = {}

    for i, w in enumerate(walls):
        points_to_process = []
        if isinstance(w, Line):
            # Skip endpoint rendering for infinite ref lines
            if w.ref and w.infinite:
                continue
            if np.array_equal(w.start, w.end):
                points_to_process.append((w.start, 0))
            else:
                points_to_process.append((w.start, 0))
                points_to_process.append((w.end, 1))
        elif isinstance(w, Circle):
            points_to_process.append((w.center, 0))
        elif isinstance(w, Point):
            points_to_process.append((w.pos, 0))
        
        for pt, end_idx in points_to_process:
            sx, sy = sim_to_screen(pt[0], pt[1], zoom, pan_x, pan_y, world_size, layout)
            found_key = None
            for k in point_map:
                if abs(k[0]-sx) <= 3 and abs(k[1]-sy) <= 3: 
                    found_key = k
                    break
            
            if found_key: 
                point_map[found_key].append((i, end_idx))
            else: 
                point_map[(sx, sy)] = [(i, end_idx)]
    return point_map

def get_snapped_pos(mx, my, walls, zoom, pan_x, pan_y, world_size, layout,
                    anchor_pos=None, exclude_wall_idx=-1,
                    snap_to_points=False, constrain_to_axis=False):
    """
    Get world position from screen coordinates with optional snapping.

    Args:
        mx, my: Screen coordinates
        walls: List of geometry entities to snap to
        zoom, pan_x, pan_y: Camera transform parameters
        world_size: Simulation world size
        layout: Screen layout dictionary
        anchor_pos: Reference point for axis constraint (optional)
        exclude_wall_idx: Entity index to exclude from snapping
        snap_to_points: If True, snap to nearby vertices (UI passes Ctrl state)
        constrain_to_axis: If True, constrain to H/V axis from anchor (UI passes Shift state)

    Returns:
        (final_x, final_y, snapped_target) where snapped_target is (entity_idx, point_idx) or None
    """
    sim_x, sim_y = screen_to_sim(mx, my, zoom, pan_x, pan_y, world_size, layout)
    final_x, final_y = sim_x, sim_y
    is_snapped_to_vertex = False
    snapped_target = None

    if snap_to_points:
        snap_threshold_px = 15
        best_dist = float('inf')
        for i, w in enumerate(walls):
            if i == exclude_wall_idx: continue
            points = []
            if isinstance(w, Line): points = [(w.start, 0), (w.end, 1)]
            elif isinstance(w, Circle): points = [(w.center, 0)]
            elif isinstance(w, Point): points = [(w.pos, 0)]

            for pt_pos, pt_idx in points:
                sx, sy = sim_to_screen(pt_pos[0], pt_pos[1], zoom, pan_x, pan_y, world_size, layout)
                dist = math.hypot(mx - sx, my - sy)
                if dist < snap_threshold_px and dist < best_dist:
                    best_dist = dist
                    final_x, final_y = pt_pos
                    snapped_target = (i, pt_idx)
                    is_snapped_to_vertex = True

    if constrain_to_axis and anchor_pos is not None and not is_snapped_to_vertex:
        dx = final_x - anchor_pos[0]
        dy = final_y - anchor_pos[1]
        if abs(dx) > abs(dy):
            final_y = anchor_pos[1]
        else:
            final_x = anchor_pos[0]

    return final_x, final_y, snapped_target

def calculate_current_temp(vel_x, vel_y, count, mass):
    if count == 0: return 0.0
    vx = vel_x[:count]
    vy = vel_y[:count]
    ke_total = 0.5 * mass * np.sum(vx**2 + vy**2)
    return ke_total / count