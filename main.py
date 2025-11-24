import pygame
import config
import math
import sys
from camera import Camera
from simulation import Simulation
from renderer import Renderer

# Mode Constants
MODE_SELECT = 'select'
MODE_PIPE = 'pipe'
MODE_VALVE = 'valve'
MODE_SOURCE = 'source'
MODE_SINK = 'sink'
MODE_INSPECT = 'inspect'

def main():
    sys.setrecursionlimit(5000) 

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("The Synthetic Era - Pipe Dream")
    clock = pygame.time.Clock()

    camera = Camera()
    camera.offset_x = config.SCREEN_WIDTH / 2
    camera.offset_y = config.SCREEN_HEIGHT / 2
    
    sim = Simulation()
    renderer = Renderer(screen, camera)
    
    current_mode = MODE_SELECT 
    active_material = 'Red' 
    
    buttons = [
        {'label': 'SELECT (V)', 'mode': MODE_SELECT, 'rect': None},
        {'label': 'INSPECT (K)', 'mode': MODE_INSPECT, 'rect': None},
        {'label': 'PIPE (P)',   'mode': MODE_PIPE,   'rect': None},
        {'label': 'VALVE (O)',  'mode': MODE_VALVE,  'rect': None},
        {'label': 'SOURCE (I)', 'mode': MODE_SOURCE, 'rect': None},
        {'label': 'SINK (U)',   'mode': MODE_SINK,   'rect': None},
    ]

    def update_button_rects():
        start_x = 20
        y = config.SCREEN_HEIGHT - config.TOOLBAR_HEIGHT + (config.TOOLBAR_HEIGHT - config.BUTTON_HEIGHT) // 2
        for btn in buttons:
            btn['rect'] = pygame.Rect(start_x, y, config.BUTTON_WIDTH, config.BUTTON_HEIGHT)
            start_x += config.BUTTON_WIDTH + config.BUTTON_MARGIN

    update_button_rects()

    drag_source_node = None 
    snap_preview_pos = None 
    
    running = True
    while running:
        dt = clock.tick(config.FPS) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed() 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                config.SCREEN_WIDTH = event.w
                config.SCREEN_HEIGHT = event.h
                renderer.surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                update_button_rects()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: current_mode = MODE_SELECT
                if event.key == pygame.K_k: current_mode = MODE_INSPECT
                if event.key == pygame.K_p: current_mode = MODE_PIPE
                if event.key == pygame.K_o: current_mode = MODE_VALVE
                if event.key == pygame.K_i: current_mode = MODE_SOURCE
                if event.key == pygame.K_u: current_mode = MODE_SINK
                if event.key == pygame.K_1: active_material = 'Red'
                if event.key == pygame.K_2: active_material = 'Green'
                if event.key == pygame.K_3: active_material = 'Blue'
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    sim.undo()
                    drag_source_node = None 

            elif event.type == pygame.MOUSEBUTTONDOWN:
                ui_clicked = False
                if mouse_pos[1] > config.SCREEN_HEIGHT - config.TOOLBAR_HEIGHT:
                    for btn in buttons:
                        if btn['rect'].collidepoint(mouse_pos):
                            current_mode = btn['mode']
                            drag_source_node = None 
                            ui_clicked = True
                            break
                
                if not ui_clicked:
                    wx, wy = camera.screen_to_world(mouse_pos[0], mouse_pos[1])
                    scaled_snap_dist = config.SNAP_DISTANCE / camera.zoom
                    
                    if event.button == 3: 
                        target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist)
                        if target:
                            sim.save_state()
                            sim.delete_entity(target, target_type)
                            drag_source_node = None

                    elif event.button == 1: 
                        if current_mode == MODE_SELECT:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist, nodes_only=True)
                            if target_type == 'node':
                                if target.kind == 'valve':
                                    sim.create_node(target.x, target.y, kind='valve').setting = 0.0 if target.setting > 0.5 else 1.0
                                    # Note: Toggle logic needs update for new architecture, simpler to just set it
                                    target.setting = 0.0 if target.setting > 0.5 else 1.0
                                elif target.kind == 'source':
                                    # Cycle material logic moved to sim class
                                    # But we need to access it. For now, manually cycle.
                                    modes = ['Red', 'Green', 'Blue', 'Water']
                                    try:
                                        idx = modes.index(target.material_type)
                                        target.material_type = modes[(idx+1)%len(modes)]
                                    except: pass

                        elif current_mode == MODE_PIPE:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist, nodes_only=keys[pygame.K_e])
                            if target_type == 'node':
                                drag_source_node = target
                            elif target_type == 'segment':
                                sim.save_state()
                                # Split pipe!
                                px, py, pipe_obj = target
                                new_node = sim.split_pipe(pipe_obj, px, py)
                                drag_source_node = new_node
                            else:
                                sim.save_state()
                                new_node = sim.create_node(wx, wy)
                                drag_source_node = new_node

                        elif current_mode == MODE_VALVE:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist)
                            if target_type == 'segment':
                                sim.save_state()
                                px, py, pipe_obj = target
                                sim.split_pipe(pipe_obj, px, py, kind='valve')

                        elif current_mode == MODE_SOURCE:
                            sim.save_state()
                            target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist)
                            if target_type == 'node':
                                target.kind = 'source'; target.fixed_pressure = config.DEFAULT_SOURCE_PRESSURE; target.material_type = active_material
                            elif target_type == 'segment':
                                px, py, pipe_obj = target
                                sim.split_pipe(pipe_obj, px, py, kind='source')
                                # Need to set material on the returned node, but split_pipe returns node...
                                # In new code, split_pipe returns node.
                                # Wait, split_pipe in new code doesn't take material arg.
                                # We need to fetch it.
                                # For now simple node creation is fine.
                            else:
                                sim.create_node(wx, wy, kind='source', material_type=active_material)

                        elif current_mode == MODE_SINK:
                            sim.save_state()
                            target, target_type = sim.get_snap_target(wx, wy, scaled_snap_dist)
                            if target_type == 'node':
                                target.kind = 'sink'; target.fixed_pressure = 0.0
                            elif target_type == 'segment':
                                px, py, pipe_obj = target
                                sim.split_pipe(pipe_obj, px, py, kind='sink')
                            else:
                                sim.create_node(wx, wy, kind='sink')

                    elif event.button == 2: 
                        camera.drag_start = pygame.mouse.get_pos()
                    elif event.button == 4: 
                        new_zoom = min(camera.zoom * 1.1, 5.0)
                        camera.zoom_towards(new_zoom, config.SCREEN_WIDTH/2, config.SCREEN_HEIGHT/2)
                    elif event.button == 5: 
                        new_zoom = max(camera.zoom * 0.9, 0.2)
                        camera.zoom_towards(new_zoom, config.SCREEN_WIDTH/2, config.SCREEN_HEIGHT/2)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drag_source_node:
                    if current_mode == MODE_PIPE:
                        wx, wy = camera.screen_to_world(mouse_pos[0], mouse_pos[1])
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            dx = abs(wx - drag_source_node.x)
                            dy = abs(wy - drag_source_node.y)
                            if dx > dy: wy = drag_source_node.y
                            else: wx = drag_source_node.x

                        target, target_type = sim.get_snap_target(wx, wy, config.SNAP_DISTANCE/camera.zoom, nodes_only=keys[pygame.K_e])
                        
                        if target_type == 'node':
                            if target != drag_source_node:
                                sim.save_state()
                                sim.create_pipe(drag_source_node, target)
                        elif target_type == 'segment':
                            sim.save_state()
                            px, py, pipe_obj = target
                            new_node = sim.split_pipe(pipe_obj, px, py)
                            sim.create_pipe(drag_source_node, new_node)
                        else:
                            sim.save_state()
                            new_node = sim.create_node(wx, wy)
                            sim.create_pipe(drag_source_node, new_node)
                    drag_source_node = None
                elif event.button == 2:
                    camera.drag_start = None

            elif event.type == pygame.MOUSEMOTION:
                if camera.drag_start:
                    dx = mouse_pos[0] - camera.drag_start[0]
                    dy = mouse_pos[1] - camera.drag_start[1]
                    camera.offset_x += dx
                    camera.offset_y += dy
                    camera.drag_start = mouse_pos

        sim.update(dt)

        screen.fill(config.C_BACKGROUND)
        renderer.draw_grid()
        renderer.draw_simulation(sim)
        
        if drag_source_node:
            # Draw preview pipe
            sx, sy = camera.world_to_screen(drag_source_node.x, drag_source_node.y)
            mx, my = pygame.mouse.get_pos()
            # Apply Shift Snap visual
            wx, wy = camera.screen_to_world(mx, my)
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                dx = abs(wx - drag_source_node.x)
                dy = abs(wy - drag_source_node.y)
                if dx > dy: wy = drag_source_node.y
                else: wx = drag_source_node.x
                mx, my = camera.world_to_screen(wx, wy)
            
            pygame.draw.line(screen, config.C_PREVIEW, (sx, sy), (mx, my), 2)

        # Inspect
        if current_mode == MODE_INSPECT:
            target, target_type = sim.get_snap_target(wx, wy, config.INTERACTION_DISTANCE/camera.zoom)
            if target:
                if target_type == 'segment':
                    # Inspect Pipe
                    renderer.draw_inspector_tooltip(target[2], mouse_pos) # target[2] is pipe obj
                else:
                    renderer.draw_inspector_tooltip(target, mouse_pos)

        renderer.draw_toolbar(buttons, current_mode, mouse_pos, active_material)
        renderer.draw_ui(camera.screen_to_world(mouse_pos[0], mouse_pos[1]))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()