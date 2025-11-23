import pygame
import config
import math
from camera import Camera
from simulation import Simulation
from renderer import Renderer

# Mode Constants
MODE_SELECT = 'select'
MODE_PIPE = 'pipe'
MODE_VALVE = 'valve'
MODE_SOURCE = 'source'
MODE_SINK = 'sink'

def main():
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
    active_material = 'Red' # Default material for placement
    
    buttons = [
        {'label': 'SELECT (V)', 'mode': MODE_SELECT, 'rect': None},
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
                if event.key == pygame.K_p: current_mode = MODE_PIPE
                if event.key == pygame.K_o: current_mode = MODE_VALVE
                if event.key == pygame.K_i: current_mode = MODE_SOURCE
                if event.key == pygame.K_u: current_mode = MODE_SINK
                
                # Material Selection
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
                    scaled_interact_dist = config.INTERACTION_DISTANCE / camera.zoom

                    if event.button == 3: 
                        target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
                        if target:
                            sim.save_state()
                            sim.delete_entity(target, target_type)
                            drag_source_node = None

                    elif event.button == 1: 
                        if current_mode == MODE_SELECT:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist, nodes_only=True)
                            if target_type == 'node' and target.kind == 'valve':
                                sim.toggle_valve(target)

                        elif current_mode == MODE_VALVE:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
                            if target_type == 'segment':
                                sim.save_state()
                                split_x, split_y, node_a, node_b = target
                                sim.split_pipe(node_a, node_b, split_x, split_y, kind='valve')

                        elif current_mode == MODE_SOURCE:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
                            sim.save_state()
                            if target_type == 'node':
                                sim.convert_node(target, 'source', material_type=active_material)
                            elif target_type == 'segment':
                                split_x, split_y, node_a, node_b = target
                                sim.split_pipe(node_a, node_b, split_x, split_y, kind='source', material_type=active_material)
                            else:
                                sim.create_node(wx, wy, kind='source', material_type=active_material)

                        elif current_mode == MODE_SINK:
                            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
                            sim.save_state()
                            if target_type == 'node':
                                sim.convert_node(target, 'sink')
                            elif target_type == 'segment':
                                split_x, split_y, node_a, node_b = target
                                sim.split_pipe(node_a, node_b, split_x, split_y, kind='sink')
                            else:
                                sim.create_node(wx, wy, kind='sink')

                        elif current_mode == MODE_PIPE:
                            thresh = scaled_snap_dist
                            if pygame.key.get_mods() & pygame.KMOD_CTRL: thresh = 2000.0
                            nodes_only = keys[pygame.K_e]

                            target, target_type = sim.get_snap_target(wx, wy, thresh, nodes_only=nodes_only)
                            
                            if target_type == 'node':
                                drag_source_node = target
                            elif target_type == 'segment':
                                sim.save_state()
                                split_x, split_y, node_a, node_b = target
                                new_node = sim.split_pipe(node_a, node_b, split_x, split_y)
                                drag_source_node = new_node
                            else:
                                sim.save_state()
                                new_node = sim.create_node(wx, wy)
                                drag_source_node = new_node
                                

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

                        scaled_snap_dist = config.SNAP_DISTANCE / camera.zoom
                        thresh = scaled_snap_dist
                        if pygame.key.get_mods() & pygame.KMOD_CTRL: thresh = 2000.0
                        nodes_only = keys[pygame.K_e]

                        target, target_type = sim.get_snap_target(wx, wy, thresh, nodes_only=nodes_only)
                        dist_dragged = math.hypot(wx - drag_source_node.x, wy - drag_source_node.y)
                        
                        if target_type == 'node':
                            if target != drag_source_node:
                                sim.save_state()
                                sim.add_pipe(drag_source_node, target.x, target.y, connect_to_end_node=target)
                        elif target_type == 'segment':
                            sim.save_state()
                            split_x, split_y, node_a, node_b = target
                            new_junction = sim.split_pipe(node_a, node_b, split_x, split_y)
                            sim.add_pipe(drag_source_node, new_junction.x, new_junction.y, connect_to_end_node=new_junction)
                        else:
                            if dist_dragged > config.MIN_PIPE_LENGTH:
                                sim.save_state()
                                sim.add_pipe(drag_source_node, wx, wy)
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

        # --- Frame Logic ---
        snap_preview_pos = None
        hover_snap_pos = None 
        valve_preview = None
        
        wx, wy = camera.screen_to_world(mouse_pos[0], mouse_pos[1])
        
        scaled_snap_dist = config.SNAP_DISTANCE / camera.zoom
        scaled_interact_dist = config.INTERACTION_DISTANCE / camera.zoom
        
        thresh = scaled_snap_dist
        if pygame.key.get_mods() & pygame.KMOD_CTRL: thresh = 2000.0
        nodes_only = keys[pygame.K_e]

        if current_mode == MODE_PIPE:
            if drag_source_node:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    dx = abs(wx - drag_source_node.x)
                    dy = abs(wy - drag_source_node.y)
                    if dx > dy: wy = drag_source_node.y
                    else: wx = drag_source_node.x
                
                target, target_type = sim.get_snap_target(wx, wy, thresh, nodes_only=nodes_only)
                
                if target_type == 'node': snap_preview_pos = (target.x, target.y)
                elif target_type == 'segment': snap_preview_pos = (target[0], target[1])
                else: snap_preview_pos = (wx, wy)
            else:
                target, target_type = sim.get_snap_target(wx, wy, thresh, nodes_only=nodes_only)
                if target:
                    if target_type == 'node': hover_snap_pos = (target.x, target.y)
                    elif target_type == 'segment': hover_snap_pos = (target[0], target[1])

        elif current_mode == MODE_VALVE:
            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
            if target and target_type == 'segment':
                cp_x, cp_y, na, nb = target
                dx = nb.x - na.x
                dy = nb.y - na.y
                angle = math.atan2(dy, dx)
                valve_preview = (cp_x, cp_y, angle)
        
        elif current_mode in [MODE_SOURCE, MODE_SINK]:
             target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist)
             if target:
                 if target_type == 'segment': hover_snap_pos = (target[0], target[1])
                 else: hover_snap_pos = (target.x, target.y)


        sim.update(dt)

        screen.fill(config.C_BACKGROUND)
        renderer.draw_grid()
        renderer.draw_simulation(sim)
        
        if drag_source_node and snap_preview_pos:
            sx, sy = camera.world_to_screen(snap_preview_pos[0], snap_preview_pos[1])
            renderer.draw_preview_line(drag_source_node, (sx, sy))
            renderer.draw_snap_indicator(snap_preview_pos)
            
        if hover_snap_pos:
            renderer.draw_snap_indicator(hover_snap_pos)
            
        if valve_preview:
            renderer.draw_valve_preview((valve_preview[0], valve_preview[1]), valve_preview[2])
        
        if current_mode == MODE_SELECT:
            target, target_type = sim.get_snap_target(wx, wy, scaled_interact_dist, nodes_only=True)
            if target_type == 'node':
                renderer.draw_telemetry(target, mouse_pos)
        
        renderer.draw_toolbar(buttons, current_mode, mouse_pos, active_material)
        renderer.draw_ui(camera.screen_to_world(mouse_pos[0], mouse_pos[1]))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()