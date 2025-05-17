import taichi as ti
import taichi.math as tm
import numpy as np

from config import Config
from physics import Physics
from cloth import Cloth
from render import Render
from obstacles import Obstacles

ti.init(arch=ti.gpu)

cfg = Config(n=40, gravity=[0, -9.81, 0])
cloth = Cloth(cfg.n, pos=[-0.1, 0.6, 0.1], pins=cfg.pin_options[cfg.pin])
obstacles = Obstacles()
phy = Physics(cfg, cloth, obstacles, E=450, nu=0.15, k_drag=1.2)

def init_cpu():
    cfg.update_ModelSelector()
    cfg.update_CollisionSelector()
    cfg.update_self_collision()
    cfg.update_friction()
    cfg.update_bending()
    cloth.init_shift()

@ti.kernel
def init_gpu():
    cloth.init_state()
    phy.compute_D0_Tk()
    phy.reset_cloth_force()

@ti.kernel
def timestep():
    # Compute D, F, P, H
    phy.compute_D()
    phy.compute_F()
    if (cfg.ModelSelector[None] == 0):
        phy.compute_P_v()
    elif (cfg.ModelSelector[None] == 1):
        phy.compute_P_c()
    else:
        phy.compute_P_n()
    phy.compute_H()

    # Update forces
    phy.reset_cloth_force()
    phy.compute_cloth_internal_force()
    phy.compute_bending_forces()

    # Self-collision
    phy.apply_self_collision()

    # Update velocity and position
    phy.forward_euler()

# Create Taichi GUI
camera = ti.ui.Camera()
window = ti.ui.Window("Cloth Deformation", (1680, 1024), vsync=True)
scene = window.get_scene()

gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 1.0))
cam_pos = np.array([0.0, 2.0, 5.0])
camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
camera.lookat(0.5, 0.0, 0.5)
camera.fov(30.0)

# Initialize sim
renderer = Render(cloth)
init_gpu()
renderer.initialize_mesh_indices()

# Run sim
while window.running:

    # keyboard controls
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.SPACE:
            # Reset simulation
            init_cpu()
            init_gpu()
        elif e.key in ['v','c','n']:
            cfg.prev_model = cfg.model
            cfg.model = e.key
            init_cpu()
            init_gpu()

    # Update timestep and vertices
    for k in range(cfg.ns):
        timestep()
        if cfg.show_force_color:
            renderer.set_force_colors()
        renderer.update_vertices()

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(renderer.vertices,
               indices=renderer.indices,
               per_vertex_color=renderer.colors,
               two_sided=True)
    if cfg.obstacle is not None:
        scene.mesh(obstacles.get_obstacle(cfg.obstacle_names.index(cfg.obstacle)).verts,
                   indices=obstacles.get_obstacle(cfg.obstacle_names.index(cfg.obstacle)).tris,
                   color=(0.8, 0.7, 0.6))

    canvas.scene(scene)
    
    # gui
    gui = window.get_gui()
    with gui.sub_window("Controls", 0.7, 0.02, 0.25, 0.6):

        # Text
        gui.text("Press SPACE to reset simulation")
        gui.text("Press 'v', 'c', 'n' to select models")
        

        # Update cfg.model with a slider
        idx_old = cfg.model_names.index(cfg.model)
        idx_new = gui.slider_int("Model ID", idx_old, 0, 2)
        if idx_new != idx_old:
            cfg.prev_model = cfg.model
            cfg.model = cfg.model_names[idx_new]
            if cfg.prev_model != cfg.model:
                init_cpu()
                init_gpu()

        # Select obstacle
        if cfg.obstacle is None:
            idx_old = -1
        else:
            idx_old = cfg.obstacle_names.index(cfg.obstacle)
        idx_new = gui.slider_int("Obstacle ID", idx_old, -1, 1)
        if idx_new != idx_old:
            if idx_new == -1:
                cfg.obstacle = None
            else:
                cfg.obstacle = cfg.obstacle_names[idx_new]
            init_cpu()
            init_gpu()

        gui.text("Pinning and shifting")
        # Select pinning
        idx_old = cfg.pin
        idx_new = gui.slider_int("Pin ID", idx_old, 0, 5)
        if idx_new != idx_old:
            cfg.pin = idx_new
            cloth.set_pins(cfg.pin_options[idx_new])
            init_cpu()
            init_gpu()
        # Slide to shift along the y-axis for all pinned vertices
        if cfg.pin != 0:
            y_shift = gui.slider_float("Shift Y Pins", cloth.shift_y, 0, -1.5)
            if y_shift != cloth.shift_y:
                cloth.shift_y = y_shift
                for i in range(cloth.pin_cnt[None]):
                    idx = cloth.pins[i]
                    cloth.x[idx][1] = cloth.off_y + cloth.shift_y
        # Slide to shift along away from each other all pinned vertices
        if cfg.pin == 5:
            y_shift = gui.slider_float("Shift Away Pins", cloth.shift_away, -0.5, 0.5)
            if y_shift != cloth.shift_away:
                cloth.shift_away = y_shift
                for i in range(cloth.pin_cnt[None]):
                    idx = cloth.pins[i]
                    if i == 0:  # first pint, shift along -x, z
                        cloth.x[idx][0] = cloth.off_x - cloth.shift_away
                        cloth.x[idx][2] = cloth.off_z + cloth.shift_away
                    if i == 1:
                        cloth.x[idx][0] = cloth.off_x - cloth.shift_away
                        cloth.x[idx][2] = cloth.off_z + 1 - cloth.shift_away
                    if i == 2:  # third pint, shift along +x, z
                        cloth.x[idx][0] = cloth.off_x + 1 + cloth.shift_away
                        cloth.x[idx][2] = cloth.off_z + cloth.shift_away
                    if i == 3:  # fourth pint, shift along -x, z
                        cloth.x[idx][0] = cloth.off_x + 1 + cloth.shift_away
                        cloth.x[idx][2] = cloth.off_z + 1 - cloth.shift_away


        gui.text("Deformation Parameters")
        # Update k_drag with a slider
        new_k_drag = gui.slider_float("Viscous Damping", phy.k_drag[None], 1.0, 10.0)
        if new_k_drag != phy.k_drag[None]:
            phy.k_drag[None] = new_k_drag
        new_youngs_modulus = gui.slider_float('Youngs Modulus', phy.YoungsModulus[None], 420, 900)
        new_possion_ratio = gui.slider_float('Poissons Ratio', phy.PoissonsRatio[None], 0.0, 0.2)
        # Update Young's Modulus with a slider
        if new_youngs_modulus != phy.YoungsModulus[None] or new_possion_ratio != phy.PoissonsRatio[None]:
            phy.YoungsModulus[None] = new_youngs_modulus
            phy.PoissonsRatio[None] = new_possion_ratio
            phy.compute_lame_parameters()  # Recompute Lame parameters after Young's modulus change

        # Update self-collision
        new_self_collision = gui.checkbox("Self Collision", cfg.self_collision)
        if new_self_collision != cfg.self_collision:
            cfg.self_collision = new_self_collision
            init_cpu()
            init_gpu()
        # if new_self_collision == 1:
        #     new_self_collision_strength = gui.slider_float("Self Collision Strength", phy.self_collision_strength[None], 0, 500)
        #     if new_self_collision_strength != phy.self_collision_strength[None]:
        #         phy.self_collision_strength[None] = new_self_collision_strength

        # Update friction
        new_friction = gui.checkbox("Friction", cfg.friction)
        if new_friction != cfg.friction:
            cfg.friction = new_friction
            init_cpu()
            init_gpu()
        if new_friction == 1:
            new_mu_friction = gui.slider_float("Friction Coefficient", phy.mu_friction[None], 0.0, 1.0)
            if new_mu_friction != phy.mu_friction[None]:
                phy.mu_friction[None] = new_mu_friction

        # Update bending
        new_bending = gui.checkbox("Bending Energy", cfg.bending)
        if new_bending != cfg.bending:
            cfg.bending = new_bending
            init_cpu()
            init_gpu()
        if new_bending == 1:
            new_bend_stiffness = gui.slider_float("Bending Stiffness", phy.bend_stiffness[None], 0.0, 10.0)
            if new_bend_stiffness != phy.bend_stiffness[None]:
                phy.bend_stiffness[None] = new_bend_stiffness
            new_bend_damping = gui.slider_float("Bending Damping", phy.bend_damping[None], 0.0, 1.0)
            if new_bend_damping != phy.bend_damping[None]:
                phy.bend_damping[None] = new_bend_damping
            new_angle_tol = gui.slider_float("Bending Angle Tolerance", phy.angle_tol[None], 0.0, np.pi/2)
            if new_angle_tol != phy.angle_tol[None]:
                phy.angle_tol[None] = new_angle_tol

        # Update force color display
        new_show_force_color = gui.checkbox("Force Color Display", cfg.show_force_color)
        if new_show_force_color != cfg.show_force_color:
            if new_show_force_color == 0:
                renderer.set_color_to_default()
            cfg.show_force_color = new_show_force_color
        if cfg.show_force_color == 1:
            new_max_force = gui.slider_float("Force Color Max", cloth.max_force[None], 0.0, 5.0)
            if new_max_force != cloth.max_force[None]:
                cloth.max_force[None] = new_max_force
                renderer.set_force_colors()

    window.show()