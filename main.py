import taichi as ti
import taichi.math as tm
import numpy as np

from config import Config
from physics import Physics
from cloth import Cloth
from render import Render
from scene import Scene, Init

ti.init(arch=ti.gpu)

cfg = Config(n=32, gravity=[0, -9.81, 0])
cloth = Cloth(cfg.n, pos=[-0.1, 0.6, 0.1], pins=[0, cfg.n-1])
obstacle = Scene(Init.CLOTH_SPHERE)
phy = Physics(cfg, cloth, obstacle, E=450, nu=0.15, k_drag=1.2)

def init_cpu():
    cfg.update_ModelSelector()
    cfg.update_CollisionSelector()

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
        phy.compute_P_c()
    elif (cfg.ModelSelector[None] == 1):
        phy.compute_P_v()
    else:
        phy.compute_P_n()
    phy.compute_H()

    # Update forces
    phy.reset_cloth_force()
    phy.compute_cloth_internal_force()
    phy.apply_self_collision()

    # Update velocity and position
    phy.forward_euler()

# Create Taichi GUI
camera = ti.ui.Camera()
window = ti.ui.Window("Cloth Deformation", (1024, 1024), vsync=True)
scene = window.get_scene()

gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 1.0))
cam_pos = np.array([0.0, 0.8, 5.0])
camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
camera.lookat(0.5, 0.2, 0.5)
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
            init_gpu()
        elif e.key in ['v','c','n']:
            cfg.prev_model = cfg.model
            cfg.model = e.key
            init_cpu()
            init_gpu()

    # Update timestep and vertices
    for k in range(cfg.ns):
        timestep()
        renderer.update_vertices()

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(renderer.vertices,
               indices=renderer.indices,
               per_vertex_color=renderer.colors,
               two_sided=True)
    if cfg.obstacle is not None:
        scene.mesh(obstacle.verts,
                   indices=obstacle.tris,
                   color=(0.8, 0.7, 0.6))

    canvas.scene(scene)
    
    # gui
    gui = window.get_gui()
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.25):

        # Text
        gui.text("Press 'v', 'c', 'n' to select models")
        gui.text("Press SPACE to reset simulation")

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
        idx_new = gui.slider_int("Obstacle ID", idx_old, -1, 0)
        if idx_new != idx_old:
            if idx_new == -1:
                cfg.obstacle = None
            else:
                cfg.obstacle = cfg.obstacle_names[idx_new]
            init_cpu()
            init_gpu()
        
        # Update k_drag with a slider
        new_k_drag = gui.slider_float("Viscous Damping", phy.k_drag[None], 1.0, 10.0)
        if new_k_drag != phy.k_drag[None]:
            phy.k_drag[None] = new_k_drag
        new_youngs_modulus = gui.slider_float('Youngs Modulus', phy.YoungsModulus[None], 420, 1e3)
        new_possion_ratio = gui.slider_float('Poissons Ratio', phy.PoissonsRatio[None], 0.0, 0.2)
        # Update Young's Modulus with a slider
        if new_youngs_modulus != phy.YoungsModulus[None] or new_possion_ratio != phy.PoissonsRatio[None]:
            phy.YoungsModulus[None] = new_youngs_modulus
            phy.PoissonsRatio[None] = new_possion_ratio
            phy.compute_lame_parameters()  # Recompute Lame parameters after Young's modulus change

    window.show()