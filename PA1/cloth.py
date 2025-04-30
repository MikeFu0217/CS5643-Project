import taichi as ti
import taichi.math as tm
import numpy as np

from scene import Scene, Init
import helper as hp

# Have to use Vulkan arch on Mac for compatibility with GGUI
ti.init(arch=ti.gpu)

# Collision Obstacle
obstacle = Scene(Init.CLOTH_TABLE)
contact_eps = 1e-2
record = False

# cloth is square with n x n particles
# Use smaller n values for debugging
n = 128 
quad_size = 1.0 / (n-1)

# particles are affected by gravity
gravity = ti.Vector([0, -9.8, 0])
particle_mass = 1.0 / (n*n)

# timestep for explicit integration
stepsize = 2e-2 / n
ns = int((1/60) // stepsize)
dt = (1/60) / ns

# spring constants
default_k_spring = 3e0*n
k_damp = default_k_spring * 1e-5 # spring damping
k_drag = 1e0 * particle_mass # viscous damping

# some particles can be pinned in place
# pins are named by indices in the particle grid
# pins = [[0,0], [0,n-1]]
pins = []

# springs
ijij_structural_numpy = hp.ge_square_edges_ijij(n, "structural")
ijij_shear_numpy = hp.ge_square_edges_ijij(n, "shear")
ijij_flextion_numpy = hp.ge_square_edges_ijij(n, "flextion")

ijij_structural = ti.Vector.field(4, dtype=int, shape=ijij_structural_numpy.shape[0])
ijij_shear = ti.Vector.field(4, dtype=int, shape=ijij_shear_numpy.shape[0])
ijij_flextion = ti.Vector.field(4, dtype=int, shape=ijij_flextion_numpy.shape[0])

ijij_structural.from_numpy(ijij_structural_numpy)
ijij_shear.from_numpy(ijij_shear_numpy)
ijij_flextion.from_numpy(ijij_flextion_numpy)

# A spherical collision object, not used until the very last part
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0.5, 0, 0.5]
ball_radius = 0.3

### System state

x = ti.Vector.field(3, dtype=float, shape=(n, n))
# TODO: Create field for particle velocities
v = ti.Vector.field(3, dtype=float, shape=(n, n))
forces = ti.Vector.field(3, dtype=float, shape=(n, n))

# Set up initial state on the y = 0.6 plane
off_x, off_z = -0.1, 0.1
@ti.kernel
def init_cloth():
    # TODO
    for i, j in ti.ndrange(n, n):
        x[i, j] = [i * quad_size + off_x, 0.6, j * quad_size + off_z]
        v[i, j] = [0, 0, 0]

# Execute a single symplectic Euler timestep
@ti.func
def apply_spring_force(i, j, i2, j2, l0, ks):
    spring_offset = x[i2, j2] - x[i, j]
    l = tm.length(spring_offset)
    if l > 1e-6:  # Avoid division by zero
        spring_dir = tm.normalize(spring_offset)
        f = ks * (l - l0) * spring_dir  # spring force
        stiffness_damp = - k_damp * spring_dir * tm.dot(v[i, j] - v[i2, j2], spring_dir)  # stiffness damp force
        f += stiffness_damp
        forces[i, j] += f
        forces[i2, j2] -= f

@ti.kernel
def timestep(k0: float, k1: float, k2: float):
    # Apply gravity
    for i, j in ti.ndrange(n, n):
        forces[i, j] = gravity * particle_mass
    
    # Apply spring forces using the refactored function
    for k in ijij_structural:  # Structural springs
        i1, j1, i2, j2 = ijij_structural[k]
        apply_spring_force(i1, j1, i2, j2, quad_size, k0)
    
    for k in ijij_shear:  # Shear springs
        i1, j1, i2, j2 = ijij_shear[k]
        apply_spring_force(i1, j1, i2, j2, quad_size * tm.sqrt(2), k1)
    
    for k in ijij_flextion:  # Flexion springs
        i1, j1, i2, j2 = ijij_flextion[k]
        apply_spring_force(i1, j1, i2, j2, quad_size * 2, k2)

    # Update velocity
    for i, j in ti.ndrange(n, n):
        # Apply mass damping
        forces[i, j] += - k_drag * v[i, j]
        # Update velocity and position
        v[i, j] += dt * forces[i, j] / particle_mass

        # ---------------------- Collision with Sphere ---------------------- #
        # if tm.length(x[i, j] - obstacle.ball_center[0]) < ball_radius + contact_eps:
        #     normal = tm.normalize(x[i, j] - obstacle.ball_center[0])
        #     v[i, j] -=  tm.min(0, tm.dot(v[i, j], normal)) * normal
        # ---------------------- Collision with Sphere ---------------------- #

        # ---------------------- Collision with Table  ---------------------- #
        # For convenience, only detect with table top
        p_x, p_y, p_z = x[i, j][0], x[i, j][1], x[i, j][2]
        top_x, top_y, top_z = obstacle.tabletop_center[0][0], obstacle.tabletop_center[0][1], obstacle.tabletop_center[0][2]
        top_h, top_r = obstacle.tabletop_height, obstacle.tabletop_radius
        d_h = tm.sqrt( (p_x-top_x)**2 + (p_z-top_z)**2 )
        
        pnt_top = (top_y + top_h/2) - p_y
        # For the bottom surface: how far above the bottom is the particle?
        pnt_bottom = p_y - (top_y - top_h/2)
        # For the side surface: how far inside the table's radius is the particle?
        pnt_side = top_r - d_h
    
        # If the particle is in inside the table top
        if p_y < top_y + top_h/2 + contact_eps and p_y > top_y - top_h/2 - contact_eps and d_h < top_r + contact_eps:                
            if pnt_side < pnt_top and pnt_side < pnt_bottom: # Collision with side
                normal = tm.normalize(ti.Vector([p_x-top_x, 0.0, p_z-top_z]))
                v[i, j] -=  tm.min(0, tm.dot(v[i, j], normal)) * normal
            elif pnt_top < pnt_side and pnt_top < pnt_bottom: # Collision with top
                normal = ti.Vector([0, 1, 0])
                v[i, j] -=  tm.min(0, tm.dot(v[i, j], normal)) * normal
            else: # Collision with bottom
                normal = ti.Vector([0, -1, 0])
                v[i, j] -=  tm.min(0, tm.dot(v[i, j], normal)) * normal
        # ---------------------- Collision with Table  ---------------------- #

        x[i, j] += dt * v[i, j]

        
    # Apply pinning
    for i, j in ti.static(pins):
        v[i, j] = [0, 0, 0]
        x[i, j] = [i * quad_size, 0.6, j * quad_size]

### GUI

# Data structures for drawing the mesh
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# Set up the mesh
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i % 20 < 4 and i % 20 >= 0) or (j % 20 < 4 and j % 20 >= 0):
            colors[i * n + j] = (1.0, 0.97, 0.95)
        else:
            colors[i * n + j] = (1.0, 0.2, 0.4)


# Copy vertex state into mesh vertex positions
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

result_dir = './recordings/'
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

# Create Taichi UI
scene = ti.ui.Scene()
camera = ti.ui.Camera()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 1.0))
cam_pos = np.array([0.0, 0.8, 5.0])
camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
camera.lookat(0.5, 0.2, 0.5)
camera.fov(30.0)

# Initialize sim
start_t = 0.0
current_t = 0.0

init_cloth()
initialize_mesh_indices()

# Run sim
while True:

    for k in range(ns):
        timestep(default_k_spring, default_k_spring, default_k_spring)
        current_t += dt
        update_vertices()

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    
    # Uncomment this part for collision
    scene.mesh(obstacle.verts,
               indices=obstacle.tris,
               color=(0.8, 0.7, 0.6))
    canvas.scene(scene)
    
    if record:
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)
    window.show()
    
if record:
    video_manager.make_video(gif=False, mp4=True)