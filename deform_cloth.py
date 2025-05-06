import taichi as ti
import taichi.math as tm
import numpy as np

from PA1.scene import Scene, Init
import PA1.helper as hp

ti.init(arch=ti.vulkan)
# ------------------- Simulation parameters ------------------- #
# timestep for explicit integration
n = 16
stepsize = 2e-2 / n
ns = int((1/60) // stepsize)
dt = (1/60) / ns

# ------------------- Physics parameters ------------------- #
# particles are affected by gravity
gravity = ti.Vector([0, -9.8, 0])
m = 1.0 / (n*n)
# physical quantities
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
# Default values for reproducing the reference
YoungsModulus[None] = 1e2
PoissonsRatio[None] = 0.3

# 3D Lame parameters
Lambda = ti.field(ti.f32, ())
Mu = ti.field(ti.f32, ())
@ti.kernel
def compute_lame_parameters():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    Lambda[None] = (E * nu) / ((1 + nu) * (1 - 2*nu))
    Mu[None] = E / (2 * (1 + nu))
compute_lame_parameters()

# ------------------- cloth geometry ------------------- #
# position of the cloth
off_x, off_y, off_z = -0.1, 0.6, 0.1
# cloth is square with n x n particles
# Use smaller n values for debugging
quad_size = 1.0 / (n-1)

N = n * n

# vertices
vertices = ti.Vector.field(3, dtype=float, shape=N)

# triangles: each triangle is defined by 3 vertices
N_triangles = (n-1) * (n-1) * 2
triangles = ti.Vector.field(3, dtype=int, shape=(N_triangles, ))
triangles_np = np.zeros((N_triangles, 3), dtype=np.int32)
for i in range(n-1):
    for j in range(n-1):
        idx = i * (n-1) + j
        triangles_np[idx, 0] = i * n + j
        triangles_np[idx, 1] = i * n + j + 1
        triangles_np[idx, 2] = (i + 1) * n + j

        triangles_np[idx + N_triangles // 2, 0] = (i + 1) * n + j
        triangles_np[idx + N_triangles // 2, 1] = i * n + j + 1
        triangles_np[idx + N_triangles // 2, 2] = (i + 1) * n + j + 1
triangles.from_numpy(triangles_np)

# pinning, indexes of vertices
pins_np = np.array([0, n-1], dtype=np.int32)
pins = ti.field(dtype=int, shape=2)
pins.from_numpy(pins_np)
# ------------------- system state ------------------- #
x_rest = ti.Vector.field(3, dtype=ti.f32, shape=N)
x_rest_np = np.zeros((N, 3), dtype=np.float32)
for i in range(n):
    for j in range(n):
        idx = i * n + j
        x_rest_np[idx, 0] = i * quad_size + off_x
        x_rest_np[idx, 1] = off_y
        x_rest_np[idx, 2] = j * quad_size + off_z
x_rest.from_numpy(x_rest_np)
x = ti.Vector.field(3, dtype=ti.f32, shape=N)
v = ti.Vector.field(3, dtype=ti.f32, shape=N)
force = ti.Vector.field(3, dtype=ti.f32, shape=N)

# ------------------- deformation functions ------------------- #
D0 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_triangles)
Tk = ti.field(dtype=ti.f32, shape=N_triangles)
D = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_triangles)
F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_triangles)
P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_triangles)
H = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_triangles)

# Compute D0 and Tk (3D space)
@ti.func
def compute_D0_Tk():
    for i in range(N_triangles):
        a, b, c = triangles[i]
        Xa, Xb, Xc = x_rest[a], x_rest[b], x_rest[c]
        N0 = tm.cross(Xb - Xa, Xc - Xa)
        D0[i] = tm.mat3(Xb - Xa, Xc - Xa, N0).transpose()
        Tk[i] = N0.norm() * 0.5
# Compute D (3D space)
@ti.func
def compute_D():
    for i in range(N_triangles):
        a, b, c = triangles[i]
        Xa, Xb, Xc = x[a], x[b], x[c]
        N0 = tm.cross(Xb - Xa, Xc - Xa)
        D[i] = tm.mat3(Xb - Xa, Xc - Xa, N0).transpose()
# Compute F (3D space)
@ti.func
def compute_F():
    for i in range(N_triangles):
        F[i] = D[i] @ (D0[i].inverse())
# Compute P (3D space)
@ti.func
def compute_P_c():
    for i in range(N_triangles):
        Fi = F[i]
        U, sigma, V = ti.svd(Fi)
        R = U @ V.transpose()
        S = V @ sigma @ V.transpose()
        strain_c = S - ti.Matrix.identity(ti.f32, 3)
        P[i] = R @ (2 * Mu[None] * strain_c + Lambda[None] * strain_c.trace() * ti.Matrix.identity(ti.f32, 3))
@ti.func
def compute_P_v():
    for i in range(N_triangles):
        Fi = F[i]
        green_strain = 0.5 * (Fi.transpose() @ Fi - ti.Matrix.identity(ti.f32, 3))
        P[i] = Fi @ (2 * Mu[None] * green_strain + Lambda[None] * green_strain.trace() * ti.Matrix.identity(ti.f32, 3))
@ti.func
def compute_P_n():
    for i in range(N_triangles):
        Fi = F[i]
        J = ti.max(Fi.determinant(), 1e-6)
        P[i] = Mu[None] * (Fi - Fi.inverse().transpose()) + Lambda[None] * ti.log(J) * Fi.inverse().transpose()
# Compute H (3D space)
@ti.func
def compute_H():
    for i in range(N_triangles):
        H[i] = - Tk[i] * P[i] @ D0[i].inverse().transpose()

# ----------------- timestep functions ------------------- #

@ti.kernel
def init_simulation():
    for i in range(N):
        x[i] = x_rest[i]
        v[i] = [0.0, 0.0, 0.0]
        force[i] = [0.0, 0.0, 0.0]
    compute_D0_Tk()

@ti.kernel
def timestep():
    # Compute D, F, P, H
    compute_D()
    compute_F()
    # compute_P_c()
    # compute_P_v()
    compute_P_n()
    compute_H()

    # Reset forces
    for i in range(N):
        force[i] = [0.0, -9.81 * m, 0.0]
    
    # Internal elastic forces
    for i in range(N_triangles):
        Hi = H[i]
        a, b, c = triangles[i]
        fb = ti.Vector([Hi[0, 0], Hi[1, 0], Hi[2, 0]])
        fc = ti.Vector([Hi[0, 1], Hi[1, 1], Hi[2, 1]])
        fa = -fb - fc
        force[a] += fa
        force[b] += fb
        force[c] += fc

    # Update velocity
    for i in range(N):
        v[i] += dt * force[i] / m
    
    for pin in pins:
        v[pin] = ti.Vector([0.0, 0.0, 0.0])
    
    # Update position
    for i in range(N):
        x[i] += dt * v[i]

# ------------------- Main function and GUI setup ------------------- #

# Data structures for drawing the mesh
indices = ti.field(int, shape=N_triangles * 3)
colors = ti.Vector.field(3, dtype=float, shape=N)

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
        # if (i % 20 < 4 and i % 20 >= 0) or (j % 20 < 4 and j % 20 >= 0):
        if (i % 4 == 0) or (j % 4 == 0):
            colors[i * n + j] = (1.0, 0.97, 0.95)
        else:
            colors[i * n + j] = (1.0, 0.2, 0.4)

# Copy vertex state into mesh vertex positions
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        idx = i * n + j
        vertices[idx] = x[idx]

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

init_simulation()
initialize_mesh_indices()

# Run sim
while True:

    for k in range(ns):
        timestep()
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
    # scene.mesh(obstacle.verts,
    #            indices=obstacle.tris,
    #            color=(0.8, 0.7, 0.6))
    canvas.scene(scene)
    
    # if record:
    #     img = window.get_image_buffer_as_numpy()
    #     video_manager.write_frame(img)
    window.show()