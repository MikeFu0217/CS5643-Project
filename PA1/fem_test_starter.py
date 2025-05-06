import taichi as ti
import taichi.math as tm
import numpy as np
from pywavefront import Wavefront
from util import *

ti.init(arch=ti.vulkan)

# Models
# 1: Co-rotated linear model
# 2: St. Venant Kirchhoff model ( StVK )
# 3: Neo-Hookean model
model_names = ['c', 'v', 'n']
model = 'c'
prev_model = model
damping_toggle = ti.field(ti.i32, ())
damping_toggle[None] = 1
ModelSelector = ti.field(ti.i32, ())
ModelSelector[None] = 0

# physical quantities
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
# Default values for reproducing the reference
YoungsModulus[None] = 1e2
PoissonsRatio[None] = 0.5

##############################################################
# TODO: Put additional parameters here
# e.g. Lame parameters
Lambda = ti.field(ti.f32, ())
Mu = ti.field(ti.f32, ())
@ti.kernel
def compute_lame_parameters():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    Lambda[None] = (E * nu) / ((1 + nu) * (1 - nu))
    Mu[None] = E / (2 * (1 + nu))

compute_lame_parameters()
##############################################################

## Load geometry of the test scenes
rect_obj = Wavefront("models/rect.obj", collect_faces=True)
rect_obj_stretch = Wavefront("models/rect_stretch.obj", collect_faces=True)
rect_obj_compress = Wavefront("models/rect_compress.obj", collect_faces=True)

va = np.array(rect_obj.vertices, dtype=np.float32)[:,:2]
va_stretch = np.array(rect_obj_stretch.vertices, dtype=np.float32)[:,:2]
va_compress = np.array(rect_obj_compress.vertices, dtype=np.float32)[:,:2]

# The objs have the exact same topology, e.g. the meshes are the same
mesh = rect_obj.mesh_list[0]
faces = np.array(mesh.faces, dtype=np.int32)
mesh_triangles = ti.field(int, shape=np.prod(faces.shape))
mesh_triangles.from_numpy(faces.ravel())

# Number of triangles
N_triangles = faces.shape[0]

triangles = ti.Vector.field(3, ti.i32, N_triangles)
for i in range(N_triangles):
    triangles[i] = ti.Vector(faces[i])

# We also need to draw the edges
edges_set = set()
for i in range(N_triangles):
    edges_set.add((faces[i][0],faces[i][1]) if faces[i][0] < faces[i][1] else (faces[i][1], faces[i][0]))
    edges_set.add((faces[i][1],faces[i][2]) if faces[i][1] < faces[i][2] else (faces[i][2], faces[i][1]))
    edges_set.add((faces[i][2],faces[i][0]) if faces[i][2] < faces[i][0] else (faces[i][0], faces[i][2]))

# Number of edges
N_edges = len(edges_set)
np_edges = np.array([list(e) for e in edges_set])
edges = ti.Vector.field(2, shape=N_edges, dtype=int)
edges.from_numpy(np_edges)

#############################################################
## Deformable object Simulation

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time model)
dh = h/substepping

# Number of vertices
N = va.shape[0]
# Mass 
m = 1/N*25
# damping parameter
k_drag = 10 * m

# Simulation components: x and v
x_rest = ti.Vector.field(2, shape=N, dtype=ti.f32)
# x_rest is initialized directly to the mesh vertices (in 2D)
# and remains unchanged.
x_rest.from_numpy(va)

x_stretch = ti.Vector.field(2, shape=N, dtype=ti.f32)
x_stretch.from_numpy(va_stretch)

x_compress = ti.Vector.field(2, shape=N, dtype=ti.f32)
x_compress.from_numpy(va_compress)

# Deformed shape
x = ti.Vector.field(2, shape=N, dtype=ti.f32)
v = ti.Vector.field(2, ti.f32, N)

force = ti.Vector.field(2, ti.f32, N)

# Pinned indices
# We pin the top and botton edges of the rectangle (known a priori)
pins = ti.field(ti.i32, N)
num_pins = ti.field(ti.i32, ())
num_pins[None] = 40
for i in range(int(0.5*num_pins[None])):
    pins[2*i] = 40*i
    pins[2*i+1] = 40*i+39

# TODO: Put additional fields here for storing D (etc.)
D0 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)  # Rest shape matrix
Tk = ti.field(dtype=ti.f32, shape=N_triangles)
D = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)  # Deformation gradient
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)  # Deformation gradient
P = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)  # First Piola-Kirchhoff stress tensor
H = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)  # Elastic force matrix


# Compute D0 and Tk
@ti.func
def compute_D0_Tk():
    for i in range(N_triangles):
        a, b, c = triangles[i]
        Xa, Xb, Xc = x_rest[a], x_rest[b], x_rest[c]
        D0[i] = tm.mat2(Xb - Xa, Xc - Xa).transpose()
        det_D0 = D0[i].determinant()
        if ti.abs(det_D0) < 1e-6:  # Prevent singularity
            D0[i] += 1e-6 * ti.Matrix.identity(ti.f32, 2)
        Tk[i] = abs(det_D0) * 0.5
# Compute D
@ti.func
def compute_D():
    for i in range(N_triangles):
        a, b, c = triangles[i]
        Xa, Xb, Xc = x[a], x[b], x[c]
        D[i] = tm.mat2(Xb - Xa, Xc - Xa).transpose()
# Compute F
@ti.func
def compute_F():
    for i in range(N_triangles):
        F[i] = D[i] @ (D0[i].inverse())
# Compute P
@ti.func
def compute_P_c():
    for i in range(N_triangles):
        Fi = F[i]
        U, sigma, V = ti.svd(Fi)
        R = U @ V.transpose()
        S = V @ sigma @ V.transpose()
        strain_c = S - ti.Matrix.identity(ti.f32, 2)
        P[i] = R @ (2 * Mu[None] * strain_c + Lambda[None] * strain_c.trace() * ti.Matrix.identity(ti.f32, 2))
@ti.func
def compute_P_v():
    for i in range(N_triangles):
        Fi = F[i]
        green_strain = 0.5 * (Fi.transpose() @ Fi - ti.Matrix.identity(ti.f32, 2))
        P[i] = Fi @ (2 * Mu[None] * green_strain + Lambda[None] * green_strain.trace() * ti.Matrix.identity(ti.f32, 2))
@ti.func
def compute_P_n():
    for i in range(N_triangles):
        Fi = F[i]
        J = ti.max(Fi.determinant(), 1e-6)
        P[i] = Mu[None] * (Fi - Fi.inverse().transpose()) + Lambda[None] * ti.log(J) * Fi.inverse().transpose()
# Compute H
@ti.func
def compute_H():
    for i in range(N_triangles):
        H[i] = - Tk[i] * P[i] @ D0[i].inverse().transpose()

# TODO: Implement the initialization and timestepping kernel for the deformable object
@ti.kernel
def init():
    compute_D0_Tk()
    for i in v:
        v[i] = ti.Vector([0,0])
@ti.kernel
def timestep():
    # TODO: Sympletic integration of the internal elastic forces 

    ## Compute D, F, P, H
    compute_D()
    compute_F()
    if (ModelSelector[None] == 0):
        compute_P_c()
    elif (ModelSelector[None] == 1):
        compute_P_v()
    else:
        compute_P_n()
    compute_H()

    # Reset force
    for i in range(N):
        force[i] = ti.Vector([0,0])

    # Internal elastic forces
    for i in range(N_triangles):
        Hi = H[i]
        a, b, c = triangles[i]
        fb = ti.Vector([Hi[0,0], Hi[1,0]])
        fc = ti.Vector([Hi[0,1], Hi[1,1]])
        fa = -fb - fc
        force[a] += fa
        force[b] += fb
        force[c] += fc
    
    # Update velocity
    for i in range(N):
        v[i] += dh*force[i]/m

    for i in ti.ndrange(num_pins[None]):
        v[pins[i]] = ti.Vector([0,0])

    # viscous damping
    for i in v:
        if damping_toggle[None]:
            v[i] -= v[i] * k_drag / m * dh 
    
    # Update position
    for i in range(N):
        x[i] += dh*v[i]
    
    # TODO: Sympletic integration of the internal elastic forces

##############################################################

is_stretch = ti.field(ti.i32, ())
is_stretch[None] = 1

def reset_state():
    ModelSelector[None] = 'cvn'.find(model)
    initialize()

@ti.kernel
def initialize():
    if is_stretch[None] == 1:
        for i in range(N):
            x[i] = x_stretch[i]
    else:
        for i in range(N):
            x[i] = x_compress[i]

    for i in range(N):
        v[i] = ti.Vector([0,0])

# initialize system state
initialize()

####################################################################
# TODO: Run your initialization code 
init()
####################################################################

paused = False
window = ti.ui.Window("Linear FEM", (600, 600))
canvas = window.get_canvas()
canvas.set_background_color((1,1,1))

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key in ['v','c','n']:
            prev_model = model
            model = e.key
            if prev_model != model:
                reset_state()
        elif e.key in ['d','D']:
            damping_toggle[None] = not damping_toggle[None]
        elif e.key == ti.GUI.UP:
            is_stretch[None] = 1
            initialize()
        elif e.key == ti.GUI.DOWN:
            is_stretch[None] = 0
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused = not paused

    ##############################################################
    if not paused:
        # TODO: run all of your simulation code here
        for i in range(substepping):
            timestep()
    ##############################################################

    # Draw wireframe of mesh
    canvas.lines(vertices=x, indices=edges, width=0.002, color=(0,0,0))

    gui = window.get_gui()
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.35):
        idx_old = model_names.index(model)
        idx_new = gui.slider_int("Model ID", idx_old, 0, 2)
        if idx_new != idx_old:
            prev_model = model
            model = model_names[idx_new]
            if prev_model != model:
                reset_state()

        E_old = YoungsModulus[None]
        E_new = gui.slider_float("Young's Modulus (E)", E_old, 1e1, 1e3)
        if E_new != E_old:
            YoungsModulus[None] = E_new
            compute_lame_parameters()

        nu_old = PoissonsRatio[None]
        nu_new = gui.slider_float("Poisson Ratio (ν)", nu_old, 0.0, 0.49)
        if nu_new != nu_old:
            PoissonsRatio[None] = nu_new
            compute_lame_parameters()

        gui.text(f"Mode: {'Co-rotated' if model=='c' else 'StVK' if model=='v' else 'Neo-Hookean'}")
        gui.text(f"E = {YoungsModulus[None]:.1f}, ν = {PoissonsRatio[None]:.3f}")
        gui.text("Press c/v/n or use slider above")
        gui.text("Up/Down: stretch/compress")
        gui.text("SPACE: pause/unpause")
        gui.text("d: toggle damping")
        gui.text(f"Damping: {'On' if damping_toggle[None] else 'Off'}")
        if paused:
            gui.text("Simulation PAUSED")

    window.show()
