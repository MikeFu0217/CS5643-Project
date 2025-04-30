---
title: Assignment 1
---

This assignment explores animations of deformable obejcts, and it has two parts. The first part is a 3D mass-and-spring cloth simulation, which simulates dynamic folding and wrinkling, but doesn't handle self-collisions in the cloth. The second part is a simulator for a 2D deformable gingerbread man, with three types of constitutive models and impulse-based collision.

# Preamble: Setting up Taichi
Install Taichi with `pip install taichi` and make sure you are able to run the [Julia Fractal](https://docs.taichi-lang.org/docs/hello_world).  You also need the `pywavefront` package for reading the collision object.

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) to install Python 3.8, and then using `pip` to install Taichi and `pywavefront` within a conda environment.  This will help keep your setup for this class separate from other Python environments you might need for other projects.  Once you have installed Anaconda you can do this with these commands:

```
conda create --name cs5643 python=3.8
conda activate cs5643
pip install taichi pywavefront
```

Be sure your Taichi version is at least 1.7.3.

For the starter code, go to our assignment repository at [this place](https://github.coecis.cornell.edu/cs5643/assignments-sp25) and make a fork. Your fork should be a private repository. This will let you easily pull updates related to bug fixes and later assignments from our repository into yours. 

Then clone your repository with:

`git clone git@github.coecis.cornell.edu:yourNetID/<path to your repo>.git`

# First part: Mass-spring cloth simulation

For the first part, you will implement a simple cloth simulator where a square piece of cloth collides with an object. The starter code `cloth.py` provides mesh constrution and rendering for the scene.

## Cloth initialization
A piece of square mass-spring cloth is composed of $N\times N$ (evenly-spaced) particles connected by varying types of springs. We use $[i,j]\ (0\leq i,j < n)$ to denote the particle located at $i^{th}$ row and $j^{th}$ column. Each particle has the following state information:

- Mass $m$. Here we set $m$ to be constant and assume that all particles have the same mass.

- Position $\mathbf{x}_{i,j}(t)$

- Velocity $\mathbf{v}_{i,j}(t)=\dot{\mathbf{x}}_{i,j}(t)$

The position and velocity are affected by different forces which we will implement in subsequent steps, and follow the Newton's second law:
$$ \mathbf{a}_{i,j}(t)=\dot{\mathbf{v}}_{i,j}(t) = \ddot{\mathbf{x}}_{i,j}(t)=\frac{\mathbf{F}_{i,j}(t)}{m}$$

You will use Symplectic Euler to update the positions of the particles.

Write functions for:

1. Initializing the cloth by setting the particle positions and velocities. You will need to set up vector fields for storing the position and velocity, and refer the particles by their indices in these vector fields. The quad should have width and height of 1. The demo has the cloth quad initialized on the $y=0.6$ plane, but you can adjust its position as desired.

2. Attaching the particles to the quad mesh, i.e. copy the particle positions to the mesh vertex positions.

The result should look like [this image](refs/phase1.png) where $N=128$.

## Springs
There are three types of springs that connect the particles. For an arbitrary particle at $[i,j]$, there are (at most) 12 connections established as follows:

- Structural: (At most) four neighboring particles at $[i,j+1],[i,j-1],[i+1,j],[i-1,j]$

- Shear: (At most) four neighboring particles along the diagonal at $[i+1,j+1],[i+1,j-1],[i-1,j+1],[i-1,j-1]$

- Flexion: (At most) four particles that are two particle away at $[i,j+2],[i,j-2],[i+2,j],[i-2,j]$

Each type of string has a different stiffness, denoted by $k_0$ (structural), $k_1$ (shear) and $k_2$ (flexion). You can make [GUI sliders](https://docs.taichi-lang.org/docs/ggui#gui-components) for each stiffness, but you can also just input different values at compilation time. In either way, play with the parameters to see how they affect the cloth 

Assuming that the cloth is initially at rest, the rest lengths of the springs are calculated by the distance between the particles connected by a spring in the initial state.
 
## Forces
### Spring forces
For two particles $p_1,p_2$ connected by a spring of stiffness $k_{\mathrm{spring}}$ with rest length $l_0$, the spring force acting on $p_1$ is
$$F_{\mathrm{spring}}=k_{\mathrm{spring}}(l_0-\|p_1-p_2\|)\frac{p_1-p_2}{\|p_1-p_2\|}$$

### Gravity
The gravity acting on each particle is simply $\mathbf{F}_{\mathrm{gravity}}=[0\ -mg\ 0]$ where $g=9.81$.

Additionally, you will write the time stepping function following Newton's second law. Pin the positions of the particles at indices $[0,0]$ and $[0,n-1]$, by simply setting their velocities to zero in the integration.

After implementing spring forces and gravity as well as setting up the time stepping function, your result should look like [this video](refs/phase2.mp4)

### Damping
Now we will add some damping so that the simulation converges. There are two types of damping:

1. Stiffness proportional damping: For two particles $\mathbf{p}_1,\mathbf{p}_2$ connected by a spring of stiffness $k_{\mathrm{spring}}$ with rest length $l_0$, the stiffness *damping* force corresponding to the spring force acting on $p_1$ is
$$\mathbf{F}_{\mathrm{stiffness\_damp}}=-k_{\mathrm{stiffness\_damp}}k_{\mathrm{spring}}\frac{\mathbf{p}_2-\mathbf{p}_1}{\|\mathbf{p}_2-\mathbf{p}_1\|}(\mathbf{v}_{i,j}\cdot \frac{\mathbf{p}_2-\mathbf{p}_1}{\|\mathbf{p}_2-\mathbf{p}_1\|})$$

2. Mass proportional damping:
$$\mathbf{F}_{\mathrm{mass\_damp}}=-k_{\mathrm{mass\_damp}}m \mathbf{v}_{i,j}$$

Changing the default parameters damping parameters $k_{\mathrm{stiffness\_damp}}$ and $k_{\mathrm{mass\_damp}}$ and observe the effects of changing the parameters. With default parameters, your result should look like [this reference video](refs/phase3.mp4).

## Collision

### Test 1: Sphere
Now we will bring in a sphere and have it collide with the cloth. During the integration, for each particle, check if the distance from the center of the sphere is smaller than the sphere radius plus a small `contact_eps`. If so, project the velocity back onto the surface of the sphere, i.e. $$\mathbf{v}\leftarrow \mathbf{v}-\min(0,\mathbf{v}\cdot\mathbf{n})*\mathbf{n}$$, where $n$ is the normal vector from the center of sphere to the particle position.

In the code, you may access the configuration of the sphere by querying attributes of `obstacle`, from which you can access `obstacle.ball_radius` and `obstacle.ball_center`.  A reasonable collision between the pinned cloth and the sphere should look something like [this](refs/phase4.mp4).

### Test 2: Table
Now, initialze your collision object with the configuration `Init.CLOTH_TABLE` instead and try to implement collision detection against a disk-shaped table. The configurations of the tabletop, such as its radius, center and width are stored as attributes of obstacle and you can use those for collision detection. Your $128 \times 128$ cloth shall behave as in this [video](refs/phase5.mp4).

# Second Part: Simulating deformable 2D objects
In this part, you will implement a simulator for a 2D deformable gingerbread man, with three types of constitutive models.

## Code Structure and GUI
You are provided with a GUI system, with the following modes and controls already implemented. Make sure you play with the starter code to get familiar with these controls.

Run `python3 fem_starter.py` and you should see an interface, with a gingerbread man displayed at the center.

By default, the GUI window is set to size 600 by 600. Do NOT resize the window if you are using a Mac, since this will mess up the mouse cursor detection in Taichi GGUI, due to potential clamping of the window display on the screen. However, window resizing is fine on Ubuntu. We have not tested on Windows machines, so please let us know if it works.

The system has three modes, namely, the Edit Mode, the Simulation Mode and Collision Mode.

In the Edit Mode, you use mouse clicks to toggle the vertices to be pinned in the simulation. A mouse click either adds the closest vertex to the list of pinned vertices, or removes it from the list if it is pinned.

In the Simulation Mode, you use mouse drags to create spring-like forces on the vertices of the object, which result in deformations of the object. The mouse drag only exerts a spring force on a *single* vertex that is closest to the starting cursor position of the mouse drag, and the strength of the force should be proportional to the distance between the starting and ending cursor positions of the drag. Once a mouse drag is initiated, a red arrow is displayed to indicate the vertex to be acted upon, as well as the magnitude and direction of the drag, and the arrow disappears upon releasing the mouse. The spring force persists throughout the mouse drag motion, and vanishes upon releasing the mouse.

In the Collision Mode, gravity is turned-on and a gingerbread house will show up to test your collision implementation. You can still use mouse drags to apply force to the gingerbread and create interesting collision motion, but since we won't implement a super robust collision detection and resolution system in this assignment, don't apply too large of a force to the gingerbread man to avoid penetration and the instability ensued.

The key/mouse controls are as follows:

* 'C'/'V'/'N': Switch between different constitute models (Corotated linear/St. Venant-Kirchhoff/Neo-Hookean). Note that switching to a different constitutive model re-initializes the positions and velocities of the object (i.e. resets to initial state).
* 'M': Switch between edit mode and simulation mode (default is edit mode)
* 'T': Switch between the collision mode and the edit mode. 
* 'R': When pressed in edit mode, remove all pins; in simulation mode, re-initialize the positions and velocities of the object.
* 'D': Toggle damping (default is no damping).
* 'SPACE': Toggle simulation state between paused and running.
* Left mouse button down (in edit mode): Adds/Removes pin on the vertex that is closest to the cursor position.
* Left mouse button down *and drag* (in simulation mode): Creates a spring force along the mouse drag direction with magnitude proportional to the amount of mouse drag that is exerted on the single vertex closest to the cursor position when the left mouse button pressed. Only one spring force is active at a time, i.e. the previous force is removed when a new force is created on an arbitrary vertex.

Now go ahead and make the gingerbread man move!

## Compute the deformation gradient $\mathbf{F}$
The first step is simple: Use the formula from class to compute the deformation gradient. You are already provided with a vector **x** that contains the vertex positions, and a vector of triangles (finite elements) where each entry is a tuple containing the indices of the three vertices composing a face. Simply write two functions: one computes the $\mathbf{D}$ (or $\mathbf{D}_0$, when called for the initial state) of each element, and the other computes $\mathbf{F}$ using the formula from class. Note that everything is in 2D, as what has been illustrated in class.

## Compute the first Piola-Kirchhoff stress tensor $\mathbf{P}(\mathbf{F})$
Now you will compute $\mathbf{P}(\mathbf{F})=\frac{\partial\psi(\mathbf{F})}{\partial \mathbf{F}}$ for three constitutive models, namely, the corotated linear model, the St. Venant-Kirchhoff (StVK) model, and the Neo-Hookean model. The value of `ModelSelector[None]=0/1/2` tracks the current model being used.

The first two calculations are provided in the [course notes](https://www.cs.cornell.edu/courses/cs5643/2025sp/notes/deformation.pdf).

We provide the first Piola-Kirchhoff tensor for the Neo-Hookean model, so you do not need to derive the derivative yourself:

$$\mathbf{P}(\mathbf{F}) = \mu*(\mathbf{F}-\mathbf{F}^{-T})+\lambda*\ln{(\det(\mathbf{F}))}*\mathbf{F}^{-T}$$

Additionally, you will need to create fields for the Lame parameters $\lambda$ and $\mu$, and compute them from the Young's Modulus and Poisson's Ratio.

## Compute forces
Now you will compute the forces exerted on each element/triangle (denoted by $e$), and map to forces exerted on the three vertices of the element. 

$$\mathbf{f}_i=-\frac{\partial U}{\partial\mathbf{x}_i}=-\sum_e A_e\frac{\partial\psi(\mathbf{F}_e)}{\partial \mathbf{F}_e}\frac{\partial\mathbf{F}_e}{\partial\mathbf{x}_i}=-\sum_e A_e\mathbf{P}(\mathbf{F}_e)\frac{\partial\mathbf{F}_e}{\partial\mathbf{x}_i}$$

## Compute symplectic Euler time integration
This part is very similar to the cloth simulation in PA1. In addition to the forces computed from the previous step, you will need to account for the vertices with pinned positions, as well as the vertex where spring force is applied. In addition, the object movement is bounded by the four edges of the GUI, i.e. at every time step, you should detect if any vertex goes beyond the boundary, and project it onto the boundary.

[This video](refs/gingerbread_pinned.mp4) shows the target behavior of the simulator, with time steps of 0.167 milliseconds. Note that your simulation result might not exactly match the video, due to arbitrary pinning of vertices. The behavior should agree though, i.e. a handle-based force should exhibit in the mouse drag direction.

Once you finish implementing the simulator, you should switch between the three different constitutive models, observe their differences, and report your findings. You are also encouraged to create sliders for the Young's modulus and Poisson's ratio (with appropriate ranges) and play with different values of these parameters. Note that for the changes in parameters to take effect, you should also update the Lame coefficients correspondingly.

## Collision against hard boundaries
The collision is essentially the same as what you did for cloth! We use the very same collision detection process for a set of boundaries stored in `house.boundaries`, where each boundary has attributes like `p`, which stores one of its edge points, `n`, which stores its normal and `eps` that stores the radius of the contact region. Try to implement the very same collision detection and response process as you did for cloth and test your collision detection by switching to the collision mode and make sure your gingerbread man bounce in the house correctly as shown in this [video](refs/gingerbread_housedance.mp4).

## Testing your implementation
To test your implementation, we provide a simple test scenario of stretching/compressing a rectangle in the alternate starter code `fem_test_starter.py`. Here, the vertices of the rest shape are stored in `x_rest`, and the vertices of the compressed/stretched shapes are stored in `x_compress` and `x_stretch`, which get assigned to the field storing the deformed vertex positions `x`. The top and bottom edges are pinned, i.e. in the compressed mode, the top edge are fixed at 0.6 times the rectangle's original height, and in the stretched mode, twice the original height. For calculating $F$, you will need $D_0$ from the rest shape $\mathbf{x}_\mathrm{rest}$ and $D$ from the deformed shape $\mathbf{x}$. If you plug your implementation into this starter code, you will be able to see a clear difference in the three models under the stretched/compressed settings that should match [this video](refs/compression_test.mp4). Pressing the UP/DOWN arrow keys switches between the stretched/compressed shapes. 

# Submission
You need to include two things in your submission:

* A pdf file including a link to the chosen commit for your submission. If you submit a  link just to your repository, we will assume that you would like us to look at the latest commit on the `main` branch.
* A demo video demonstrating the functionalities of your simulators (please make sure that we can easily run your code and generate similar results shown in the demo). You can do something creative with both of the simulators in this demo, like shading the tables and gingerbread better, adding some particle system like snow to the background. You can also gain some bonus credit if you would like to go an extra mile to implement self-collision for the cloth or the gingerbread man. Any other technical improvements like adding friction to all the contacts are also encouraged (but of course, not necessary)!

