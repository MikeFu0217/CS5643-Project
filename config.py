import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Config:
    def __init__(self,
                 n: int = 64,
                 gravity: list = [0, -9.81, 0],
                 contact_eps: float = 0.01,
                 self_contact_eps: float = 0.02,):
        # Grid size n
        self.n = n

        # Gravity
        self.gravity = ti.Vector(gravity)

        # Simulation parameters
        self.stepsize = 2e-2 / self.n
        self.ns = int((1/60) // self.stepsize)
        self.dt = (1/60) / self.ns

        # Model selection
        self.model_names = "vcn"
        self.model = 'v'
        self.prev_model = 'v'
        self.ModelSelector = ti.field(ti.i32, ())
        self.ModelSelector[None] = 0

        # Collision
        self.obstacle_names = ["Sphere", "Table"]
        self.obstacle = "Sphere"
        self.CollisionSelector = ti.field(ti.i32, ())
        self.CollisionSelector[None] = 0

        # Pinning
        self.pin_options = [[], [0], [0, n-1], [0, n*n-1], [0, n*(n-1)], [0, n-1, n*(n-1), n*n-1]]
        self.pin = 0

        # Self-collision
        self.contact_eps = contact_eps
        self.self_contact_eps = self_contact_eps
        self.self_collision = 1
        self.self_collision_enabled = ti.field(ti.i32, ())
        self.self_collision_enabled[None] = 1

        # Friction
        self.friction = 0
        self.friction_enabled = ti.field(ti.i32, ())
        self.friction_enabled[None] = 0

        # Bending
        self.bending = 0
        self.bending_enabled = ti.field(ti.i32, ())
        self.bending_enabled[None] = 0

        # Force color display
        self.show_force_color = 0

    def update_ModelSelector(self):
        self.ModelSelector[None] = self.model_names.find(self.model)

    def update_CollisionSelector(self):
        if self.obstacle in self.obstacle_names:
            self.CollisionSelector[None] = self.obstacle_names.index(self.obstacle)
        else:
            self.CollisionSelector[None] = -1

    def update_self_collision(self):
        if self.self_collision == 0:
            self.self_collision_enabled[None] = 0
        else:
            self.self_collision_enabled[None] = 1
    
    def update_friction(self):
        if self.friction == 0:
            self.friction_enabled[None] = 0
        else:
            self.friction_enabled[None] = 1

    def update_bending(self):
        if self.bending == 0:
            self.bending_enabled[None] = 0
        else:
            self.bending_enabled[None] = 1