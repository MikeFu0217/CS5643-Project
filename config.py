import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Config:
    def __init__(self,
                 n: int = 64,
                 gravity: list = [0, -9.81, 0],
                 contact_eps: float = 0.01,):
        # Grid size n
        self.n = n

        # Gravity
        self.gravity = ti.Vector(gravity)

        # Simulation parameters
        self.stepsize = 2e-2 / self.n
        self.ns = int((1/60) // self.stepsize)
        self.dt = (1/60) / self.ns

        # Model selection
        self.model_names = "cvn"
        self.model = 'c'
        self.prev_model = 'c'
        self.ModelSelector = ti.field(ti.i32, ())
        self.ModelSelector[None] = 0

        # Collision
        self.obstacle_names = ["Sphere"]
        self.obstacle = "Sphere"
        self.CollisionSelector = ti.field(ti.i32, ())
        self.CollisionSelector[None] = 0

        self.contact_eps = contact_eps

    def update_ModelSelector(self):
        self.ModelSelector[None] = self.model_names.find(self.model)

    def update_CollisionSelector(self):
        if self.obstacle in self.obstacle_names:
            self.CollisionSelector[None] = self.obstacle_names.index(self.obstacle)
        else:
            self.CollisionSelector[None] = -1
