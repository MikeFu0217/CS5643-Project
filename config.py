import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Config:
    def __init__(self,
                 n: int = 64,
                 gravity: list = [0, -9.81, 0]):
        # Grid size n
        self.n = n

        # Gravity
        self.gravity = ti.Vector(gravity)

        # Simulation parameters
        self.stepsize = 2e-2 / self.n
        self.ns = int((1/60) // self.stepsize)
        self.dt = (1/60) / self.ns

        # Model selection
        self.model_names = ['c', 'v', 'n']
        self.model = 'c'
        self.prev_model = 'c'