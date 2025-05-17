import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Cloth:
    def __init__(self,
                 n,
                 pos: list = [-0.1, 0.6, 0.1],
                 pins: list = [0, 1]
                 ):
        self.n = n
        self.N = n * n
        self.off_x, self.off_y, self.off_z = pos
        self.quad_size = 1.0 / (self.n - 1)
        
        # for pin shifting
        self.shift_y = 0.0
        self.shift_away = 0.0

        # vertices
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.N))
        
        # trangles: each triangle is defined by 3 vertices
        self.N_triangles = (self.n - 1) * (self.n - 1) * 2
        
        self.triangles_np = np.zeros((self.N_triangles, 3), dtype=np.int32)
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                idx = i * (self.n - 1) + j
                self.triangles_np[idx, 0] = i * self.n + j
                self.triangles_np[idx, 1] = i * self.n + j + 1
                self.triangles_np[idx, 2] = (i + 1) * self.n + j

                self.triangles_np[idx + self.N_triangles // 2, 0] = (i + 1) * self.n + j
                self.triangles_np[idx + self.N_triangles // 2, 1] = i * self.n + j + 1
                self.triangles_np[idx + self.N_triangles // 2, 2] = (i + 1) * self.n + j + 1

        self.triangles = ti.Vector.field(3, dtype=int, shape=(self.N_triangles, ))
        self.triangles.from_numpy(self.triangles_np)

        # pinning, indexes of vertices
        self.MAX_PINS = 4
        self.pins = ti.field(dtype=int, shape=(self.MAX_PINS, ))
        self.pin_cnt  = ti.field(int, shape=())
        self.set_pins(pins)

        # cloth system state
        x_rest_np = np.zeros((self.N, 3), dtype=np.float32)
        for i in range(self.n):
            for j in range(self.n):
                idx = i * self.n + j
                x_rest_np[idx, 0] = i * self.quad_size + self.off_x
                x_rest_np[idx, 1] = self.off_y
                x_rest_np[idx, 2] = j * self.quad_size + self.off_z

        self.x_rest = ti.Vector.field(3, dtype=ti.f32, shape=self.N)
        self.x_rest.from_numpy(x_rest_np)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.N)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.N)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=self.N)

        self.max_force = ti.field(ti.f32, ())
        self.max_force[None] = 1.5

    def set_pins(self, pins: list):
        self.pins.fill(-1)
        for i, idx in enumerate(pins):
            self.pins[i] = idx
        self.pin_cnt[None] = len(pins)


    @ti.func
    def init_state(self):
        # Initialize the current position and velocity of the cloth
        for i in range(self.N):
            self.x[i] = self.x_rest[i]
            self.v[i] = [0.0, 0.0, 0.0]

    def init_shift(self):
        self.shift_y = 0.0
        self.shift_away = 0.0