import taichi as ti
import taichi.math as tm
import numpy as np

from cloth import Cloth

@ti.data_oriented
class Render:
    def __init__(self, cloth: Cloth):
        self.n = cloth.n
        self.cloth = cloth
        self.vertices = ti.Vector.field(3, dtype=float, shape=cloth.N)
        self.indices = ti.field(int, shape=cloth.N_triangles * 3)
        self.colors = ti.Vector.field(3, dtype=float, shape=cloth.N)


    # Set up the mesh
    @ti.kernel
    def initialize_mesh_indices(self):
        for i, j in ti.ndrange(self.n - 1, self.n - 1):
            quad_id = (i * (self.n - 1)) + j
            # 1st triangle of the square
            self.indices[quad_id * 6 + 0] = i * self.n + j
            self.indices[quad_id * 6 + 1] = (i + 1) * self.n + j
            self.indices[quad_id * 6 + 2] = i * self.n + (j + 1)
            # 2nd triangle of the square
            self.indices[quad_id * 6 + 3] = (i + 1) * self.n + j + 1
            self.indices[quad_id * 6 + 4] = i * self.n + (j + 1)
            self.indices[quad_id * 6 + 5] = (i + 1) * self.n + j

        for i, j in ti.ndrange(self.n, self.n):
            if (i % 4 == 0) or (j % 4 == 0):
                self.colors[i * self.n + j] = (1.0, 0.97, 0.95)
            else:
                self.colors[i * self.n + j] = (1.0, 0.2, 0.4)

    # Copy vertex state into mesh vertex positions
    @ti.kernel
    def update_vertices(self):
        for i, j in ti.ndrange(self.n, self.n):
            idx = i * self.n + j
            self.vertices[idx] = self.cloth.x[idx]

    @ti.func
    def _map_force_to_color(self, f_mag):
        # 将力大小归一化到 [0,1]
        # 0 -> 纯蓝 (0,0,1)
        # 1 -> 纯红 (1,0,0)
        ti.loop_config(serialize=True)
        t = f_mag / self.max_force[None]
        t = ti.min(t, 1.0)
        # 简单线性插值：红分量从 0→1，蓝分量从 1→0
        return ti.Vector([t, 0.0, 1.0 - t])
    
    @ti.kernel
    def set_force_colors(self):
        for i in range(self.cloth.N):
            f = self.cloth.force[i]
            f_mag = tm.length(f)
            t = f_mag / self.cloth.max_force[None]
            # 简单线性插值：红分量从 0→1，蓝分量从 1→0
            self.colors[i][0] = t
            self.colors[i][1] = 0.0
            self.colors[i][2] = 1.0 - t


    def set_color_to_default(self):
        for i, j in ti.ndrange(self.n, self.n):
            if (i % 4 == 0) or (j % 4 == 0):
                self.colors[i * self.n + j] = (1.0, 0.97, 0.95)
            else:
                self.colors[i * self.n + j] = (1.0, 0.2, 0.4)
