import taichi as ti
import taichi.math as tm
import numpy as np

from config import Config
from cloth import Cloth
from scene import Scene, Init

@ti.data_oriented
class Physics:
    def __init__(self,
                 cfg: Config,
                 cloth: Cloth,
                 obstacle: Scene,
                 E: float = 450,
                 nu: float = 0.15,
                 k_drag: float = 1.2,):
        self.cfg = cfg
        self.cloth = cloth
        self.obstacle = obstacle

        self.m = 1.0/(cfg.n * cfg.n)

        self.k_drag = ti.field(ti.f32, ())
        self.k_drag[None] = k_drag

        # physical quantities
        self.YoungsModulus = ti.field(ti.f32, ())
        self.PoissonsRatio = ti.field(ti.f32, ())
        # Default values for reproducing the reference
        self.YoungsModulus[None] = E
        self.PoissonsRatio[None] = nu

        # 3D Lame parameters
        self.Lambda = ti.field(ti.f32, ())
        self.Mu = ti.field(ti.f32, ())
        self.compute_lame_parameters()

        # deformation gradient
        self.D0 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cloth.N_triangles)
        self.Tk = ti.field(dtype=ti.f32, shape=cloth.N_triangles)
        self.D = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cloth.N_triangles)
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cloth.N_triangles)
        self.P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cloth.N_triangles)
        self.H = ti.Matrix.field(3, 3, dtype=ti.f32, shape=cloth.N_triangles)

    @ti.kernel
    def compute_lame_parameters(self):
        E = self.YoungsModulus[None]
        nu = self.PoissonsRatio[None]
        self.Lambda[None] = (E * nu) / ((1 + nu) * (1 - 2*nu))
        self.Mu[None] = E / (2 * (1 + nu))

    def set_parameters(self, E: float, nu: float, k_drag: float):
        self.YoungsModulus[None] = E
        self.PoissonsRatio[None] = nu
        self.k_drag[None] = k_drag
        self.compute_lame_parameters()

    def get_parameters(self):
        return {
            'E': self.YoungsModulus[None],
            'nu': self.PoissonsRatio[None],
            'lambda': self.Lambda[None],
            'mu': self.Mu[None]
        }
    
    # Compute D0 and Tk (3D space)
    @ti.func
    def compute_D0_Tk(self):
        for i in range(self.cloth.N_triangles):
            a, b, c = self.cloth.triangles[i]
            Xa, Xb, Xc = self.cloth.x_rest[a], self.cloth.x_rest[b], self.cloth.x_rest[c]
            N0 = tm.cross(Xb - Xa, Xc - Xa)
            self.D0[i] = tm.mat3(Xb - Xa, Xc - Xa, N0).transpose()
            self.Tk[i] = N0.norm() * 0.5

    # Compute D (3D space)
    @ti.func
    def compute_D(self):
        for i in range(self.cloth.N_triangles):
            a, b, c = self.cloth.triangles[i]
            Xa, Xb, Xc = self.cloth.x[a], self.cloth.x[b], self.cloth.x[c]
            N0 = tm.cross(Xb - Xa, Xc - Xa)
            self.D[i] = tm.mat3(Xb - Xa, Xc - Xa, N0).transpose()
    
    # Compute F (3D space)
    @ti.func
    def compute_F(self):
        for i in range(self.cloth.N_triangles):
            self.F[i] = self.D[i] @ (self.D0[i].inverse())

    # Compute P (3D space)
    @ti.func
    def compute_P_c(self):
        for i in range(self.cloth.N_triangles):
            Fi = self.F[i]
            U, sigma, V = ti.svd(Fi)
            R = U @ V.transpose()
            S = V @ sigma @ V.transpose()
            strain_c = S - ti.Matrix.identity(ti.f32, 3)
            self.P[i] = R @ (2 * self.Mu[None] * strain_c + self.Lambda[None] * strain_c.trace() * ti.Matrix.identity(ti.f32, 3))
    @ti.func
    def compute_P_v(self):
        for i in range(self.cloth.N_triangles):
            Fi = self.F[i]
            green_strain = 0.5 * (Fi.transpose() @ Fi - ti.Matrix.identity(ti.f32, 3))
            self.P[i] = Fi @ (2 * self.Mu[None] * green_strain + self.Lambda[None] * green_strain.trace() * ti.Matrix.identity(ti.f32, 3))
    @ti.func
    def compute_P_n(self):
        for i in range(self.cloth.N_triangles):
            Fi = self.F[i]
            J = ti.max(Fi.determinant(), 1e-6)
            self.P[i] = self.Mu[None] * (Fi - Fi.inverse().transpose()) + self.Lambda[None] * ti.log(J) * Fi.inverse().transpose()

    # Compute H (3D space)
    @ti.func
    def compute_H(self):
        for i in range(self.cloth.N_triangles):
            self.H[i] = - self.Tk[i] * self.P[i] @ self.D0[i].inverse().transpose()

    # Reset force
    @ti.func
    def reset_cloth_force(self):
        for i in range(self.cloth.N):
            self.cloth.force[i] = self.cfg.gravity * self.m

    # Compute internal force
    @ti.func
    def compute_cloth_internal_force(self):
        for i in range(self.cloth.N_triangles):
            Hi = self.H[i]
            a, b, c = self.cloth.triangles[i]
            fb = ti.Vector([Hi[0, 0], Hi[1, 0], Hi[2, 0]])
            fc = ti.Vector([Hi[0, 1], Hi[1, 1], Hi[2, 1]])
            fa = - fb - fc
            self.cloth.force[a] += fa
            self.cloth.force[b] += fb
            self.cloth.force[c] += fc

    # Forward Euler integration
    @ti.func
    def forward_euler(self):
        # Update velocity
        for i in range(self.cloth.N):
            # Update velocity
            self.cloth.v[i] += self.cloth.force[i] / self.m * self.cfg.dt
            # viscous damping
            self.cloth.v[i] *= 1 - self.k_drag[None] * self.cfg.dt

            # Collision with sphere
            if self.cfg.CollisionSelector[None] == 0:
                # Sphere collision
                if tm.length(self.cloth.x[i] - self.obstacle.ball_center[0]) < self.obstacle.ball_radius + self.cfg.contact_eps:
                    normal = tm.normalize(self.cloth.x[i] - self.obstacle.ball_center[0])
                    self.cloth.v[i] -=  tm.min(0, tm.dot(self.cloth.v[i], normal)) * normal
            
        # Pinning
        for i in range(self.cloth.pin_cnt[None]):
            idx = self.cloth.pins[i]
            self.cloth.v[idx] = [0.0, 0.0, 0.0]
            
        # Update position
        for i in range(self.cloth.N):
            self.cloth.x[i] += self.cloth.v[i] * self.cfg.dt