import taichi as ti
import taichi.math as tm
import numpy as np

from config import Config
from cloth import Cloth
from obstacles import Obstacles

@ti.data_oriented
class Physics:
    def __init__(self,
                 cfg: Config,
                 cloth: Cloth,
                 obstacles: Obstacles,
                 E: float = 450,
                 nu: float = 0.15,
                 k_drag: float = 1.2,
                 self_collision_strength: float = 200.0,
                 mu_friction : float = 0.3,):
        self.cfg = cfg
        self.cloth = cloth
        self.obstacles = obstacles

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

        # self-collision
        self.self_collision_strength = ti.field(dtype=ti.f32, shape=())
        self.self_collision_strength[None] = self_collision_strength

        # Friction
        self.mu_friction = ti.field(ti.f32, ())
        self.mu_friction[None] = mu_friction

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

    @ti.func
    def apply_self_collision(self):
        for i in range(self.cloth.N):
            for j in range(self.cloth.N):
                if i != j:
                    pi = self.cloth.x[i]
                    pj = self.cloth.x[j]
                    dir = pi - pj
                    dist = dir.norm()
                    eps = self.cfg.self_contact_eps
                    if dist < eps:
                        force = self.self_collision_strength[None] * (eps - dist) * dir.normalized()
                        self.cloth.force[i] += force


    @ti.func
    def apply_friction(self, i, vn, vt):
        vt_mag = vt.norm()
        if vt_mag > 1e-6:
            friction_force = tm.min(self.mu_friction[None] * abs(vn), vt_mag)
            self.cloth.v[i] -= friction_force * (vt / vt_mag)


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
                obstacle = self.obstacles.sphere
                if tm.length(self.cloth.x[i] - obstacle.ball_center[0]) < obstacle.ball_radius + self.cfg.contact_eps:
                    normal = tm.normalize(self.cloth.x[i] - obstacle.ball_center[0])
                    vn = tm.dot(self.cloth.v[i], normal)
                    vt = self.cloth.v[i] - vn * normal

                    # Remove normal penetration component
                    self.cloth.v[i] -=  tm.min(0, vn) * normal

                    # Apply friction
                    self.apply_friction(i, vn, vt)

            
            # Collision with table
            elif self.cfg.CollisionSelector[None] == 1:
                # Table collision
                obstacle = self.obstacles.table
                p_x, p_y, p_z = self.cloth.x[i][0], self.cloth.x[i][1], self.cloth.x[i][2]
                top_x, top_y, top_z = obstacle.tabletop_center[0][0], obstacle.tabletop_center[0][1], obstacle.tabletop_center[0][2]
                top_h, top_r = obstacle.tabletop_height, obstacle.tabletop_radius
                d_h = tm.sqrt( (p_x-top_x)**2 + (p_z-top_z)**2 )
                
                pnt_top = (top_y + top_h/2) - p_y
                # For the bottom surface: how far above the bottom is the particle?
                pnt_bottom = p_y - (top_y - top_h/2)
                # For the side surface: how far inside the table's radius is the particle?
                pnt_side = top_r - d_h
            
                # If the particle is in inside the table top
                if p_y < top_y + top_h/2 + self.cfg.contact_eps and p_y > top_y - top_h/2 - self.cfg.contact_eps and d_h < top_r + self.cfg.contact_eps:                
                    if pnt_side < pnt_top and pnt_side < pnt_bottom: # Collision with side
                        normal = tm.normalize(ti.Vector([p_x-top_x, 0.0, p_z-top_z]))
                        self.cloth.v[i] -=  tm.min(0, tm.dot(self.cloth.v[i], normal)) * normal
                    elif pnt_top < pnt_side and pnt_top < pnt_bottom: # Collision with top
                        normal = ti.Vector([0, 1, 0])
                        self.cloth.v[i] -=  tm.min(0, tm.dot(self.cloth.v[i], normal)) * normal
                    else: # Collision with bottom
                        normal = ti.Vector([0, -1, 0])
                        self.cloth.v[i] -=  tm.min(0, tm.dot(self.cloth.v[i], normal)) * normal
            
        # Pinning
        for i in range(self.cloth.pin_cnt[None]):
            idx = self.cloth.pins[i]
            self.cloth.v[idx] = [0.0, 0.0, 0.0]

        # Update position
        for i in range(self.cloth.N):
            self.cloth.x[i] += self.cloth.v[i] * self.cfg.dt