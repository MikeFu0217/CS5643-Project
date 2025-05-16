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

        # --- BUILD HINGES & REST ANGLES (Python side) ---
        import numpy as _np

        # Get triangle indices and rest positions from Cloth
        tri_np = self.cloth.triangles_np           # shape=(N_tri,3)
        x_rest_np = self.cloth.x_rest.to_numpy()   # shape=(N,3)

        # Collect triangles sharing each undirected edge
        edge_to_tri = {}
        for t_idx, tri in enumerate(tri_np):
            for e in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])):
                key = tuple(sorted(e))
                edge_to_tri.setdefault(key, []).append(t_idx)

        hinges_list = []
        rest_angles_list = []
        for (v0, v1), tris in edge_to_tri.items():
            if len(tris) == 2:
                t1, t2 = tris
                tri1, tri2 = tri_np[t1], tri_np[t2]
                # Find opposite vertices v2, v3
                v2 = [v for v in tri1 if v not in (v0, v1)][0]
                v3 = [v for v in tri2 if v not in (v0, v1)][0]
                hinges_list.append([v0, v1, v2, v3])
                # Compute rest dihedral angle theta_0
                X0, X1, X2, X3 = x_rest_np[[v0, v1, v2, v3]]
                n1 = _np.cross(X1 - X0, X2 - X0)
                n2 = _np.cross(X3 - X0, X1 - X0)
                cos0 = _np.dot(n1, n2) / (_np.linalg.norm(n1) * _np.linalg.norm(n2))
                rest_angles_list.append(_np.arccos(_np.clip(cos0, -1.0, 1.0)))

        # Determine hinge count dynamically
        hinge_count = len(hinges_list)
        self.N_hinges = hinge_count
        # Temporarily store Python-side lists, write to Taichi fields later
        self._hinges_list = hinges_list
        self._rest_angles_list = rest_angles_list
        # --- END BUILD HINGES & REST ANGLES ---

        # --- DECLARE BENDING FIELDS ---
        # Bending stiffness coefficient
        self.bend_stiffness = ti.field(ti.f32, shape=())
        self.bend_stiffness[None] = 1.0  # tune this value as needed

        # Hinges: each stores (v0, v1, v2, v3)
        self.hinges = ti.Vector.field(4, dtype=ti.i32, shape=self.N_hinges)
        # Rest dihedral angles theta_0
        self.rest_angles = ti.field(dtype=ti.f32, shape=self.N_hinges)

        # Write Python lists into Taichi fields
        import numpy as _np  # ensure already imported
        self.hinges.from_numpy(_np.array(self._hinges_list, dtype=_np.int32))
        self.rest_angles.from_numpy(_np.array(self._rest_angles_list, dtype=_np.float32))

        # Other bending parameters
        self.bend_damping = ti.field(ti.f32, shape=())
        self.bend_damping[None] = 0.1  # tune this value as needed
        self.angle_tol = ti.field(ti.f32, shape=())
        self.angle_tol[None] = 1e-3  # tune this value as needed
        # --- END DECLARE BENDING FIELDS ---
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

    # Compute bending force
    @ti.func
    def compute_bending_angle(self, a: int, b: int, c: int, d: int) -> ti.f32:
        # Compute dihedral angle between triangles (a,b,c) and (a,b,d)
        Xa = self.cloth.x[a]
        Xb = self.cloth.x[b]
        Xc = self.cloth.x[c]
        Xd = self.cloth.x[d]
        # Normals of the two triangles
        N1 = tm.cross(Xb - Xa, Xc - Xa)
        N2 = tm.cross(Xd - Xa, Xb - Xa)
        # Cosine of dihedral angle
        cos_theta = tm.dot(N1, N2) / (N1.norm() * N2.norm() + 1e-6)
        # Clamp to valid range
        cos_theta = ti.min(1.0, ti.max(-1.0, cos_theta))
        return ti.acos(cos_theta)
    
    @ti.func
    def compute_bending_forces(self):
        """
        Apply triangle-hinge bending forces with deadzone and damping.
        """

        for i in range(self.N_hinges):
            if self.cfg.bending_enabled[None] == 1:
                a, b, c, d = self.hinges[i]
                Xa = self.cloth.x[a]; Xb = self.cloth.x[b]
                Xc = self.cloth.x[c]; Xd = self.cloth.x[d]

                # compute normals and hinge height
                N1 = tm.cross(Xb - Xa, Xc - Xa)
                N2 = tm.cross(Xd - Xa, Xb - Xa)
                edge_vec = Xb - Xa
                len_edge = edge_vec.norm() + 1e-6
                h_avg = 0.5 * ((N1.norm() + N2.norm()) / len_edge)

                # current dihedral angle
                theta = self.compute_bending_angle(a, b, c, d)
                # deviation from rest
                delta = theta - self.rest_angles[i]
                # --- DEADZONE: clamp small deviations to zero ---
                delta = ti.select(ti.abs(delta) < self.angle_tol[None], 0.0, delta)
                # --- clamp maximum angle change to avoid explosion ---
                delta = ti.max(-0.5, ti.min(0.5, delta))

                # raw force magnitude with geometric weight
                f_mag = - self.bend_stiffness[None] * h_avg * delta
                # clamp max force
                maxF: ti.f32 = 50.0
                f_mag = ti.max(-maxF, ti.min(maxF, f_mag))
                # apply damping
                f_mag *= (1.0 - self.bend_damping[None])

                # distribute along edge direction
                edge_dir = edge_vec.normalized()
                self.cloth.force[a] += -0.5 * f_mag * edge_dir
                self.cloth.force[b] += -0.5 * f_mag * edge_dir
                self.cloth.force[c] +=  0.5 * f_mag * edge_dir
                self.cloth.force[d] +=  0.5 * f_mag * edge_dir

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
                if self.cfg.self_collision_enabled[None] == 1:
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
                    if self.cfg.friction_enabled[None] == 1:
                        self.apply_friction(i, vn, vt)

            
            # Collision with table
            elif self.cfg.CollisionSelector[None] == 1:
                # Retrieve table obstacle parameters
                obstacle = self.obstacles.table
                top_center = obstacle.tabletop_center[0]        # Vector3: center of tabletop
                top_h      = obstacle.tabletop_height          # float : height of table
                top_r      = obstacle.tabletop_radius          # float : radius of table cylinder

                # Particle position
                p = self.cloth.x[i]
                p_x, p_y, p_z = p[0], p[1], p[2]
                top_x, top_y, top_z = top_center[0], top_center[1], top_center[2]

                # Compute horizontal (XZ-plane) distance from table axis
                d_h = tm.sqrt((p_x - top_x) ** 2 + (p_z - top_z) ** 2)

                # Distances to the three table surfaces
                # How far above/below the top face
                pnt_top    = (top_y + top_h / 2) - p_y
                # How far above/below the bottom face
                pnt_bottom = p_y - (top_y - top_h / 2)
                # How far inside/outside the cylindrical side
                pnt_side   = top_r - d_h

                # Check if particle is within contact epsilon of any table face
                if (p_y < top_y + top_h/2 + self.cfg.contact_eps and
                    p_y > top_y - top_h/2 - self.cfg.contact_eps and
                    d_h < top_r + self.cfg.contact_eps):

                    # Determine which face is collided: side vs. top vs. bottom
                    normal = ti.Vector([0.0, 0.0, 0.0])
                    if pnt_side < pnt_top and pnt_side < pnt_bottom:
                        # Side collision: normal points radially outward in XZ plane
                        normal = tm.normalize(ti.Vector([p_x - top_x, 0.0, p_z - top_z]))
                    elif pnt_top < pnt_side and pnt_top < pnt_bottom:
                        # Top face collision: normal is +Y
                        normal = ti.Vector([0.0, 1.0, 0.0])
                    else:
                        # Bottom face collision: normal is -Y
                        normal = ti.Vector([0.0, -1.0, 0.0])

                    # Decompose velocity into normal and tangential components
                    vn = tm.dot(self.cloth.v[i], normal)            # normal component (scalar)
                    vt = self.cloth.v[i] - vn * normal              # tangential component (vector)

                    # Remove any penetrating velocity along the normal
                    self.cloth.v[i] -= tm.min(0.0, vn) * normal

                    # Apply friction along the tangential direction if enabled
                    if self.cfg.friction_enabled[None] == 1:
                        self.apply_friction(i, vn, vt)
            
        # Pinning
        for i in range(self.cloth.pin_cnt[None]):
            idx = self.cloth.pins[i]
            self.cloth.v[idx] = [0.0, 0.0, 0.0]

        # Update position
        for i in range(self.cloth.N):
            self.cloth.x[i] += self.cloth.v[i] * self.cfg.dt

        # --- VELOCITY THRESHOLDING: zero out tiny residual velocities ---
        v_tol: ti.f32 = 1e-4
        for i in range(self.cloth.N):
            if self.cloth.v[i].norm() < v_tol:
                self.cloth.v[i] = ti.Vector([0.0, 0.0, 0.0])