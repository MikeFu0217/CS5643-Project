import taichi as ti
import taichi.math as tm
from pywavefront import Wavefront

import numpy as np
from enum import Enum
from math import pi

vec2 = tm.vec2
vec3 = tm.vec3
vec3i = ti.types.vector(3, int)

@ti.dataclass
class Boundary:
    p: vec2
    n: vec2
    eps:float

class Init(Enum):
    CLOTH_SPHERE="cloth_sphere"
    CLOTH_TABLE="cloth_table"
    GB_BOX = "giner_bread_&_box"

# Scene-related Data
@ti.data_oriented
class Scene:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.startScene()

    def startScene(self):
        if self.initial_state == Init.CLOTH_SPHERE:
            self.init_sphere()
        elif self.initial_state == Init.CLOTH_TABLE:
            print(f"Initializing table")
            self.init_table()
            self.init_table_mesh()
        elif self.initial_state == Init.GB_BOX:
            self.init_gingerbread_box()

    def init_box_boundaries(self):
        self.house_width = 0.9
        self.house_height = 0.6
        self.house_roof_height = 0.3
        self.house_xcenter = 0.5
        self.house_ycenter = 0.4
        
        if not hasattr(self, "boundaries"):
            self.nboundary = 5
            self.boundaries = Boundary.field(shape=(5,))
            ps_np = np.array([  [self.house_xcenter - 0.5 * self.house_width, self.house_ycenter + 0.5 * self.house_height],
                                [self.house_xcenter - 0.5 * self.house_width, self.house_ycenter - 0.5 * self.house_height],
                                [self.house_xcenter + 0.5 * self.house_width, self.house_ycenter - 0.5 * self.house_height],
                                [self.house_xcenter + 0.5 * self.house_width, self.house_ycenter + 0.5 * self.house_height],
                                [self.house_xcenter, self.house_ycenter + 0.5 * self.house_height + self.house_roof_height]],
                                dtype=np.float32)
            rooftop_right = (ps_np[4] - ps_np[3]) / np.linalg.norm(ps_np[4] - ps_np[3])
            rooftop_left = (ps_np[0] - ps_np[4]) / np.linalg.norm(ps_np[0] - ps_np[4])

            self.boundaries.p.from_numpy(ps_np)
            self.boundaries.n.from_numpy(np.array([[1, 0],
                                                    [0, 1],
                                                    [-1, 0],
                                                    [-rooftop_right[1], rooftop_right[0]], 
                                                    [-rooftop_left[1], rooftop_left[0]]], dtype=np.float32))
            self.boundaries.eps.from_numpy(np.ones(5,  dtype=np.float32) * 1e-2)
            self.boundary_indices = ti.field(shape=(10,), dtype=ti.i32)

    @ti.kernel
    def init_boundary_indices(self):
        for i in range(5):
            self.boundary_indices[2 * i] = i
            self.boundary_indices[2 * i + 1] = (i + 1) % 5

    def init_sphere(self):
        # and for interest there is a collision object, just a sphere
        self.ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
        self.ball_center[0] = [0.5, 0, 0.5]
        self.ball_radius = 0.3

        # Read geometry
        obj = Wavefront('./models/sphere.obj', collect_faces=True)
        va = self.ball_radius * np.array(obj.vertices, dtype=np.float32) + self.ball_center.to_numpy()
        self.verts = ti.Vector.field(3, shape=va.shape[0], dtype=float)
        self.verts.from_numpy(va)

        ta = np.array(obj.meshes['Icosphere'].faces, dtype=np.int32)
        self.tris = ti.field(int, shape=np.prod(ta.shape))
        self.tris.from_numpy(ta.ravel())

    def init_table(self):
        self.lod = 20
        tabletop_center_np = np.asarray([[0.5, 0.1, 0.5]], dtype=np.float32) 
        tablebottom_center_np = np.asarray([[0.5, -0.4, 0.5]], dtype=np.float32) 
        self.tabletop_center = ti.Vector.field(3, dtype=float, shape=(1, ))
        self.tablebottom_center = ti.Vector.field(3, dtype=float, shape=(1, ))
        self.tabletop_center.from_numpy(tabletop_center_np)
        self.tablebottom_center.from_numpy(tablebottom_center_np)

        self.tabletop_height = 0.04
        self.tablebottom_height = 0.03
        self.tabletop_radius = 0.4
        self.tablebottom_radius = 0.15
        self.tablepost_radius = 0.02

        tablepost_center_np = 0.5 * (tabletop_center_np - np.array([[0.0, 0.5 * self.tabletop_height, 0.0]], dtype=np.float32) + tablebottom_center_np + np.array([[0.0, 0.5 * self.tablebottom_height, 0.0]], dtype=np.float32))
        self.tablepost_center = ti.Vector.field(3, shape=(1, ), dtype=float)
        self.tablepost_center.from_numpy(tablepost_center_np)
        self.tablepost_height = (tabletop_center_np[0, 1] - tablebottom_center_np[0, 1]) - (self.tabletop_height + self.tablebottom_height) * 0.5

        tabletop_upside_center_np = tabletop_center_np + np.array([[0.0, 0.5 * self.tabletop_height, 0.0]], dtype=np.float32)
        self.tabletop_upside_center = ti.Vector.field(3, shape=(1, ), dtype=float)
        self.tabletop_upside_center.from_numpy(tabletop_upside_center_np)

        # Read geometry
        self.num_tableplate_vs = (self.lod + 1) * 2
        self.num_table_vs = self.num_tableplate_vs * 3
        self.verts = ti.Vector.field(3, shape=self.num_table_vs, dtype=float)

        self.num_tableplate_fs = 4 * self.lod # 4 lod triangles for a plate with lod edges on its rim
        self.num_table_fs = int(self.num_tableplate_fs * 2.5)
        self.tris = ti.field(int, shape=self.num_table_fs * 3) 
    
    def init_gingerbread_box(self):
        self.init_box_boundaries()
        self.init_boundary_indices()

    @ti.func
    def fill_plate_verts(self, local_i:int, slice_angle:float, plate_r:float, plate_h:float, plate_c: tm.vec3):
        table_vert = vec3(0.0, 0.0, 0.0)
        if local_i < self.lod:
            angle = slice_angle * local_i
            table_vert = vec3(ti.cos(angle) * plate_r, plate_h * 0.5, ti.sin(angle) * plate_r) + plate_c
        elif local_i == self.lod:
            table_vert = vec3(0.0, plate_h * 0.5, 0.0) + plate_c
        elif local_i >= self.num_tableplate_vs // 2 and local_i < self.num_tableplate_vs - 1:
            angle = slice_angle * (local_i - self.num_tableplate_vs // 2)
            table_vert = vec3(ti.cos(angle) * plate_r, -plate_h * 0.5, ti.sin(angle) * plate_r) + plate_c
        else:
            table_vert = vec3(0.0, - plate_h * 0.5, 0.0) + plate_c
        return table_vert


    @ti.func
    def fill_plateSide_tris(self, local_fi, vi_start):
        tris = vec3i(0, 0, 0)
        if local_fi < self.lod:
            local_fi = local_fi
            tris.x = local_fi + vi_start
            tris.y = (local_fi + 1) % self.lod + vi_start
            tris.z = local_fi + vi_start + self.num_tableplate_vs // 2
        elif local_fi >= self.lod and local_fi < self.lod * 2:
            local_fi = local_fi - self.lod
            tris.x = local_fi + vi_start + self.num_tableplate_vs // 2
            tris.y = (local_fi + 1) % self.lod + vi_start + self.num_tableplate_vs // 2
            tris.z = (local_fi + 1) % self.lod + vi_start 
        return tris

    @ti.func
    def fill_plateCap_tris(self, local_fi, vi_start):
        tris = vec3i(0, 0, 0)
        if local_fi < self.lod:
            tris.x = self.num_tableplate_vs // 2 - 1 + vi_start
            tris.y = local_fi + vi_start
            tris.z = (local_fi + 1) % self.lod + vi_start
        elif local_fi >= self.lod and local_fi < self.lod * 2:
            local_fi = local_fi - self.lod
            vi_start = vi_start + self.num_table_vs // 2
            tris.x = self.num_tableplate_vs // 2 - 1 + vi_start
            tris.y = local_fi + vi_start
            tris.z = (local_fi + 1) % self.lod + vi_start
        elif local_fi >= 2 * self.lod and local_fi < self.lod * 3:
            local_fi = local_fi - 2 * self.lod
            tris.x = local_fi + vi_start
            tris.y = (local_fi + 1) % self.lod + vi_start
            tris.z = local_fi + vi_start + self.num_tableplate_vs // 2
        else:
            local_fi = local_fi - 3 * self.lod
            tris.x = local_fi + vi_start + self.num_tableplate_vs // 2
            tris.y = (local_fi + 1) % self.lod + vi_start + self.num_tableplate_vs // 2
            tris.z = (local_fi + 1) % self.lod + vi_start 
            
        return tris

    @ti.kernel
    def init_table_mesh(self):
        slice_angle = 2.0 * pi / float(self.lod)
        for i in range(self.num_table_vs):
            if i < self.num_tableplate_vs:
                self.verts[i] = self.fill_plate_verts(i, slice_angle, self.tabletop_radius, self.tabletop_height, self.tabletop_center[0])
            elif i >= self.num_tableplate_vs and i < self.num_tableplate_vs * 2:
                self.verts[i] = self.fill_plate_verts(i-self.num_tableplate_vs, slice_angle, self.tablebottom_radius, self.tablebottom_height, self.tablebottom_center[0])
            else:
                self.verts[i] = self.fill_plate_verts(i-self.num_tableplate_vs*2, slice_angle, self.tablepost_radius, self.tablepost_height, self.tablepost_center[0])

        for i in range(self.num_table_fs):
            tri = vec3i(0, 0, 0)
            if i < self.num_tableplate_fs:
                # Triangles for the tabletop
                tri = self.fill_plateCap_tris(i, 0)
            elif i >= self.num_tableplate_fs and i < self.num_tableplate_fs * 2:
                # Triangles for the tablebottom
                tri = self.fill_plateCap_tris(i - self.num_tableplate_fs, self.num_tableplate_vs)
            else:
                # Triangles for the tablepost
                tri = self.fill_plateSide_tris(i - 2 * self.num_tableplate_fs, self.num_tableplate_vs * 2)

            self.tris[3 * i] = tri.x
            self.tris[3 * i + 1] = tri.y
            self.tris[3 * i + 2] = tri.z
