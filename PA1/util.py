import taichi as ti
import taichi.math as tm

# For pinning vertices / creating handle-based forces
@ti.data_oriented
class DistanceMap:
    def __init__(self, N, x):
        self.N = N
        self.x = x

        # To store mouse-vertex distance detection
        self.d = ti.field(ti.f32, N)
        self.d_temp = ti.field(ti.f32, N)
        self.di = ti.field(ti.i32, N)
        self.di_temp = ti.field(ti.i32, N)


    def get_closest_vertex(self, p: tm.vec2):
        self.fillDMap(p)
        minDN = self.N
        fromD2Dtemp = True

        while minDN > 1:
            minDN = self.DC_min(minDN, fromD2Dtemp)
            fromD2Dtemp = not fromD2Dtemp

        if fromD2Dtemp:
            min_dist = self.d[0]
            min_idx = self.di[0]
        else:
            min_dist = self.d_temp[0]
            min_idx = self.di_temp[0]

        # Random clicks that are very far away do not affect the vertices
        if min_dist > 0.2:
            min_idx = -1
        return min_idx


    # Fill a vertex-mouse distance map
    @ti.kernel
    def fillDMap(self, p: tm.vec2):
        for i in self.d:
            self.d[i] = (self.x[i] - p).dot(self.x[i] - p)
            self.di[i] = i


    # Use divide and conquer to find the minimum distance
    @ti.kernel
    def DC_min(self, arr_N: ti.i32, fromD2Dtemp: bool) -> ti.i32:
        if fromD2Dtemp:
            for i in range(0, arr_N // 2):
                self.d_temp[i] = ti.min(self.d[i * 2], self.d[i * 2 + 1])
                self.di_temp[i] = self.di[i * 2] if self.d[i * 2] < self.d[i * 2 + 1] else self.di[i * 2 + 1]

            if arr_N % 2 != 0:
                self.d_temp[arr_N // 2 - 1] = ti.min(self.d_temp[arr_N // 2 - 1], self.d[arr_N - 1])
                self.di_temp[arr_N // 2 - 1] = self.di_temp[arr_N // 2 - 1] if self.d_temp[arr_N // 2 - 1] < self.d[arr_N - 1] else self.di[
                    arr_N - 1]
        else:
            for i in range(0, arr_N // 2):
                self.d[i] = ti.min(self.d_temp[i * 2], self.d_temp[i * 2 + 1])
                self.di[i] = self.di_temp[i * 2] if self.d_temp[i * 2] < self.d_temp[i * 2 + 1] else self.di_temp[i * 2 + 1]

            if arr_N % 2 != 0:
                self.d[arr_N // 2 - 1] = ti.min(self.d[arr_N // 2 - 1], self.d_temp[arr_N - 1])
                self.di[arr_N // 2 - 1] = self.di[arr_N // 2 - 1] if self.d[arr_N // 2 - 1] < self.d_temp[arr_N - 1] else self.di_temp[arr_N - 1]

        return arr_N // 2