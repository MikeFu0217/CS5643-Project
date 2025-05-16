from PA1.scene import *

@ti.data_oriented
class Obstacles:
    def __init__(self):
        self.sphere = Scene(Init.CLOTH_SPHERE)
        self.table = Scene(Init.CLOTH_TABLE)

    def get_obstacle(self, i: int):
        if i == 0:
            return self.sphere
        elif i == 1:
            return self.table
        else:
            return None