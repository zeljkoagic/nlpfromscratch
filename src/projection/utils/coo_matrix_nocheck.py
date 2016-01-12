import numpy as np

class CooMatrix:
    def __init__(self, row, col, data, shape):
        self.row = np.array(row, dtype=np.uint32)
        self.col = np.array(col, dtype=np.uint32)
        self.data = np.array(data)
        self.shape = shape
