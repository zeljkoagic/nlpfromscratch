import numpy as np

class CooMatrix:
    def __init__(self, row, col, data, shape):
        self.row = np.array(row)
        self.col = np.array(col)
        self.data = np.array(data)
        self.shape = shape
