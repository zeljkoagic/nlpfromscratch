import numpy as np

class CooMatrix:
    def __init__(self, row, col, data, shape):
        self.row = np.array(row, dtype=np.int32)
        self.col = np.array(col, dtype=np.int32)
        self.data = np.array(data)
        self.shape = shape

    @classmethod
    def standardize(cls, M):
        M.data = (M.data - M.data.mean()) / M.data.std()
