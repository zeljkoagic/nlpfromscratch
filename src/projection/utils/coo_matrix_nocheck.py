import numpy as np

class CooMatrix:
    def __init__(self, row, col, data, shape):
        self.row = np.array(row, dtype=np.int32)
        self.col = np.array(col, dtype=np.int32)
        self.data = np.array(data)
        self.shape = shape

    def standardize(self):
        self.data -= self.data.mean()
        self.data /= self.data.std()
