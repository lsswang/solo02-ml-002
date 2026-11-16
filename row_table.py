"""Host-side row storage adapter. Access traffic counts are observable by callers."""
import numpy as np


class ArrayRowTable:
    def __init__(self, values):
        self._values = np.array(values, dtype=np.float64, copy=True)
        self.shape = self._values.shape
        self.read_rows = self.write_rows = 0

    def read(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self.read_rows += ids.size
        return self._values[ids].copy()

    def write(self, ids, values):
        ids = np.asarray(ids, dtype=np.int64)
        self.write_rows += ids.size
        self._values[ids] = values

    def traffic(self):
        return {'read_rows': self.read_rows, 'write_rows': self.write_rows}
