import numpy as np
from phi.flow import struct
from phi.physics.field import AnalyticField


@struct.definition()
class GaussianClash(AnalyticField):

    def __init__(self):
        AnalyticField.__init__(self, rank=1)

    def sample_at(self, idx, collapse_dimensions=True):
        leftloc = np.random.uniform(0.2, 0.4)
        leftamp = np.random.uniform(0, 3)
        leftsig = np.random.uniform(0.05, 0.15)
        rightloc = np.random.uniform(0.6, 0.8)
        rightamp = np.random.uniform(-3, 0)
        rightsig = np.random.uniform(0.05, 0.15)
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self):
        AnalyticField.__init__(self, rank=1)
        self.loc = np.random.uniform(0.4, 0.6)
        self.amp = np.random.uniform(-0.05, 0.05) * 32
        self.sig = np.random.uniform(0.1, 0.4)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data