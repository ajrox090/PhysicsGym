import unittest
import numpy as np

from src.util.burgers_util import GaussianClash


class MyTestCase(unittest.TestCase):

    def test_gaussian_clash(self):
        self.force = GaussianClash(1)

        idx = np.random.uniform(0, 1, size=10)
        a = self.force.sample_at(idx)
        assert a is not None
        print(a)


if __name__ == '__main__':
    unittest.main()
