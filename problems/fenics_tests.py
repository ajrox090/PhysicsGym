import unittest

import time
from heat_steady_dev import HeatSteady
from burger_steady_dev import BurgerSteady
from poisson_dev import Poisson


class MyTestCase(unittest.TestCase):

    def test_heat_steady(self):
        print(time.ctime(time.time()))
        print('')
        print('heat_steady_test:')
        print('  2D steady heat equation in a rectangle.')
        savfig = True
        hst = HeatSteady(savfig=savfig)
        hst.define()
        hst.solve()

        print('')
        print('heat_steady_test:')
        print('  Normal end of execution.')
        print('')
        print(time.ctime(time.time()))
        self.assertEqual(True, True)

    def test_burgers_steady(self):
        print(time.ctime(time.time()))
        print('')
        print('burgers_steady_viscous_test:')
        print('  FENICES/Python version')
        print('  Solve the steady 1d Burgers equation.')

        e_num = 16
        nu = 0.1
        bst = BurgerSteady(nu=nu, e_num=e_num)
        bst.define()
        bst.solve()

        print("")
        print("burgers_steady_viscous_test:")
        print("  Normal end of execution.")
        print('')
        print(time.ctime(time.time()))
        self.assertEqual(True, True)

    def test_poisson(self):
        print(time.ctime(time.time()))
        print('')
        print('poisson_test:')
        print('  Poisson equation on the unit square.')

        pois = Poisson(savfig=True)
        pois.define()
        pois.solve()

        print('')
        print('poisson_test:')
        print('  Normal end of execution.')
        print('')
        print(time.ctime(time.time()))
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
