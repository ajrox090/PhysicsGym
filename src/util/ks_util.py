from phi import math


def ks_initial(x: math.Tensor):
    return math.cos(x) - 0.1 * math.cos(x / 16) * (1 - 2 * math.sin(x / 16))


def ks_initial2(x: math.Tensor):
    return math.sin(x) - 0.1 * math.sin(x / 16) * (1 - 2 * math.cos(x / 16))
