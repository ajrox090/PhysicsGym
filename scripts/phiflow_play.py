from phi.flow import *


def main():
    domain = dict(x=100, y=100, bounds=Box(x=1, y=1), extrapolation=extrapolation.BOUNDARY)
    grid = StaggeredGrid(SoftGeometryMask(Sphere([0, 0], radius=1)), **domain)  # with anti-aliasing
    view(grid, gui='console')


if __name__ == "__main__":
    main()
