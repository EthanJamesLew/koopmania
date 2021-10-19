import sympy as sp
import koopmania as km


class VanDerPol(km.SymbolicContinuousSystem):
    def __init__(self):
        x0, x1 = sp.symbols('x0 x1')
        xdot = [x1, (1 - x0**2) * x1 - x0]
        super(VanDerPol, self).__init__((x0, x1), xdot)


class CoupledVanDerPol(km.SymbolicContinuousSystem):
    """https://ths.rwth-aachen.de/research/projects/hypro/coupled-van-der-pol-oscillator/"""
    def __init__(self):
        x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2')
        dx1 = y1
        dy1 = (1 - (x1**2))*y1 - x1 + (x2 - x1)
        dx2 = y2
        dy2 = (1 - (x2**2))*y2 - x2 + (x1 - x2)
        xdot = [dx1, dy1, dx2, dy2]
        super(CoupledVanDerPol, self).__init__((x1, y1, x2, y2), xdot)
