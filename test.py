from geogen import *
from manuSolver import *
from ngsolve import *

from geogen import *
from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

fractal_level = 0
d = 1
(mesh,l,L_p) = MakeGeometry(fractal_level)
(gfu, flux_top, e) = SolveManuPoisson(mesh, d, L_p)

print("error is", e)