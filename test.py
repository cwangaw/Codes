from geogen import *
from manuSolver import *
from ngsolve import *

fractal_level = 4
d = 1
(mesh,l,L_p) = MakeGeometry(fractal_level)
(gfu, flux_top, e) = SolveManuPoisson(mesh, d, L_p)

print("Error:", e)