from solver import *
from utilities import *
from ngsolve import *

mesh,l,L_p = MakeCSGeometry(3)
Draw(mesh)

# generate directory to save the results from this run
resultsdir='results'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
outdir = makedir(resultsdir)
mesh_it = 0

# set up boundary conditions
bc = {"d": {"bottom": 1}, "n": {"side": 0}, "r": {"top": 0}}
(uh, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, 5, 1, 20*L_p, 0, True, 1e-8, 4, mesh_it, outdir)

Draw(uh)