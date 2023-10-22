from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

# ask user to input the parameter for refining the fractal structure
fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))

# set up the order of Lagrangian finite element
poly_deg = 3

# set up if we want to refine the mesh adaptively
is_adaptive = False

# set up the error tolerance for mesh refinement
tol = 1e-5

# set up the parameter for pde
d = 1

# set up max number of iteration
max_it = 2

# mesh generation
(mesh, ell_e, ell_p) = MakeGeometry(fractal_level)

# initialize lists to store lam, total flux through the top edge, and d*ell_p/lam for each run
lam_lst = []
flux_top_lst = []
asymp_coeff_lst = []

# set up boundary conditions
bc = {"d": {"bottom": 1}, "n": {"right": 0, "left":0}, "r": {"top": 0}}

# solve the pde with lam = 0, 0.5*ell_p, ell_p, 1.5*ell_p, ..., 50*ell_p 
for i in range(101):
    lam = i*0.5*ell_p
    if max_it > 0:
        (mesh, _, _) = MakeGeometry(fractal_level)
    (uh, flux_top, _, _) = SolvePoisson(mesh, bc, poly_deg, d, lam, 0, is_adaptive, tol, max_it)

    lam_lst.append(lam)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(d*ell_p/lam)

# plot the log scaled total flux through the top edge vs d*ell_p/lam
plt.xlabel("$\Lambda$")
plt.loglog(lam_lst, flux_top_lst, "-", label="Total flux through the top edge")
plt.loglog(lam_lst[1:], asymp_coeff_lst, color = 'r', linestyle = '-', label="$d*L_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()

input("<Press enter to quit>")