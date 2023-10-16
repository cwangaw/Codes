from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

# ask user to input the parameter for refining the fractal structure
fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))

# set up the order of Lagrangian finite element
poly_deg = 3

# set up if we want to refine the mesh adaptively
is_adaptive = True

# set up the error tolerance for mesh refinement
tol = 1e-5

# set up the parameter for pde
d = 1

# mesh generation
(mesh, ell_e, ell_p) = MakeGeometry(fractal_level)

# initialize lists to store lam, total flux through the top edge, and d*ell_p/lam for each run
lam_lst = []
flux_top_lst = []
asymp_coeff_lst = []

# solve the pde with lam = 0, 0.5*ell_p, ell_p, 1.5*ell_p, ..., 50*ell_p 
for i in range(101):
    lam = i*0.5*ell_p
    (gfu, flux_top) = SolvePoisson(mesh, poly_deg, d, lam, 0, 1, 0, 0, 0, is_adaptive, tol)

    lam_lst.append(lam)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(d*ell_p/lam)

# plot the log scaled total flux through the top edge vs d*ell_p/lam
plt.xlabel("$\Lambda$")
plt.loglog(lam_lst, flux_top_lst, "-", label="Total flux through the top edge")
plt.loglog(lam_lst[1:], asymp_coeff_lst, color = 'r', linestyle = '-', label="$d*ell_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()

input("<Press enter to quit>")