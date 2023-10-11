from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

# ask user to input the parameter for refining the fractal structure
fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))

# set up the order of Lagrangian finite element
poly_deg = 3

# set up the parameter for pde
D = 1

# mesh generation
(mesh,l,L_p) = MakeGeometry(fractal_level)

# initialize lists to store Lambda, total flux through the top edge, and D*L_p/Lambda for each run
Lambda_lst = []
flux_top_lst = []
asymp_coeff_lst = []

# solve the pde with Lambda = 0, 0.5*L_p, L_p, 1.5*L_p, ..., 50*L_p 
for i in range(101):
    Lambda = i*0.5*L_p
    (gfu, flux_top) = SolvePoisson(mesh, poly_deg, D, Lambda, 0, 1, 0, 0, 0)

    Lambda_lst.append(Lambda)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(D*L_p/Lambda)

# plot the log scaled total flux through the top edge vs D*L_p/Lambda
plt.xlabel("$\Lambda$")
plt.loglog(Lambda_lst, flux_top_lst, "-", label="Total flux through the top edge")
plt.loglog(Lambda_lst[1:], asymp_coeff_lst, color = 'r', linestyle = '-', label="$D*L_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()

input("<Press enter to quit>")