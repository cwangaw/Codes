from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
d = 1
poly_deg = 3
(mesh,l,L_p) = MakeGeometry(fractal_level)

Lambda_lst = []
flux_top_lst = []
asymp_coeff_lst = []
for i in range(101):
    Lambda = i*0.5*L_p
    (gfu, flux_top) = SolvePoisson(mesh, poly_deg, d, Lambda, 0, 1, 0, 0, 0)

    Lambda_lst.append(Lambda)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(d*L_p/Lambda)


plt.xlabel("$\Lambda$")
plt.loglog(Lambda_lst, flux_top_lst, "-", label="Total flux through the top edge")
plt.loglog(Lambda_lst[1:], asymp_coeff_lst, color = 'r', linestyle = '-', label="$dL_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()

input("<Press enter to quit>")