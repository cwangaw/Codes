from geogen import *
from solver import *
from ngsolve import *
import matplotlib.pyplot as plt

fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
d = 1
(mesh,l,L_p) = MakeGeometry(fractal_level)
(gfu, flux_top) = SolvePoisson(mesh, d, 0, fractal_level)

lam_lst = []
flux_top_lst = []
asymp_coeff_lst = []
for i in range(101):
    lam = i*0.5*L_p
    (gfu, flux_top) = SolvePoisson(mesh, d, lam, fractal_level)

    lam_lst.append(lam)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(d*L_p/lam)


plt.xlabel("$\Lambda$")
plt.loglog(lam_lst, flux_top_lst, "-", label="Total flux through the top edge")
plt.loglog(lam_lst[1:], asymp_coeff_lst, color = 'r', linestyle = '-', label="$dL_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()

input("<Press enter to quit>")