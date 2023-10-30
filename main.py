from solver import *
from utilities import *
from ngsolve import *
import matplotlib.pyplot as plt

'''
    The boundary value problem is - div (d grad u) = 0 on \Omega,
    with boundary conditions:
        on bottom edge: u = 1
        on left edge: d * du/dn = 0
        on right edge: d * du/dn = 0
        on top: lam * du/dn + u = 0

    The weak form is  a(u, v) = l(v), where
    if lam > 0, we have
        a(u, v) = (d * grad(u), grad(v)) + <d/lam * u, v>_("top")
        and
        l(v) = (f, v)
    if lam = 0, we have
        a(u, v) = (d * grad(u), grad(v))
        and
        l(v) = (f, v)
    Note: if lam = 0, the boundary condition on the top is Dirichlet, not Robin   
'''

# generate directory to save the results from this run
resultsdir='results'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
outdir = makedir(resultsdir)
mesh_it = 0

# set up parameters for saving the fem solution
bool_savesolution = False
savename = outdir + "/femsol/uh"
if bool_savesolution == True:
    if not os.path.exists(outdir + '/femsol'):
        os.makedirs(outdir + '/femsol')

# ask user to input the parameter for refining the fractal structure
fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))

# set up the order of Lagrangian finite element
poly_deg = 5

# set up if we want to refine the mesh adaptively
is_adaptive = True

# set up the error tolerance for mesh refinement
tol = 1e-5

# set up the parameter for pde
d = 1

# set up max number of iteration
max_it = 3

# mesh generation
(mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
with open(outdir+'/misc.txt', 'a') as misc:
    misc.write('Solving the Robin bc on a fractal boundary problem' + '\n')
    misc.write('\n')
    misc.write('pre-fractal parameters:' + '\n')
    misc.write('l: ' + str(ell_e) + '\n')
    misc.write('L_p: ' + str(ell_p) + '\n')
    
# initialize lists to store lam, total flux through the top edge, and d*ell_p/lam for each run
lam_lst = []
flux_top_lst = []
asymp_coeff_lst = []

# set up boundary conditions
bc = {"d": {"bottom": 1}, "n": {"right": 0, "left":0}, "r": {"top": 0}}

# total number of lambda's with which we are going to solve the pde
tot = int(ceil(ell_p/ell_e))

# solve the pde with lam = 0, 0.5*ell_p, ell_p, 1.5*ell_p, ..., 50*ell_p 
for i in range(tot+1):
    lam = i*(1/tot)*ell_p/20
    if max_it > 0:
        (mesh, _, _) = MakeGeometry(fractal_level)
    (uh, flux_top, _, _, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, 0, is_adaptive, tol, max_it, mesh_it, outdir)
    
    if bool_savesolution==True:
        if i==0:
            vtkout = VTKOutput(mesh,coefs=[uh],names=["sol"],filename=savename,subdivision=2)
        vtkout.Do(time = i/tot)
    
    lam_lst.append(lam)
    flux_top_lst.append(flux_top)
    if i > 0:
        asymp_coeff_lst.append(d*ell_p/lam)

# Draw uh
Draw(uh)

# plot the log scaled total flux through the top edge vs d*ell_p/lam
plt.xlabel("$\Lambda/L_p$")
plt.loglog([x/ell_p for x in lam_lst], flux_top_lst, "*-", label="Total flux through the top edge")
plt.loglog([x/ell_p for x in lam_lst[1:]], asymp_coeff_lst, "*-", color = 'r', label="$d*L_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()
plt.savefig(outdir+"/flux_plot.pdf")

input("<Press enter to quit>")

# save results in a csv file
with open(outdir+'/results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Lambda", "Lambda/L_p", "flux Phi", "asymptotic coefficient"])
    writer.writerow([lam_lst[0], lam_lst[0]/ell_p, flux_top_lst[0], "nan"])
    for i in range(1,len(lam_lst)):
        writer.writerow([lam_lst[i], lam_lst[i]/ell_p, flux_top_lst[i], asymp_coeff_lst[i-1]])