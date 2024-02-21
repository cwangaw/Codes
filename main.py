from solver_jac import *
from utilities import *
from ngsolve import *
import matplotlib.pyplot as plt
import pickle
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
        
# set up for storing the arc length plots
bool_savearclength = True
if bool_savearclength == True:
    if not os.path.exists(outdir + '/arclength'):
        os.makedirs(outdir + '/arclength')

# ask user to input the parameter for refining the fractal structure
fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))

# set up the order of Lagrangian finite element
poly_deg = 5
    
# set up if we want to refine the mesh adaptively
is_adaptive = True

# set up the error tolerance for mesh refinement
tol = 1e-8

# set up the parameter for pde
d = 1

# set up max number of iteration
max_it = 40

# mesh generation
(mesh, ell_e, ell_p) = MakeGeometry(fractal_level)

with open(outdir+'/misc.txt', 'a') as misc:
    misc.write('Solving the Robin bc on a fractal boundary problem' + '\n')
    misc.write('\n')
    misc.write('pre-fractal parameters:' + '\n')
    misc.write('l: ' + str(ell_e) + '\n')
    misc.write('L_p: ' + str(ell_p) + '\n')
    
# initialize lists to store lam, total flux through the top edge, and d*ell_p/lam for each run
flux_top_lst = []
asymp_coeff_lst = []

# set up boundary conditions
bc = {"d": {"bottom": 1}, "n": {"right": 0, "left":0}, "r": {"top": 0}}

# lam from 0 to l
#lam_lst = np.array([ell_e*exp(n/2) for n in range(-10, 1)])

# lam from l to L_p
#factor = (ell_p/ell_e)**(0.1)
#lam_lst = np.array([ell_e*(factor**n) for n in range(0,10)])

# lam greater than L_p
#lam_lst = np.array([ell_p**n for n in range(1,11)])

length_inter = 40
factor = (ell_p/ell_e)**(1/length_inter)
lam_lst = np.array([ell_e*exp(n/4) for n in range(-28, 0)] + [ell_e*(factor**n) for n in range(length_inter)] + [ell_p**(n/4) for n in range(4,46)])
#lam_lst = [ell_p**(n/2) for n in range(11,25)]
#lam_lst = [ell_e*10**(-3), sqrt(ell_e*ell_p), ell_p*10**3]

use_uh_lst = [True for i in range(len(lam_lst))]
    
# solve the pde with lam = 0, 0.5*ell_p, ell_p, 1.5*ell_p, ..., 50*ell_p 
for i in range(len(lam_lst)):
    lam = lam_lst[i]
    if max_it > 0 and i > 0:
        (mesh, _, _) = MakeGeometry(fractal_level)
    
    if lam < ell_e:
        tol = 1e-9
    else:
        tol = 1e-8
    (uh, fes, flux_top, _, _, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, 0, is_adaptive, use_uh_lst[i], tol, max_it, mesh_it, outdir)
    
    if bool_savesolution==True:
        if i==0:
            vtkout = VTKOutput(mesh,coefs=[uh],names=["sol"],filename=savename,subdivision=2)
        vtkout.Do(time = i/(len(lam_lst)-1))
    
    flux_top_lst.append(flux_top)
    asymp_coeff_lst.append(d*ell_p/lam)
    
    if bool_savearclength == True:
        length_and_eval = []
        if fractal_level >= 1:
            l, p = fractal_verts(fractal_level)

            edge_id = -1

            length_lst = []
            eval_orig_lst = []
            eval_lst = []

            vert_lst = []

            # make a list of vertices of interest
            for el in fes.Elements(BND):
                v = el.vertices[1]
                x = mesh[v].point[0]
                y = mesh[v].point[1]
                if y < 1:
                    continue
                else:
                    vert_lst.append((x,y))
                    eval_orig_lst.append(uh.vec.data[fes.GetDofNrs(v)[0]])

            ind_lst = list(range(len(vert_lst)))
            vert_sort = [ [] for _ in range(len(p)-1) ]

            # iterate through top edges
            for top_edge_ind in range(len(p)-1):
                for ind in ind_lst[:]:
                    x = vert_lst[ind][0]
                    y = vert_lst[ind][1]
                    if abs(sqrt((x-p[top_edge_ind][0])**2+(y-p[top_edge_ind][1])**2) + sqrt((x-p[top_edge_ind+1][0])**2+(y-p[top_edge_ind+1][1])**2) - (1/3)**fractal_level) < 1e-10:
                        vert_sort[top_edge_ind].append(ind)
                        ind_lst.remove(ind)
                        

            for edge_id in range(len(p)-1):
                for ind in vert_sort[edge_id]:
                    length_lst.append(l[edge_id] - sqrt((vert_lst[ind][0]-p[edge_id][0])**2+(vert_lst[ind][1]-p[edge_id][1])**2))
                    eval_lst.append(eval_orig_lst[ind])
            
            for ind in range(len(length_lst)):
                length_and_eval.append((length_lst[ind],eval_lst[ind]))
            length_and_eval.sort()
        else:
            if i == 0:
                x_lst, pnts_lst = eval_pts(fractal_level, 10)

            val_lst = []
            iter = 0
            for p_ind in range(len(pnts_lst)):
                p = pnts_lst[p_ind]
                eval = uh(mesh(p[0],p[1]))
                length_and_eval.append((x_lst[p_ind],eval))
               
        
               
        # plot the arc length for the lambda
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim([0-1e-3, 1+1e-3])
        plt.xlabel("arc length")
        plt.ylabel("$u_h$")
        plt.plot(*zip(*length_and_eval), '-')
        plt.title("$\lambda$ = " + str(lam), style='italic')
        plt.savefig(outdir+"/arclength/"+str(i)+".pdf")
        plt.close()        
        
        with open(outdir+'/arclength/'+str(i)+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([lam])
            for ind in range(0,len(length_and_eval)):
                writer.writerow([length_and_eval[ind][0], length_and_eval[ind][1]]) #asymp_coeff_lst[i]])


# Draw uh
Draw(uh)

# plot the log scaled total flux through the top edge vs d*ell_p/lam
plt.figure()
plt.xlabel("$\Lambda$")
#plt.xlabel("$1/\Lambda$")
#plt.loglog([x for x in lam_lst[1:]], [1/flux-1/flux_top_lst[0] for flux in flux_top_lst[1:]], "*-", label="1/flux-1/flux_top_lst[0]")
plt.loglog([x/ell_p for x in lam_lst], flux_top_lst, "*-", label="Total flux through the top edge")
plt.loglog([x/ell_p for x in lam_lst], asymp_coeff_lst, "*-", color = 'r', label="$d*L_p/\Lambda$")
leg = plt.legend(loc='upper center')
plt.title('Log scaled, level of refinement = ' + str(fractal_level), style='italic')
plt.ion()
plt.show()
plt.savefig(outdir+"/flux_plot.pdf")


input("<Press enter to quit>")

# save results in a csv file
with open(outdir+'/results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Lambda", "Lambda/L_p", "flux Phi", "exact solution"])
    for i in range(0,len(lam_lst)):
        writer.writerow([lam_lst[i], lam_lst[i]/ell_p, flux_top_lst[i], 1/(lam_lst[i]+1)]) #asymp_coeff_lst[i]])