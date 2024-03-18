import netgen.gui
import matplotlib.pyplot as plt
import os, sys
import csv
import datetime

from utilities import *
from netgen.csg import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingStep
from netgen.occ import *
from netgen.webgui import Draw as DrawGeo

from ngsolve import *
from netgen.occ import unit_square
import numpy as np

from geogen import *

# set up the number of threads to use with TaskManager()
#SetNumThreads(32)

# set up the msg print level
ngsglobals.msg_level=0

def SolvePoisson(mesh, bc, deg=1, d=1, lam=1, f=0, bool_adaptive = False, use_uh = True, tol = 1e-5, max_it = 50, mesh_it = 0, outdir = 'results'):
    '''
    Input:
        mesh:           ngsolve mesh of the unit square without requiring the top edge to be straight
        bc:             a 3-item nested dictionary { "d": {}, "n":{}, "r":{} }
                        corresponding to dirichlet, neumann, and robin boundary conditions
                        each pair in the nested dictionary has key: the label of the edge, 
                                                               value: the coefficient function (data)
        deg:            degree of Lagrange finite element space
        d:              a positive constant, or eventually function (conductivity)
        lam:            non-negative constant from Robin boundary condition
        f:              forcing term
        bool_adaptive:  whether the mesh will be refined adaptively or not
        tol:            if sqrt(sum_eta2) <= tol, the result is accepted
        max_it:         when the repetition of refine-solve procedure reaches max_it, 
                            the result is accepted regardless of the error estimate
        mesh_it:        current index for mesh storage
        
    Returns:
        uh:             finite element solution
        flux_top:       the flux through the top of uh
        runtimes:       a list with the runtime after each refinement (after the initial run only) and solving
        errs:           a list of 2-tuple (ndof, sqrt(sum_eta2)) for each solve, the first entry is the number of dofs,
                            the second entry is an approximation of H1-seminorm error
        mesh_it:        updated index for mesh storage
        
    The boundary value problem is - div (d grad u) = f on \Omega,
        with boundary conditions given by the nested dictionary bc:
            Dirichlet: u = g
            Neumann: d * du/dn = g
            Robin: lam * du/dn + u = g
    
    The weak form is  a(u, v) = l(v), where
    if lam > 0, we have
        a(u, v) = (d * grad(u), grad(v)) + <d/lam * u, v>_(robin)
        and
        l(v) = (f, v) + <g, v>_(neumann) + <d/lam)*g_t, v>_(robin)
    if lam = 0, we have
        a(u, v) = (d * grad(u), grad(v))
        and
        l(v) = (f, v) + <g, v>_(neumann)
    Note: if lam = 0, the boundary condition on the top is Dirichlet, not Robin    
    '''
    
    # output all the initial parameters
    with open(outdir+'/misc.txt', 'a') as misc:
        misc.write("\n")
        misc.write("*** SolvePoisson() ***" + "\n")
        misc.write("boundary conditions: " + str(bc) + "\n")
        misc.write("fem space order: " + str(deg) + "\n")
        misc.write("d = " + str(d) + "\n")
        misc.write("lam = " + str(lam) + "\n")
        misc.write("f = " + str(f) + "\n")
        misc.write("adaptivity: " + str(bool_adaptive) + "\n")
        misc.write("tolerance = " + str(tol) + "\n")
        misc.write("max refinement iteration = " + str(max_it) + "\n")
        
    has_robin = bool(bc["r"])
    has_dirichlet = bool(bc["d"])
    
    if has_robin:
        robin_str = '|'.join(map(str,list(bc["r"].keys())))
    if has_dirichlet:
        dirichlet_str = '|'.join(map(str,list(bc["d"].keys())))
        
    # initialize the finite element
    # clarify the dirichlet boundary conditions according to if lam is positive or zero
    if lam > 0:
        if has_dirichlet:
            fes = H1(mesh, order=deg, dirichlet=dirichlet_str, autoupdate=True)
    elif has_dirichlet or has_robin:
        if has_dirichlet and has_robin:
            new_dirichlet_str = dirichlet_str + "|" + robin_str
        elif has_dirichlet:
            new_dirichlet_str = dirichlet_str
        elif has_robin:
            new_dirichlet_str = robin_str
        fes = H1(mesh, order=deg, dirichlet=new_dirichlet_str, autoupdate=True)
    
    u = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()   # symbolic object

    # intialize and define the bilinear form a(u,v)
    a = BilinearForm(fes)
    a += d*grad(u)*grad(v)*dx
    if lam > 0 and has_robin:
        a += (d/lam)*u*v*ds(robin_str)

    # intialize and define the linear form l(v)
    l = LinearForm(fes)
    l += f*v*dx
    # add the neumann boundary terms
    for label in bc["n"].keys():
        l += bc["n"][label]*v*ds(label)  
    # add the robin boundary terms if lam > 0
    if lam > 0:
        for label in bc["r"].keys():
            l += (d/lam)*bc["r"][label]*v*ds(label)

    # solution
    uh = GridFunction(fes, autoupdate=True)  
    #c = Preconditioner(a, "bddc") # Register c to a BEFORE assembly
    #c = MultiGridPreconditioner(a, inverse = "sparsecholesky")
    # save current mesh
    outmeshdir = outdir+"/mesh"
    if not os.path.exists(outmeshdir):
        os.makedirs(outmeshdir)

    def SaveMesh(mesh_it):
        meshname = "mesh" + str(mesh_it) + ".vol"
        mesh.ngmesh.Save(outmeshdir + "/" + meshname)
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("saving mesh: " + meshname + "\n")
            misc.write("\tnumber of elements: " + str(mesh.ne) + "\n")
            misc.write("\tnumber of dofs:" + str(fes.ndof) + "\n")
        mesh_it = mesh_it + 1
        return mesh_it
    
    # solve for the free dofs
    def SolveBVP():
        # set up boundary conditions
        if lam > 0:
            if has_dirichlet:
                uh.Set(mesh.BoundaryCF(bc["d"]), BND)
        elif has_dirichlet or has_robin:
            new_dirichlet_dict = {**bc["d"], **bc["r"]}
            uh.Set(mesh.BoundaryCF(new_dirichlet_dict), BND)
        
        # assemble the bilinear and the linear form
        a.Assemble()
        l.Assemble()
        #jac = a.mat.CreateSmoother(fes.FreeDofs())
        r = l.vec - a.mat * uh.vec
        
        #inv = CGSolver(a.mat, c.mat, maxsteps=3000, printrates=False)   
        #inv = CGSolver(adev, predev, maxsteps=2000, printrates=False)
        
        #uh.vec.data += inv * r
        uh.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * r
           
    errs = []
    runtimes = []
    
    # finite element space and gridfunction to represent
    # the heatflux:
    space_flux = HDiv(mesh, order=deg-1, autoupdate=True)
    gf_flux = GridFunction(space_flux, "flux", autoupdate=True)
    
    def CalcError():
        flux = d * grad(uh)
        # interpolate finite element flux into H(div) space:
        gf_flux.Set (flux)

        # gradient-recovery error estimator
        err = 1/d*(flux-gf_flux)*(flux-gf_flux)
        eta2 = Integrate (err, mesh, VOL, element_wise=True)
        eta2 = eta2.NumPy()
        # Integrate ( *dx(element_boundary=True), mesh, VOL, element_wise=True)

        max_eta2 = max(eta2)
        sum_eta2 = sum(eta2)
        errs.append((fes.ndof, sqrt(sum_eta2)))
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("H1 error estimator: " + str(sqrt(sum_eta2)) + "\n")
        # if we want to refine the mesh adaptively,
        # we label the elements in need of refinements here
        if bool_adaptive and it < max_it:
            # *** MARK step
            # Mark cells for refinement for which eta > frac eta_max for frac = .95, .90, ...;
            # choose frac so that marked elements account for a given part of total error
   
            frac = .95
            delfrac = .05
            part = .5
            marked = np.zeros(mesh.ne, dtype=bool) # marked starts as False for all elements
            sum_marked_eta2 = 0. # sum over marked elements of squared error indicators

            while sum_marked_eta2 < part*sum_eta2:
                new_marked = (~marked) & (eta2 > frac*max_eta2)
                sum_marked_eta2 += sum(eta2[new_marked])
                marked += new_marked
                frac -= delfrac
               
            for el in mesh.Elements():
                mesh.SetRefinementFlag(el, marked[el.nr]) 
            
            if mesh.dim == 3:
                for el in mesh.Elements(BND):
                    mesh.SetRefinementFlag(el, False)
                
        return sqrt(sum_eta2)

    # start the timer
    start = datetime.datetime.now()

    # refine the mesh and solve the equation until the H1 error is within the tolerance
    # we note down the time after each time we run SolveBVP()
    with TaskManager():
        #mesh_it = SaveMesh(mesh_it)
        SolveBVP()
        time_delta = datetime.datetime.now() - start
        runtimes.append(time_delta.total_seconds())
        it = 0
        while CalcError() > tol and it < max_it:
            mesh.Refine()
            #mesh_it = SaveMesh(mesh_it)
            SolveBVP()
            time_delta = datetime.datetime.now() - start
            runtimes.append(time_delta.total_seconds())
            it += 1
            if it == max_it:
                warningmsg = "Number of iterations reaches max_it " + str(max_it) + " before the H1 error estimator reaching tolerance " + str(tol) + "\n"
                print(warningmsg)
                with open(outdir+'/misc.txt', 'a') as misc:
                    misc.write(warningmsg)
    
    if has_robin:
        if use_uh == True and lam > 0:
            # the flux through the robin boundaries equals to <(d/lam)*u>_("r")
            flux_top = 0
            for label in bc["r"].keys():
                flux_top += Integrate((d/lam)*(-bc["r"][label]+uh), mesh, BND, definedon=mesh.Boundaries(label))
        else:
            # the flux through the old robin boundaries is <-d*du/dn>_(old_robin)
            n = specialcf.normal(mesh.dim)
            if mesh.dim == 2:
                gradu0 = GridFunction(fes)
                gradu1 = GridFunction(fes)
                gradu0.Set(grad(uh)[0])
                gradu1.Set(grad(uh)[1])
                flux_top = Integrate(-d*(gradu0*n[0]+gradu1*n[1]), mesh, BND, definedon=mesh.Boundaries(robin_str))
            else:
                gradu0 = GridFunction(fes)
                gradu1 = GridFunction(fes)
                gradu2 = GridFunction(fes)
                gradu0.Set(grad(uh)[0])
                gradu1.Set(grad(uh)[1])
                gradu2.Set(grad(uh)[2])
                flux_top = Integrate(-d*(gradu0*n[0]+gradu1*n[1]+gradu2*n[2]), mesh, BND, definedon=mesh.Boundaries(robin_str))                
    else:
        flux_top = 0
    
    with open(outdir+'/misc.txt', 'a') as misc:
        misc.write('flux through the robin boundaries: ' + str(flux_top) + '\n')
    return uh, fes, flux_top, runtimes, errs, mesh_it

# generate points on the top boundary with the distance of 
# consecutive points as ell_e/nints_per_seg
def eval_pts(fractal_level, nints_per_seg):
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    sgmnts = []
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = GrowKochCurve(2,3,pnts,sgmnts,fractal_level)

    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1 / (3.0**fractal_level)
    
    length_lst = [0]
    pnts_lst = [(0,1)]
    
    current_length = 0
    
    for v in reversed(sgmnts):
        for i in range(nints_per_seg):
            current_length += ell_e / nints_per_seg
            length_lst.append(current_length)
            x_new = pnts_lst[-1][0] + (pnts[v[0]][0]-pnts[v[1]][0])/nints_per_seg
            y_new = pnts_lst[-1][1] + (pnts[v[0]][1]-pnts[v[1]][1])/nints_per_seg
            pnts_lst.append((x_new,y_new))
    
    return length_lst, pnts_lst


def fractal_verts(fractal_level):
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    sgmnts = []
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = GrowKochCurve(2,3,pnts,sgmnts,fractal_level)

    ell_e = 1 / (3.0**fractal_level)
    
    #counterclockwise
    pnts_lst = [(1,1)]
    for sgmnt in sgmnts:
        pnts_lst.append(pnts[sgmnt[1]])
    length_lst =[ell_e*n for n in range(4**fractal_level,-1,-1)]
    
    return length_lst, pnts_lst


    
if __name__ == "__main__":
    # generate directory to save the results from this run
    resultsdir='results'
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    outdir = makedir(resultsdir)
    outmeshdir = outdir+"/mesh"
    mesh_it = 0
    os.makedirs(outmeshdir)
    
    bool_savesolution = False
    savename = outdir + "/femsol/uh"
    if bool_savesolution== True:
        if not os.path.exists(outdir + '/femsol'):
            os.makedirs(outdir + '/femsol')
    
    # set up the order of Lagrangian finite element
    poly_deg = 5
    
    # set up the desired tolerance for the parameter eta in mesh refinment
    tol = 1e-7
    
    use_uh = True
    
    if len(sys.argv) > 1 and sys.argv[1] == "singular":
        # write down the test
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("Testing singular solution on a L-shape domain" + "\n")
            
        # set up the parameters
        d = 1
        lam = 1

        # set up the source function and the boundary conditions
        f = 1
        bc = {"d": {"dirichlet": 0}, "n": {}, "r": {}}
        
        errs_lst = []
        runtimes_lst = []
        
        is_adaptive = [False, True]
        max_it = [5, 10]
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            mesh = MakeGeometry("lshape")
            
            # solve the pde
            (uh, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], use_uh, tol, max_it[i], mesh_it, outdir)
            if bool_savesolution == True:
                savesolution(mesh, uh, savename+str(int(is_adaptive[i])))
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
    elif len(sys.argv) > 1 and sys.argv[1] == "lam":
        # write down the test
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("Testing the Robin bc on a fractal boundary problem with lam = " + str(sys.argv[2]) + "\n")
            
        d = 1
        fractal_level = 5
        f = 0
        lam = float(sys.argv[2])
        
        (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
        
        # set up boundary conditions
        bc = {"d": {"bottom": 1}, "n": {"right": 0, "left": 0}, "r": {"top": 0}}
        
        errs_lst = []
        runtimes_lst = []
        is_adaptive = [False, True]
        max_it = [0, 10]
        use_uh = True
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            if i > 0:
                (mesh, _, _) = MakeGeometry(fractal_level)
            (uh, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], use_uh, tol, max_it[i], mesh_it, outdir)
            if bool_savesolution == True:
                savesolution(mesh, uh, savename+str(int(is_adaptive[i])))
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
    elif len(sys.argv) > 1 and sys.argv[1] == "3dlam":
        # write down the test
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("Testing the Robin bc on a fractal boundary problem with lam = " + str(sys.argv[2]) + "\n")
            
        d = 1
        fractal_level = 3
        f = 0
        lam = float(sys.argv[2])
        
        (mesh, ell_e, ell_p) = MakeMidKochSurfaceGeo(fractal_level)
        
        # set up boundary conditions
        bc = {"d": {"bottom": 1}, "n": {"side": 0}, "r": {"top": 0}}
        
        errs_lst = []
        runtimes_lst = []
        is_adaptive = [False, True]
        max_it = [0, 5]
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            (mesh, _, _) = MakeMidKochSurfaceGeo(fractal_level)
            (uh, _, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], True, tol, max_it[i], mesh_it, outdir)
            #if bool_savesolution == True:
            #    savesolution(mesh, uh, savename+str(int(is_adaptive[i])))
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
                    
    elif len(sys.argv) > 1 and sys.argv[1] == "3d":
        # write down the test
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("Testing the Robin bc on a fractal boundary problem with a manufactured solution" + "\n")

        # set up the parameter for refining the fractal structure
        fractal_level = 1
        
        # mesh generation
        (mesh, ell_e, ell_p) = MakeGeometry("3d14quad",fractal_level)
        # set up parameters for the pde
        d = 1
        lam = 1e-4

        # set up manufactured solution
        # and the corresponding source term and boundary data
        manu_sol = ((x-2/3)**2+(y-2/3)**2+(z-1)**2)**(1/3.)
        f = -((d*manu_sol.Diff(x)).Diff(x) + (d*manu_sol.Diff(y)).Diff(y) + (d*manu_sol.Diff(z)).Diff(z))
        n = specialcf.normal(mesh.dim)
        du_nu = manu_sol.Diff(x)*n[0]+manu_sol.Diff(y)*n[1]+manu_sol.Diff(z)*n[2]
        g_b = manu_sol
        g_s = d*du_nu
        g_t = lam*du_nu + manu_sol
        bc = {"d": {"bottom": g_b}, "n": {"side": g_s}, "r": {"top": g_t}}
        # Compare the running time and number of dofs for 
        # traditionally and adaptively refined mesh
        result = open(outdir+"/comparison.txt", "a+")
        
        errs_lst = []
        runtimes_lst = []
        is_adaptive = [False, True]
        max_it = [3, 5]
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            (mesh, _, _) = MakeGeometry("3d14quad",fractal_level)
            (uh, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], tol, max_it[i], mesh_it, outdir)
            if bool_savesolution == True:
                savesolution(mesh, uh, savename+str(int(is_adaptive[i])))
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
            # calculate and print the L2 error
            e = Integrate((uh-manu_sol)**2, mesh, VOL)
            e = sqrt(e)
            
            # the flux through the robin boundaries equals to <(d/lam)*u>_("r")
            real_flux_top = Integrate((d/lam)*(-g_t+manu_sol), mesh, BND, definedon=mesh.Boundaries("top"))
            real_flux_top_2 = Integrate(-d*du_nu, mesh, BND, definedon=mesh.Boundaries("top"))

            # append the results into the file
            result.write("Adaptivity: " + str(is_adaptive[i]) + ", Solving time: " + str(runtimes[-1]) + ", nDoFs:" + str(errs[-1][0]) + ", Error:" + str(e)  + ", uh flux top:" + str(real_flux_top) + ", duh flux top 2:" + str(real_flux_top_2) + '\n')
 
        result.close()       
    else:
        # write down the test
        with open(outdir+'/misc.txt', 'a') as misc:
            misc.write("Testing the Robin bc on a fractal boundary problem with a manufactured solution" + "\n")

        # set up the parameter for refining the fractal structure
        fractal_level = 1
        
        # mesh generation
        (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
        # set up parameters for the pde
        d = 1
        lam = 1e-5

        # set up manufactured solution
        # and the corresponding source term and boundary data
        manu_sol = ((x-2/3)**2+(y-1)**2)**(1/3.)*sin(2./3.*atan2(y-1,-(x-2/3)))
        f = -((d*manu_sol.Diff(x)).Diff(x) + (d*manu_sol.Diff(y)).Diff(y))
        n = specialcf.normal(mesh.dim)
        du_nu = manu_sol.Diff(x)*n[0]+manu_sol.Diff(y)*n[1]
        g_b = manu_sol
        g_l = d*du_nu
        g_r = d*du_nu
        g_t = lam*du_nu + manu_sol
        bc = {"d": {"bottom": g_b}, "n": {"right": g_r, "left": g_l}, "r": {"top": g_t}}
        # Compare the running time and number of dofs for 
        # traditionally and adaptively refined mesh
        result = open(outdir+"/comparison.txt", "a+")
        
        errs_lst = []
        runtimes_lst = []
        is_adaptive = [False, True]
        max_it = [5, 100]
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            (mesh, _, _) = MakeGeometry(fractal_level)
            (uh, flux, runtimes, errs, mesh_it) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], use_uh, tol, max_it[i], mesh_it, outdir)
            #if bool_savesolution == True:
            #    savesolution(mesh, uh, savename+str(int(is_adaptive[i])))
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
            # calculate and print the L2 error
            e = Integrate((uh-manu_sol)**2, mesh, VOL)
            e = sqrt(e)
            
            # the flux through the robin boundaries equals to <(d/lam)*u>_("r")
            real_flux_top = Integrate((d/lam)*(-g_t+manu_sol), mesh, BND, definedon=mesh.Boundaries("top"))
            real_flux_top_2 = Integrate(-d*du_nu, mesh, BND, definedon=mesh.Boundaries("top"))

            # append the results into the file
            result.write("Adaptivity: " + str(is_adaptive[i]) + ", Solving time: " + str(runtimes[-1]) + ", nDoFs:" + str(errs[-1][0]) + ", Error:" + str(e)  + ", uh flux top:" + str(real_flux_top) + ", duh flux top 2:" + str(real_flux_top_2) + '\n')
    
        result.close()
    
    # plot the uh
    Draw(uh)
    

    #set up plot title corresponding to the problem solved
    if len(sys.argv) > 1 and sys.argv[1] == "singular":
        plot_title = "L-shaped domain"
    elif len(sys.argv) > 1 and sys.argv[1] == "lam":
        plot_title = "robin on fractal top boundary, lam = " + sys.argv[2]
    else:
        plot_title = "manufactured solution"
        
    # plot the H1 error estimate vs the number of DoFs
    plt.figure()
    plt.xlabel("ndof")
    plt.ylabel("H1 error-estimate")
    ndof_n, err_n = zip(*(errs_lst[0]))
    ndof_a, err_a = zip(*(errs_lst[1]))
    plt.loglog(ndof_n,err_n, '*-', label="not adaptive")
    new_start = plot_coef(ndof_n,err_n,[min(ndof_n), min(err_n)])
    plt.loglog(ndof_a,err_a, "*-", label="adaptive")
    plot_coef(ndof_a,err_a,new_start)
    leg = plt.legend(loc='upper right')
    plt.title(plot_title, style='italic')
    plt.savefig(outdir+"/err_ndofs.pdf")
    
    # plot the H1 error estimate vs runtime
    plt.figure()
    plt.xlabel("runtime")
    plt.ylabel("H1 error-estimate")
    plt.loglog(runtimes_lst[0],err_n, '*-', label="not adaptive")
    new_start = plot_coef(runtimes_lst[0],err_n, [min(runtimes_lst[0]),min(err_n)])
    plt.loglog(runtimes_lst[1],err_a, "*-", label="adaptive")
    plot_coef(runtimes_lst[1],err_a,new_start)
    leg = plt.legend(loc='upper right')
    plt.title(plot_title, style='italic')
    plt.savefig(outdir+"/err_runtimes.pdf")
    
    # save results in a csv file
    with open(outdir+'/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["H1 error estimate", "nDoFs", "runtime"])
        writer.writerow([])
        writer.writerow(["uniformly refined mesh"])
        for i in range(len(ndof_n)):
            writer.writerow([err_n[i], ndof_n[i], runtimes_lst[0][i]])
        writer.writerow([])
        writer.writerow(["adaptively refined mesh"])
        for i in range(len(ndof_a)):
            writer.writerow([err_a[i], ndof_a[i], runtimes_lst[1][i]])
        

    