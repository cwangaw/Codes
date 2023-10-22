import netgen.gui
import matplotlib.pyplot as plt
import timeit
import contextlib, io, sys

from ngsolve import *
from numpy import sin,cos,pi,zeros,sqrt,arctan2
#from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry

def SolvePoisson(mesh, bc, deg=1, d=1, lam=1, f=0, bool_adaptive = False, tol = 1e-5, max_it = 50):
    '''
    Input:
        mesh: ngsolve mesh of the unit square without requiring the top edge to be straight
        d:    a positive constant, or eventually function (conductivity)
        lam:  non-negative constant from Robin boundary condition
        f:    forcing term
        bc:   a 3-item nested dictionary { "d": {}, "n":{}, "r":{} }
                corresponding to dirichlet, neumann, and robin boundary conditions
                each pair in the nested dictionary has key: the label of the edge, 
                                                       value: the coefficient function (data)
        deg:  degree of Lagrange finite element space
    Returns:
        uh:   finite element solution
        flux_top: the flux through the top of uh
    
    The boundary value problem is - div (d grad u) = f on \Omega,
    with boundary conditions:
        on bottom edge: u = g_b
        on left edge: d * du/dn = g_l 
        on right edge: d * du/dn = g_r
        on top: lam * du/dn + u = g_t
    
    The weak form is  a(u, v) = l(v), where
    if lam > 0, we have
        a(u, v) = (d * grad(u), grad(v)) + <d/lam * u, v>_("top")
        and
        l(v) = (f, v) + <g_l, v>_("left") + <g_r, v>_("right") + <d/lam)*g_t, v>_("top")
    if lam = 0, we have
        a(u, v) = (d * grad(u), grad(v))
        and
        l(v) = (f, v) + <g_l, v>_("left") + <g_r, v>_("right")
    Note: if lam = 0, the boundary condition on the top is dirichlet, not Robin    
    '''
    
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
    
    # solve for the free dofs
    def SolveBVP():
        # assemble the bilinear and the linear form
        a.Assemble()
        l.Assemble()
        
        # set up boundary conditions
        if lam > 0:
            if has_dirichlet:
                uh.Set(mesh.BoundaryCF(bc["d"]), BND)
        elif has_dirichlet or has_robin:
            new_dirichlet_dict = {**bc["d"], **bc["r"]}
            uh.Set(mesh.BoundaryCF(new_dirichlet_dict), BND)
        
        # solve the linear system    
        r = l.vec - a.mat * uh.vec
        uh.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
           
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

        # if we want to refine the mesh adaptively,
        # we label the elements in need of refinements here
        if bool_adaptive:
            # *** MARK step
            # Mark cells for refinement for which eta > frac eta_max for frac = .95, .90, ...;
            # choose frac so that marked elements account for a given part of total error
            frac = .95
            delfrac = .05
            part = .5
            marked = zeros(mesh.ne, dtype=bool) # marked starts as False for all elements
            sum_marked_eta2 = 0. # sum over marked elements of squared error indicators

            while sum_marked_eta2 < part*sum_eta2:
                new_marked = (~marked) & (eta2 > frac*max_eta2)
                sum_marked_eta2 += sum(eta2[new_marked])
                marked += new_marked
                frac -= delfrac

            for el in mesh.Elements():
                 mesh.SetRefinementFlag(el, marked[el.nr])
        
        return sqrt(sum_eta2)

    # start the timer
    start = timeit.default_timer()

    # refine the mesh and solve the equation until the H1 error is within the tolerance
    # we note down the time after each time we run SolveBVP()
    with TaskManager():
        SolveBVP()
        runtimes.append(timeit.default_timer() - start)
        it = 0
        while CalcError() > tol and it < max_it:
            mesh.Refine()
            SolveBVP()
            runtimes.append(timeit.default_timer() - start)
            it += 1
        
    if has_robin:
        if lam > 0:
            # the flux through the robin boundaries equals to <(d/lam)*u>_("r")
            flux_top = Integrate((d/lam)*uh, mesh, BND, definedon=mesh.Boundaries(robin_str))
        else:
            # the flux through the old robin boundaries is <-d*du/dn>_(old_robin)
            n = specialcf.normal(mesh.dim)
            gradu0 = GridFunction(fes)
            gradu1 = GridFunction(fes)
            gradu0.Set(grad(uh)[0])
            gradu1.Set(grad(uh)[1])
            flux_top = Integrate(-d*(gradu0*n[0]+gradu1*n[1]), mesh, BND, definedon=mesh.Boundaries(robin_str))
    else:
        flux_top = 0
        
    return uh, flux_top, runtimes, errs

# update the pnts list and sgmnts list
def FractalStructure(p_start, p_end, pnts, sgmnts, current_level):
    num_pts = len(pnts)

    # if we have not yet reached the top level, add new points and new segments,
    # otherwise we only add the segment connecting p_start and p_end
    if (current_level > 0):
        # define the 3 new points added to current points
        p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (1/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (1/3) )
        p2 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (2/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (2/3) )
        p1 = ( p0[0] + cos(-pi/3) * (p2[0] - p0[0]) - sin(-pi/3) * (p2[1] - p0[1]), p0[1] + sin(-pi/3) * (p2[0] - p0[0]) + cos(-pi/3) * (p2[1] - p0[1]))
        
        pnts = pnts + [p0, p1, p2]
        (pnts, sgmnts) = FractalStructure(p_start,num_pts,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = FractalStructure(num_pts,num_pts+1,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = FractalStructure(num_pts+1,num_pts+2,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = FractalStructure(num_pts+2,p_end,pnts,sgmnts,current_level-1)
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
        
    return pnts, sgmnts
    
# make mesh of fractal domain
def MakeGeometry(fractal_level, h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    # the bottom, right, and left edges
    sgmnts = [(0,1), (1,2), (3,0)]
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = FractalStructure(2,3,pnts,sgmnts,fractal_level)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc="bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc="right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc="left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc="top")
    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1 / (3.0**fractal_level)
    ell_p = (4.0/3.0)**fractal_level
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh, ell_e, ell_p

def MakeLGeometry(h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [ (0,0), (1,0), (1,1), (-1,1), (-1,-1), (0,-1) ]
    
    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])
        if i > 0:
            geo.Append (["line", i-1, i], bc="dirichlet")
    
    geo.Append (["line", len(pnts)-1, 0], bc="dirichlet")
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh
    
if __name__ == "__main__":
    # set up the order of Lagrangian finite element
    poly_deg = 2
    
    # set up the desired tolerance for the parameter eta in mesh refinment
    tol = 1e-5
    
    if len(sys.argv) > 1 and sys.argv[1] == "singular":
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
            mesh = MakeLGeometry()
            
            # solve the pde
            (uh, flux, runtimes, errs) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], tol, max_it[i])
            
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
    elif len(sys.argv) > 1 and sys.argv[1] == "lam":
        d = 1
        fractal_level = 3
        f = 0
        lam = float(sys.argv[2])
        
        (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
        
        # set up boundary conditions
        bc = {"d": {"bottom": 1}, "n": {"right": 0, "left": 0}, "r": {"top": 0}}
        
        errs_lst = []
        runtimes_lst = []
        is_adaptive = [False, True]
        max_it = [5, 10]
        for i in range(2):
            # initialize a new mesh, on which we solve the pde
            (mesh, _, _) = MakeGeometry(fractal_level)
            (uh, flux, runtimes, errs) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive[i], tol, max_it[i])

            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
    else:
        # set up the parameter for refining the fractal structure
        fractal_level = 3
        
        # mesh generation
        (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
        # set up parameters for the pde
        d = 2.56
        lam = ell_p

        # set up manufactured solution
        # and the corresponding source term and boundary data
        manu_sol = x**3 * y
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
        result = open("results/comparison.txt", "a")
        
        errs_lst = []
        runtimes_lst = []
        for is_adaptive in [False, True]:
            # initialize a new mesh, on which we solve the pde
            (mesh, _, _) = MakeGeometry(fractal_level)
            (uh, flux, runtimes, errs) = SolvePoisson(mesh, bc, poly_deg, d, lam, f, is_adaptive, tol)
            errs_lst.append(errs)
            runtimes_lst.append(runtimes)
            
            # calculate and print the L2 error
            e = Integrate((uh-manu_sol)**2, mesh, VOL)
            e = sqrt(e)
            
            # append the results into the file
            result.write("Adaptivity: " + str(is_adaptive) + ", Solving time: " + str(runtimes[-1]) + ", nDoFs:" + str(errs[-1][0]) + ", Error:" + str(e) + '\n')
            
        result.write('\n')
        result.close()
    
    # plot the uh
    Draw(uh)
    
    # plot the H1 error vs the number of DoFs
    plt.figure()
    plt.xlabel("ndof")
    plt.ylabel("H1 error-estimate")
    ndof_n, err_n = zip(*(errs_lst[0]))
    ndof_a, err_a = zip(*(errs_lst[1]))
    plt.loglog(ndof_n,err_n, '*-', label="not adaptive")
    plt.loglog(ndof_a,err_a, "*-", label="adaptive")
    leg = plt.legend(loc='upper right')
    plt.savefig("results/err_ndofs.png")
    
    # plot the H1 error vs runtime
    plt.figure()
    plt.xlabel("runtime")
    plt.ylabel("H1 error-estimate")
    plt.loglog(runtimes_lst[0],err_n, '*-', label="not adaptive")
    plt.loglog(runtimes_lst[1],err_a, "*-", label="adaptive")
    leg = plt.legend(loc='upper right')
    plt.savefig("results/err_runtimes.png")    

    