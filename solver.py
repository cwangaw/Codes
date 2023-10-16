import netgen.gui
import matplotlib.pyplot as plt
import timeit
import contextlib, io, sys

from ngsolve import *
from numpy import sin,cos,pi
#from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry

def SolvePoisson(mesh, deg=1, d=1, lam=1, f=0, g_b=0, g_l=0, g_r=0, g_t=0, bool_adaptive = False, tol = 1e-5):
    '''
    Input:
        mesh: ngsolve mesh of the unit square without requiring the top edge to be straight
        d:    a positive constant, or eventually function (conductivity)
        lam:  non-negative constant from Robin boundary condition
        f:    forcing term
        g_b:  Dirichlet data imposed on bottom of domain
        g_l:  Neumann data imposed on left edge of domain
        g_r:  Neumann data imposed on right edge of domain
        g_t:  Robin data
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
    # initialize the finite element
    if lam > 0:
        fes = H1(mesh, order=deg, dirichlet="bottom", autoupdate=True)
    else:
        fes = H1(mesh, order=deg, dirichlet="bottom|top", autoupdate=True)
    
    u = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()   # symbolic object

    # intialize and define the bilinear form a(u,v)
    a = BilinearForm(fes)
    if lam > 0:
        a += d*grad(u)*grad(v)*dx + (d/lam)*u*v*ds("top")
    else:
        a += d*grad(u)*grad(v)*dx
    a.Assemble()

    # intialize and define the linear form l(v)
    l = LinearForm(fes)
    if lam > 0:
        l += f*v*dx + g_l*v*ds("left") + g_r*v*ds("right") + (d/lam)*g_t*v*ds("top")
    else:
        l += f*v*dx + g_l*v*ds("left") + g_r*v*ds("right")
    l.Assemble()

    # solution
    uh = GridFunction(fes, autoupdate=True)  
    
    # set up boundary conditions
    if lam > 0:
        uh.Set(g_b, BND)
    else:
        uh.Set(mesh.BoundaryCF({ "top" : g_t, "bottom" : g_b}), definedon=mesh.Boundaries("bottom|top"))

    # solve for the free dofs
    def SolveBVP():
        a.Assemble()
        l.Assemble()
        # Redraw (blocking=True)
        r = l.vec - a.mat * uh.vec
        uh.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
        
    
    errs = []
    
    # finite element space and gridfunction to represent
    # the heatflux:
    space_flux = HDiv(mesh, order=2, autoupdate=True)
    gf_flux = GridFunction(space_flux, "flux", autoupdate=True)
    
    def CalcError():
        flux = d * grad(uh)
        # interpolate finite element flux into H(div) space:
        gf_flux.Set (flux)

        # gradient-recovery error estimator
        err = 1/d*(flux-gf_flux)*(flux-gf_flux)
        elerr = Integrate (err, mesh, VOL, element_wise=True)
        # Integrate ( *dx(element_boundary=True), mesh, VOL, element_wise=True)

        maxerr = max(elerr)
        sumerr = sqrt(sum(elerr))
        errs.append((fes.ndof, sumerr))
        
        # if we want to refine the mesh adaptively,
        # we label the elements in need of refinements here
        if bool_adaptive:
            for el in mesh.Elements():
                mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25*maxerr)
        
        return sumerr

    # start the timer
    start = timeit.default_timer()

    # refine the mesh and solve the equation until the H1 error is within the tolerance
    with TaskManager():
        SolveBVP()
        while CalcError() > tol:
            mesh.Refine()
            SolveBVP()
    
    # stop the timer
    stop = timeit.default_timer()
    runtime = stop - start
    
    # plot the H1 error vs the number of DoFs
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("ndof")
    plt.ylabel("H1 error-estimate")
    ndof,err = zip(*errs)
    plt.plot(ndof,err, "-*")

    plt.ion()
    plt.show()

    input("<press enter to quit>")
        
    if lam > 0:
        # the flux through the top of uh equals to <(d/lam)*u>_("top")
        flux_top = Integrate((d/lam)*uh, mesh, BND, definedon=mesh.Boundaries("top"))
    else:
        # the flux through the top of uh is <-d*du/dn>_("top")
        n = specialcf.normal(mesh.dim)
        gradu0 = GridFunction(fes)
        gradu1 = GridFunction(fes)
        gradu0.Set(grad(uh)[0])
        gradu1.Set(grad(uh)[1])
        flux_top = Integrate(-d*(gradu0*n[0]+gradu1*n[1]), mesh, BND, definedon=mesh.Boundaries("top"))
        
    return uh, flux_top, runtime, fes.ndof

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

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc = "bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc = "right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc = "left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc = "top")

    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1 / (3.0**fractal_level)
    ell_p = (4.0/3.0)**fractal_level
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh, ell_e, ell_p
    
if __name__ == "__main__":
    # set up the parameter for refining the fractal structure
    fractal_level = 3
    
    # set up the order of Lagrangian finite element
    poly_deg = 2
    
    # set up the desired tolerance for the parameter eta in mesh refinment
    tol = 1e-5
    
    # mesh generation
    (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
    
    # set up parameters for the pde
    d = 2.56
    lam = ell_p

    # set up manufactured solution
    # and the corresponding source term and boundary data
    manu_sol = x**3 * y
    n = specialcf.normal(mesh.dim)
    du_nu = manu_sol.Diff(x)*n[0]+manu_sol.Diff(y)*n[1]
    f = -((d*manu_sol.Diff(x)).Diff(x) + (d*manu_sol.Diff(y)).Diff(y))
    g_b = manu_sol
    g_l = d*du_nu
    g_r = d*du_nu
    g_t = lam*du_nu + manu_sol
    
    # Compare the running time and number of dofs for 
    # traditionally and adaptively refined mesh
    result = open("comparison.txt", "a")
    
    for is_adaptive in [True, False]:
        # initialize a new mesh, on which we solve the pde
        (mesh, ell_e, ell_p) = MakeGeometry(fractal_level)
        (uh, flux, runtime, ndofs) = SolvePoisson(mesh, poly_deg, d, lam, f, g_b, g_l, g_r, g_t, is_adaptive, tol)
        
        # calculate and print the L2 error
        e = Integrate((uh-manu_sol)**2, mesh, VOL)
        e = sqrt(e)
        
        # append the results into the file
        result.write("Adaptivity: " + str(is_adaptive) + ", Solving time: " + str(runtime) + ", nDoFs:" + str(ndofs) + ", Error:" + str(e) + '\n')
    
    result.write('\n')
    result.close()
    