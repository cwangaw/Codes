import netgen.gui

from ngsolve import *
from numpy import sin,cos,pi
#from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry

def SolvePoisson(mesh, deg=1, D=1, Lambda=1, f=0, g_b=0, g_l=0, g_r=0, g_t=0):
    '''
    Input:
        mesh: ngsolve mesh of the unit square without requiring the top edge to be straight
        D:    a positive constant, or eventually function (conductivity)
        Lambda:  non-negative constant from Robin boundary condition
        f:    forcing term
        g_b:  Dirichlet data imposed on bottom of domain
        g_l:  Neumann data imposed on left edge of domain
        g_r:  Neumann data imposed on right edge of domain
        g_t:  Robin data
        deg:  degree of Lagrange finite element space
    Returns:
        uh:   finite element solution
        flux_top: the flux through the top of uh
    
    The boundary value problem is - div (D grad u) = f on \Omega,
    with boundary conditions:
        on bottom edge: u = g_b
        on left edge: D * du/dn = g_l 
        on right edge: D * du/dn = g_r
        on top: Lambda * du/dn + u = g_t
    
    The weak form is  a(u, v) = L(v), where
    if Lambda > 0, we have
        a(u, v) = (D * grad(u), grad(v)) + <D/Lambda * u, v>_("top")
        and
        L(v) = (f, v) + <g_l, v>_("left") + <g_r, v>_("right") + <D/Lambda)*g_t, v>_("top")
    if Lambda = 0, we have
        a(u, v) = (D * grad(u), grad(v))
        and
        L(v) = (f, v) + <g_l, v>_("left") + <g_r, v>_("right")
    Note: if Lambda = 0, the boundary condition on the top is Dirichlet, not Robin    
    '''

    # initialize the finite element
    if Lambda > 0:
        fes = H1(mesh, order=deg, dirichlet="bottom", autoupdate=True)
    else:
        fes = H1(mesh, order=deg, dirichlet="bottom|top", autoupdate=True)
    
    u = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()   # symbolic object

    # intialize and define the bilinear form a(u,v)
    a = BilinearForm(fes)
    if Lambda > 0:
        a += D*grad(u)*grad(v)*dx + (D/Lambda)*u*v*ds("top")
    else:
        a += D*grad(u)*grad(v)*dx
    a.Assemble()

    # intialize and define the linear form L(v)
    L = LinearForm(fes)
    if Lambda > 0:
        L += f*v*dx + g_l*v*ds("left") + g_r*v*ds("right") + (D/Lambda)*g_t*v*ds("top")
    else:
        L += f*v*dx + g_l*v*ds("left") + g_r*v*ds("right")
    L.Assemble()

    uh = GridFunction(fes)  # solution
    
    # set up boundary conditions
    if Lambda > 0:
        uh.Set(g_b, BND)
    else:
        uh.Set(mesh.BoundaryCF({ "top" : g_t, "bottom" : g_b}), definedon=mesh.Boundaries("bottom|top"))

    # solve for the free dofs
    r = L.vec - a.mat * uh.vec
    uh.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
    
    if Lambda > 0:
        # the flux through the top of uh equals to <(D/Lambda)*u>_("top")
        flux_top = Integrate((D/Lambda)*uh, mesh, BND, definedon=mesh.Boundaries("top"))
    else:
        # the flux through the top of uh is <-D*du/dn>_("top")
        n = specialcf.normal(mesh.dim)
        gradu0 = GridFunction(fes)
        gradu1 = GridFunction(fes)
        gradu0.Set(grad(uh)[0])
        gradu1.Set(grad(uh)[1])
        flux_top = Integrate(-D*(gradu0*n[0]+gradu1*n[1]), mesh, BND, definedon=mesh.Boundaries("top"))
        
    return(uh, flux_top)

# update the pnts list and sgmnts list
def fractal_structure(p_start, p_end, pnts, sgmnts, current_level):
    num_pts = len(pnts)

    # if we have not yet reached the top level, add new points and new segments,
    # otherwise we only add the segment connecting p_start and p_end
    if (current_level > 0):
        # define the 3 new points added to current points
        p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (1/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (1/3) )
        p2 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (2/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (2/3) )
        p1 = ( p0[0] + cos(-pi/3) * (p2[0] - p0[0]) - sin(-pi/3) * (p2[1] - p0[1]), p0[1] + sin(-pi/3) * (p2[0] - p0[0]) + cos(-pi/3) * (p2[1] - p0[1]))
        
        pnts = pnts + [p0, p1, p2]
        (pnts, sgmnts) = fractal_structure(p_start,num_pts,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts,num_pts+1,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts+1,num_pts+2,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts+2,p_end,pnts,sgmnts,current_level-1)
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
        
    return pnts, sgmnts
    
# make mesh of fractal domain
def MakeGeometry(fractal_level):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    # the bottom, right, and left edges
    sgmnts = [(0,1), (1,2), (3,0)]
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = fractal_structure(2,3,pnts,sgmnts,fractal_level)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc = "bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc = "right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc = "left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc = "top")

    # calculate l = length of the shortest edge
    #           L_p = length of the fractal structure on the top
    l = 1 / (3.0**fractal_level)
    L_p = (4.0/3.0)**fractal_level
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=0.1))
    
    return mesh,l,L_p


if __name__ == "__main__":
    # set up the parameter for refining the fractal structure
    fractal_level = 3
    
    # set up the order of Lagrangian finite element
    poly_deg = 4

    # mesh generation
    (mesh,l,L_p) = MakeGeometry(fractal_level)
    
    # set up parameters for the pde
    D = 2.56
    Lambda = L_p

    # set up manufactured solution
    # and the corresponding source term and boundary data
    manu_sol = x**3 * y
    n = specialcf.normal(mesh.dim)
    du_nu = manu_sol.Diff(x)*n[0]+manu_sol.Diff(y)*n[1]
    f = -((D*manu_sol.Diff(x)).Diff(x) + (D*manu_sol.Diff(y)).Diff(y))
    g_b = manu_sol
    g_l = D*du_nu
    g_r = D*du_nu
    g_t = Lambda*du_nu + manu_sol
    (uh, flux) = SolvePoisson(mesh, poly_deg, D, Lambda, f, g_b, g_l, g_r, g_t)
    
    # plot the solution
    Draw(uh)
    
    # calculate and print the L2 error
    e = Integrate((uh-manu_sol)**2, mesh, VOL)
    e = sqrt(e)
    print("Error:", e)

    