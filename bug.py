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

def SolvePoisson(mesh, bc, deg=1, d=1, lam=1, f=0, bool_adaptive = False, tol = 1e-5, max_it = 50):
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
    #c = MultiGridPreconditioner(a, inverse = "sparsecholesky") # Register c to a BEFORE assembly

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

        r = l.vec - a.mat * uh.vec
        
        # preconditioner
        #inv = CGSolver(a.mat, c.mat)
        #uh.vec.data += inv * r
        
        # direct solver
        uh.vec.data += a.mat.Inverse(fes.FreeDofs()) * r
           
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
        SolveBVP()
        time_delta = datetime.datetime.now() - start
        runtimes.append(time_delta.total_seconds())
        it = 0
        while CalcError() > tol and it < max_it:
            mesh.Refine()
            SolveBVP()
            time_delta = datetime.datetime.now() - start
            runtimes.append(time_delta.total_seconds())
            it += 1
            if it == max_it:
                warningmsg = "Number of iterations reaches max_it " + str(max_it) + " before the H1 error estimator reaching tolerance " + str(tol) + "\n"
                print(warningmsg)

    return uh, runtimes, errs

def MakeGeometry(h_max = 0.1):
    cube =  Box(Pnt(0,0,0), Pnt(1,1,1))
    for i in range(6):
        if cube.faces[i].center[0]==0 or cube.faces[i].center[0]==1 or cube.faces[i].center[1]==0 or cube.faces[i].center[1]==1:
            cube.faces[i].name = 'side'
        elif cube.faces[i].center[2] == 0:
            cube.faces[i].name = 'bottom'
        else:
            cube.faces[i].name = 'top'
    geo = OCCGeometry(cube)
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    return mesh
    
if __name__ == "__main__":
    # set up the order of Lagrangian finite element
    poly_deg = 5

    # set up if we want to refine the mesh adaptively
    is_adaptive = True

    # set up the error tolerance for mesh refinement
    tol = 1e-8

    # set up the parameter for pde
    d = 1
    lam = d

    # set up max number of iteration
    max_it = 5
    
    # mesh generation
    mesh = MakeGeometry()
    
    # set up boundary conditions
    bc = {"d": {"bottom": 1}, "n": {"side": 0}, "r": {"top": 0}}
    
    (uh, _, _) = SolvePoisson(mesh, bc, poly_deg, d, lam, 0, is_adaptive, tol, max_it)
    
    Draw(mesh)