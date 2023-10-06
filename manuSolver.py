from ngsolve import *

def SolveManuPoisson(mesh, d, lam):
    manu_sol = x * y
    n = specialcf.normal(mesh.dim)
    du_nu = manu_sol.Diff(x)*n[0]+manu_sol.Diff(y)*n[1]
    
    if lam > 0:
        fes = H1(mesh, order=3, dirichlet="bottom", autoupdate=True)
    else:
        fes = H1(mesh, order=3, dirichlet="bottom|top", autoupdate=True)
    
    u = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()   # symbolic object

    a = BilinearForm(fes)
    
    if lam > 0:
        a += d*grad(u)*grad(v)*dx + (1/lam)*u*v*ds("top")
    else:
        a += d*grad(u)*grad(v)*dx
    a.Assemble()

    f = LinearForm(fes)
    f += 0 * v * dx + du_nu * v * ds("left|right")
    if lam > 0:
        f += (1/lam)*(lam*du_nu+manu_sol)*v*ds("top")
    f.Assemble()

    gfu = GridFunction(fes)  # solution
    gfu.Set(manu_sol, BND)


    r = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
    
    if lam > 0:
        flux_top = Integrate((d/lam)*gfu, mesh, BND, definedon=mesh.Boundaries("top"))
    else:
        n = specialcf.normal(mesh.dim)
        gradu0 = GridFunction(fes)
        gradu1 = GridFunction(fes)
        gradu0.Set(grad(gfu)[0])
        gradu1.Set(grad(gfu)[1])
        flux_top = Integrate(-d*(gradu0*n[0]+gradu1*n[1]), mesh, BND, definedon=mesh.Boundaries("top"))
        
    # calculate L2 error
    e = Integrate((gfu-manu_sol)**2, mesh, VOL)
    e = sqrt(e)
    
    return(gfu, flux_top, e)