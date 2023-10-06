from ngsolve import *

def SolvePoisson(mesh, d, lam):
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
    f += 0 * v *dx
    f.Assemble()

    gfu = GridFunction(fes)  # solution
    if lam > 0:
        gfu.Set(1, BND)
    else:
        gfu.Set(mesh.BoundaryCF({ "top" : 0, "bottom" : 1}), definedon=mesh.Boundaries("bottom|top"))

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
    return(gfu, flux_top)