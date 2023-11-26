from ngsolve import *
from ngsolve.webgui import Draw
from time import time

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
for l in range(5): mesh.Refine()
fes = H1(mesh, order=2, dirichlet=".*")
print ("ndof =", fes.ndof)

u, v = fes.TnT()
with TaskManager():
    a = BilinearForm(grad(u)*grad(v)*dx+u*v*dx).Assemble()
    f = LinearForm(x*v*dx).Assemble()

gfu = GridFunction(fes)

jac = a.mat.CreateSmoother(fes.FreeDofs())

with TaskManager():
    inv_host = CGSolver(a.mat, jac, maxiter=2000)
    ts = time()
    gfu.vec.data = inv_host * f.vec
    te = time()
    print ("steps =", inv_host.GetSteps(), ", time =", te-ts)

try:
    from ngsolve.ngscuda import *
except:
    print ("no CUDA library or device available, using replacement types on host")

ngsglobals.msg_level=1
fdev = f.vec.CreateDeviceVector(copy=True)

adev = a.mat.CreateDeviceMatrix()
jacdev = jac.CreateDeviceMatrix()

inv = CGSolver(adev, jacdev, maxsteps=2000, printrates=False)

ts = time()
res = (inv * fdev).Evaluate()
te = time()

print ("Time on device:", te-ts)
diff = Norm(gfu.vec - res)
print ("diff = ", diff)