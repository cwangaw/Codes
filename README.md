# Usage

## Windows

For a windows user, you need to use Python3 with version 3.7 to make NGSolve work.

To solve for the fractal boundary problem described in our notes, use
```
py -3.7 -i main.py
```

To compare the performance of adaptive mesh refinement and global mesh refinement for 

* solving the fractal boundary problem described in our notes with a user-defined lam, for example, _lam = 2.56_, use
```
py -3.7 -i solver.py lam 2.56
```

* solving the fractal boundary problem with a manufactured solution, use
```
py -3.7 -i solver.py
```

* solving -div(grad u) = 1 with homogeneous Dirichlet boundary condition on an L-shape domain, where the adaptive mesh works much better, use
```
py -3.7 -i solver.py singular
```

## Linux

To solve for the fractal boundary problem described in our notes, use
```
python3 -i main.py
```

To compare the performance of adaptive mesh refinement and global mesh refinement for 

* solving the fractal boundary problem described in our notes with a user-defined lam, for example, _lam = 2.56_, use
```
python3 -i solver.py lam 2.56
```

* solving the fractal boundary problem with a manufactured solution, use
```
python3 -i solver.py
```

* solving -div(grad u) = 1 with homogeneous Dirichlet boundary condition on an L-shape domain, where the adaptive mesh works much better, use
```
python3 -i solver.py singular
```