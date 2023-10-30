from ngsolve import *

import numpy as np
import scipy.integrate
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import pdb, os

def makedir(resultsdir='results'):
    """
    Check that the directory resultsdir exists, and, if so, create a new subdirectory
    for the output using the first available name from 000000, 000001, ...  Then return its name.
    """
    assert os.path.isdir(resultsdir), 'The subdirectory {} must exist in the current directory'.format(resultsdir)
    id = 0
    while os.path.exists(resultsdir + '/{:06d}'.format(id)):
        id += 1
    outdir = resultsdir + '/{:06d}'.format(id)
    os.mkdir(outdir)
    print("*** created directory {} for output".format(outdir))
    return outdir

def savesolution(mesh, uh, savename):
    vtk = VTKOutput(mesh, coefs=[uh], names=["sol"], filename=savename, subdivision=2)
    vtk.Do()