from ngsolve import *

import numpy as np
import scipy.integrate
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import pdb, os
from scipy.stats import linregress

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
    
def plot_coef(x, y, start, rounding = False):
    n = 10 #np.size(x)
    
    # we exclude the first 1/4 data
    x = np.array(x)
    y = np.array(y)

    p1, p0 = np.polyfit(np.log(x), np.log(y), deg=1)
 
    # calculating regression coefficients
    if rounding:
        p1 = round(p1)
    
    delta_x = exp(log(x.max()/x.min())/(n-1))
    delta_y = exp(log(y.max()/y.min())/(n-1))
    mid_x = start[0]
    mid_y = start[1]
    plt.loglog([mid_x,mid_x*delta_x,mid_x,mid_x], [mid_y,mid_y*exp(p1*log(delta_x)),mid_y*exp(p1*log(delta_x)),mid_y])
    # label
    if p1 < 0:
        plt.text(mid_x*sqrt(delta_x),mid_y*sqrt(exp(p1*log(delta_x))), "{0:.2f}".format(p1), horizontalalignment='left')
    else:
        plt.text(mid_x*sqrt(delta_x),mid_y*sqrt(exp(p1*log(delta_x))), "{0:.2f}".format(p1), horizontalalignment='right')
    
    # assign a suggested new starting point for plotting the slope triangle
    return [mid_x, mid_y*exp(p1*log(delta_x))**1.5]