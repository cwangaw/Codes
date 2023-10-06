import netgen.gui

from numpy import sin,cos,pi
from ngsolve import Mesh
from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry

''' we define the list of points p_i,
    and a list of tuples (i_1,i_2), 
    representing a line with starting point p_{i_1} and ending point p_{i_2}
'''

# update the pnts list and sgmnts list
def fractal_structure(p_start, p_end, pnts, sgmnts, current_level):
    num_pts = len(pnts)

    p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (1/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (1/3) )
    p2 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (2/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (2/3) )
    p1 = (p0[0] + cos(-pi/3) * (p2[0] - p0[0]) - sin(-pi/3) * (p2[1] - p0[1]), p0[1] + sin(-pi/3) * (p2[0] - p0[0]) + cos(-pi/3) * (p2[1] - p0[1]))

    pnts = pnts + [p0, p1, p2]
    
    if (current_level > 0):
        (pnts, sgmnts) = fractal_structure(p_start,num_pts,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts,num_pts+1,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts+1,num_pts+2,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = fractal_structure(num_pts+2,p_end,pnts,sgmnts,current_level-1)
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
    return pnts, sgmnts
    
# define number of levels here
def MakeGeometry(fractal_level):
    geo = SplineGeometry()

    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    sgmnts = [(0,1), (1,2), (3,0)]
    (pnts, sgmnts) = fractal_structure(2,3,pnts,sgmnts,fractal_level)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc = "bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc = "right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc = "left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc = "top")

    l = 1 / (3.0**fractal_level)
    L_p = (4.0/3.0)**fractal_level
    
    mesh = Mesh(geo.GenerateMesh(maxh=0.1))
    return mesh,l,L_p
