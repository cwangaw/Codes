from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.occ import *
import numpy as np

def MakeGeometry(geotype, fractal_level=2, scaling=1, h_max=2):
    if scaling<0 or scaling>1:
        print("WARNING: Illegal scaling factor! We use 1 as the default setting.")
        scaling = 1
    
    if geotype == "2d":
        return MakeKochCurveGeo(fractal_level, h_max)
    elif geotype == "2dquad":
        return MakeQuadKochCurveGeo(fractal_level, scaling, h_max)
    elif geotype == "2dmidquad":
        return MakeMidQuadKochCurveGeo(fractal_level, h_max)
    elif geotype == "lshape":
        return MakeLShapeGeo(h_max)
    elif geotype == "3d":
        return MakeKochSurfaceGeo(fractal_level, scaling, h_max)
    elif geotype == "3dmid":
        return MakeMidKochSurfaceGeo(fractal_level, h_max)
    elif geotype == "3dquad":
        return MakeQuadKochSurfaceGeo(fractal_level, scaling, h_max)
    elif geotype == "3dmidquad":
        return MakeMidQuadKochSurfaceGeo(fractal_level, h_max)
    elif geotype == "3d14quad":
        return MakeOneFourthQuadKochSurfaceGeo(fractal_level, h_max)
    else:
        print("None of the following types is selected: 2d, 2dquad, 2dmidquad, 3d, 3dmid, 3dquad, 3dmidquad, 3d14quad")
        print("Defaulty, we generate the square domain with the top edge replaced by Koch Snowflake")
        return MakeKochCurveGeo(fractal_level, h_max)

def GrowKochCurve(p_start, p_end, pnts, sgmnts, current_level):
    num_pts = len(pnts)

    # if we have not yet reached the top level, add new points and new segments,
    # otherwise we only add the segment connecting p_start and p_end
    if (current_level > 0):
        # define the 3 new points added to current points
        p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (1/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (1/3) )
        p2 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (2/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (2/3) )
        p1 = ( p0[0] + cos(-pi/3) * (p2[0] - p0[0]) - sin(-pi/3) * (p2[1] - p0[1]), p0[1] + sin(-pi/3) * (p2[0] - p0[0]) + cos(-pi/3) * (p2[1] - p0[1]))
        
        pnts = pnts + [p0, p1, p2]
        (pnts, sgmnts) = GrowKochCurve(p_start,num_pts,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = GrowKochCurve(num_pts,num_pts+1,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = GrowKochCurve(num_pts+1,num_pts+2,pnts,sgmnts,current_level-1)
        (pnts, sgmnts) = GrowKochCurve(num_pts+2,p_end,pnts,sgmnts,current_level-1)
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
        
    return pnts, sgmnts

# make mesh of fractal domain
def MakeKochCurveGeo(fractal_level, h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    # the bottom, right, and left edges
    sgmnts = [(0,1), (1,2), (3,0)]
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = GrowKochCurve(2,3,pnts,sgmnts,fractal_level)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc="bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc="right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc="left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc="top")
    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1 / (3.0**fractal_level)
    ell_p = (4.0/3.0)**fractal_level
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh, ell_e, ell_p

# update the pnts list and sgmnts list
def GrowQuadKochCurve(scaling, p_start, p_end, pnts, sgmnts, current_level, pos):
    # if we have not yet reached the top level, add new points and new segments,
    # otherwise we only add the segment connecting p_start and p_end
    if (current_level > 0):
        # define the 4 new points 
        # add to pnts if is new
        # otherwise, retrieve the index
        
        if pos==0:
            p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * ((1-scaling/3)/2), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * ((1-scaling/3)/2) )
            p3 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * ((1+scaling/3)/2), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * ((1+scaling/3)/2) )
        elif pos==-1:
            newpend =  (pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * 2/(3-scaling), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * 2/(3-scaling))
            p0 = ( pnts[p_start][0] + (newpend[0] - pnts[p_start][0]) * ((1-scaling/3)/2), pnts[p_start][1] + (newpend[1] - pnts[p_start][1]) * ((1-scaling/3)/2) )
            p3 = ( pnts[p_start][0] + (newpend[0] - pnts[p_start][0]) * ((1+scaling/3)/2), pnts[p_start][1] + (newpend[1] - pnts[p_start][1]) * ((1+scaling/3)/2) )
        else:
            newpstart = (pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * 1/(3-scaling) * (1-scaling), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * 1/(3-scaling) * (1-scaling))
            p0 = ( newpstart[0] + (pnts[p_end][0] - newpstart[0]) * ((1-scaling/3)/2), newpstart[1] + (pnts[p_end][1] - newpstart[1]) * ((1-scaling/3)/2) )
            p3 = ( newpstart[0] + (pnts[p_end][0] - newpstart[0]) * ((1+scaling/3)/2), newpstart[1] + (pnts[p_end][1] - newpstart[1]) * ((1+scaling/3)/2) )
            
        p1 = ( p0[0] + (p3[1] - p0[1]), p0[1] - (p3[0] - p0[0]))
        p2 = ( p1[0] + (p3[0] - p0[0]), p1[1] + (p3[1] - p0[1]))
        
        indices = []
        for pt in [p0, p1, p2, p3]:
            bool_exists = False
            for i in range(len(pnts)):
                if abs(pt[0]-pnts[i][0]) < 1e-12 and abs(pt[1]-pnts[i][1]) < 1e-12:
                    bool_exists = True
                    indices = indices + [i]
            if bool_exists == False:
                indices = indices + [len(pnts)]
                pnts = pnts + [pt]
                
        
        (pnts, sgmnts) = GrowQuadKochCurve(scaling,p_start,indices[0],pnts,sgmnts,current_level-1,-1)
        (pnts, sgmnts) = GrowQuadKochCurve(scaling,indices[0],indices[1],pnts,sgmnts,current_level-1,0)
        (pnts, sgmnts) = GrowQuadKochCurve(scaling,indices[1],indices[2],pnts,sgmnts,current_level-1,0)
        (pnts, sgmnts) = GrowQuadKochCurve(scaling,indices[2],indices[3],pnts,sgmnts,current_level-1,0)
        (pnts, sgmnts) = GrowQuadKochCurve(scaling,indices[3],p_end,pnts,sgmnts,current_level-1,1)      
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
        
    return pnts, sgmnts
    
# make mesh of fractal domain
def MakeQuadKochCurveGeo(fractal_level, scaling = 1, h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    # the bottom, right, and left edges
    sgmnts = [(0,1), (1,2), (3,0)]
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = GrowQuadKochCurve(scaling,2,3,pnts,sgmnts,fractal_level,0)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc="bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc="right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc="left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc="top")
    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1/(3.0**fractal_level)

    a = 2/3 + scaling
    b = (1-scaling)/3
    
    ell_p = a**fractal_level*(1+b/(a-1))-b/(a-1)
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh, ell_e, ell_p

# update the pnts list and sgmnts list
def GrowMidQuadKochCurve(p_start, p_end, pnts, sgmnts, current_level, isMid):
    # if we have not yet reached the top level, add new points and new segments,
    # otherwise we only add the segment connecting p_start and p_end
    if (current_level > 0) and isMid:
        # define the 4 new points 
        # add to pnts if is new
        # otherwise, retrieve the index
        p0 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (1/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (1/3) )
        p3 = ( pnts[p_start][0] + (pnts[p_end][0] - pnts[p_start][0]) * (2/3), pnts[p_start][1] + (pnts[p_end][1] - pnts[p_start][1]) * (2/3) )
        p1 = ( p0[0] + (p3[1] - p0[1]), p0[1] - (p3[0] - p0[0]))
        p2 = ( p1[0] + (p3[0] - p0[0]), p1[1] + (p3[1] - p0[1]))

        indices = []
        for pt in [p0, p1, p2, p3]:
            bool_exists = False
            for i in range(len(pnts)):
                if abs(pt[0]-pnts[i][0]) < 1e-12 and abs(pt[1]-pnts[i][1]) < 1e-12:
                    bool_exists = True
                    indices = indices + [i]
            if bool_exists == False:
                indices = indices + [len(pnts)]
                pnts = pnts + [pt]
                
        (pnts, sgmnts) = GrowMidQuadKochCurve(indices[0],indices[1],pnts,sgmnts,current_level-1,1)
        (pnts, sgmnts) = GrowMidQuadKochCurve(indices[1],indices[2],pnts,sgmnts,current_level-1,1)
        (pnts, sgmnts) = GrowMidQuadKochCurve(indices[2],indices[3],pnts,sgmnts,current_level-1,1)
        
        (pnts, sgmnts) = GrowMidQuadKochCurve(p_start,indices[0],pnts,sgmnts,current_level-1,0)
        (pnts, sgmnts) = GrowMidQuadKochCurve(indices[3],p_end,pnts,sgmnts,current_level-1,0)
    else:
        sgmnts = sgmnts + [(p_start, p_end)]
        
    return pnts, sgmnts
    
# make mesh of fractal domain
def MakeMidQuadKochCurveGeo(fractal_level, h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [(0,0), (1,0), (1,1), (0,1)]
    # the bottom, right, and left edges
    sgmnts = [(0,1), (1,2), (3,0)]
    
    # add points and segments for the top fractal structure
    (pnts, sgmnts) = GrowMidQuadKochCurve(2,3,pnts,sgmnts,fractal_level,1)

    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])

    geo.Append (["line", sgmnts[0][0], sgmnts[0][1]], bc="bottom")
    geo.Append (["line", sgmnts[1][0], sgmnts[1][1]], bc="right")
    geo.Append (["line", sgmnts[2][0], sgmnts[2][1]], bc="left")
    
    for i in range(3,len(sgmnts)):
        geo.Append (["line", sgmnts[i][0], sgmnts[i][1]], bc="top")
    # calculate ell_e = length of the shortest edge
    #           ell_p = length (parimeter) of the fractal structure on the top
    ell_e = 1/(3.0**fractal_level)
    ell_p = 1 + (2.0/3.0)**fractal_level
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh, ell_e, ell_p

def MakeLShapeGeo(h_max = 0.2):
    geo = SplineGeometry()
    
    # the four vertices of the square domian
    pnts = [ (0,0), (1,0), (1,1), (-1,1), (-1,-1), (0,-1) ]
    
    for i in range(len(pnts)):
        geo.AppendPoint (pnts[i][0], pnts[i][1])
        if i > 0:
            geo.Append (["line", i-1, i], bc="dirichlet")
    
    geo.Append (["line", len(pnts)-1, 0], bc="dirichlet")
    
    # mesh generation
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))
    
    return mesh

def Tetra(pnt1, pnt2, pnt3, pnt4):
    seg12 = Segment(pnt1, pnt2)
    seg13 = Segment(pnt1, pnt3)
    seg14 = Segment(pnt1, pnt4)
    seg23 = Segment(pnt2, pnt3)
    seg24 = Segment(pnt2, pnt4)
    seg34 = Segment(pnt3, pnt4)

    w123 = Wire([seg12, seg23, seg13])
    w124 = Wire([seg12, seg24, seg14])
    w134 = Wire([seg13, seg34, seg14])
    w234 = Wire([seg23, seg24, seg34])

    f123 = Face(w123)
    f124 = Face(w124)
    f134 = Face(w134)
    f234 = Face(w234)

    h123 = Vec(pnt4.x-(pnt1.x+pnt2.x+pnt3.x)/3, pnt4.y-(pnt1.y+pnt2.y+pnt3.y)/3, pnt4.z-(pnt1.z+pnt2.z+pnt3.z)/3)
    h234 = Vec(pnt1.x-(pnt2.x+pnt3.x+pnt4.x)/3, pnt1.y-(pnt2.y+pnt3.y+pnt4.y)/3, pnt1.z-(pnt2.z+pnt3.z+pnt4.z)/3)
    h134 = Vec(pnt2.x-(pnt1.x+pnt3.x+pnt4.x)/3, pnt2.y-(pnt1.y+pnt3.y+pnt4.y)/3, pnt2.z-(pnt1.z+pnt3.z+pnt4.z)/3)
    h124 = Vec(pnt3.x-(pnt1.x+pnt2.x+pnt4.x)/3, pnt3.y-(pnt1.y+pnt2.y+pnt4.y)/3, pnt3.z-(pnt1.z+pnt2.z+pnt4.z)/3)

    t = Prism(f123,h123) * Prism(f234,h234) * Prism(f134,h134) * Prism(f124,h124)
    
    return t

def GrowKochSurface(solid, scaling, pnt1, pnt2, pnt3, current_level):
    if current_level > 0:
        midpnt3 = Pnt((pnt1.x+pnt2.x)/2, (pnt1.y+pnt2.y)/2, (pnt1.z+pnt2.z)/2)
        midpnt2 = Pnt((pnt1.x+pnt3.x)/2, (pnt1.y+pnt3.y)/2, (pnt1.z+pnt3.z)/2)
        midpnt1 = Pnt((pnt2.x+pnt3.x)/2, (pnt2.y+pnt3.y)/2, (pnt2.z+pnt3.z)/2)
        
        center = Pnt((midpnt1.x+midpnt2.x+midpnt3.x)/3, (midpnt1.y+midpnt2.y+midpnt3.y)/3, (midpnt1.z+midpnt2.z+midpnt3.z)/3)
        
        newpnt1 = Pnt(center.x+scaling*(midpnt1.x-center.x), center.y+scaling*(midpnt1.y-center.y), center.z+scaling*(midpnt1.z-center.z))
        newpnt2 = Pnt(center.x+scaling*(midpnt2.x-center.x), center.y+scaling*(midpnt2.y-center.y), center.z+scaling*(midpnt2.z-center.z))
        newpnt3 = Pnt(center.x+scaling*(midpnt3.x-center.x), center.y+scaling*(midpnt3.y-center.y), center.z+scaling*(midpnt3.z-center.z))
        
        vec1 = newpnt2-newpnt1
        vec2 = newpnt3-newpnt1
        vec1 = [vec1.x, vec1.y, vec1.z]
        vec2 = [vec2.x, vec2.y, vec2.z]
        normal = np.cross(vec1,vec2)
        normal = normal * (sqrt(6)/3) * np.linalg.norm(vec1) / np.linalg.norm(normal)
        
        newpnt4 = Pnt(normal[0]+center.x, normal[1]+center.y, normal[2]+center.z)
        
        newtetra = Tetra(newpnt1,newpnt2,newpnt3,newpnt4)
        newtetra.bc("top")
        solid = solid + newtetra.bc("top")
    
    if current_level > 1:
        solid = GrowKochSurface(solid, scaling, pnt1, midpnt3, midpnt2, current_level-1)
        solid = GrowKochSurface(solid, scaling, pnt2, midpnt1, midpnt3, current_level-1)
        solid = GrowKochSurface(solid, scaling, pnt3, midpnt2, midpnt1, current_level-1)
        solid = GrowKochSurface(solid, scaling, newpnt1, newpnt4, newpnt3, current_level-1)
        solid = GrowKochSurface(solid, scaling, newpnt3, newpnt4, newpnt2, current_level-1)
        solid = GrowKochSurface(solid, scaling, newpnt2, newpnt4, newpnt1, current_level-1)
    
    return solid

def MakeKochSurfaceGeo(fractal_level, scaling = 1, h_max = 2):
    pnt1 = Pnt(0,0,0)
    pnt2 = Pnt(1,0,0)
    pnt3 = Pnt(1/2,sqrt(3)/2,0)

    seg1 = Segment(pnt1, pnt2)
    seg2 = Segment(pnt2, pnt3)
    seg3 = Segment(pnt3, pnt1)

    w = Wire([seg1, seg2, seg3])

    f = Face(w)

    solid = Prism(f,Z)
    
    for i in range(5):
        if solid.faces[i].center[2] == 0:
            solid.faces[i].name = 'bottom'
        elif solid.faces[i].center[2] < 1:
            solid.faces[i].name = 'side'
        else:
            solid.faces[i].name = 'top'
    
    solid = GrowKochSurface(solid, scaling, Pnt(0,0,1), Pnt(1,0,1), Pnt(1/2,sqrt(3)/2,1), fractal_level)
    geo = OCCGeometry(solid)
    mesh = Mesh(geo.GenerateMesh(maxh = h_max))
    
    l = (scaling/2)**fractal_level
    
    a = 3/4 + 3*scaling*scaling/4
    b = (1-scaling*scaling)/4
    
    A_p = sqrt(3)/4 * (a**fractal_level*(1+b/(a-1))-b/(a-1))
    
    return mesh,l,A_p

def GrowMidKochSurface(solid, pnt1, pnt2, pnt3, current_level):
    if current_level > 0:
        newpnt3 = Pnt((pnt1.x+pnt2.x)/2, (pnt1.y+pnt2.y)/2, (pnt1.z+pnt2.z)/2)
        newpnt2 = Pnt((pnt1.x+pnt3.x)/2, (pnt1.y+pnt3.y)/2, (pnt1.z+pnt3.z)/2)
        newpnt1 = Pnt((pnt2.x+pnt3.x)/2, (pnt2.y+pnt3.y)/2, (pnt2.z+pnt3.z)/2)
        
        vec1 = newpnt2-newpnt1
        vec2 = newpnt3-newpnt1
        vec1 = [vec1.x, vec1.y, vec1.z]
        vec2 = [vec2.x, vec2.y, vec2.z]
        normal = np.cross(vec1,vec2)
        normal = normal * (sqrt(6)/3) * np.linalg.norm(vec1) / np.linalg.norm(normal)
        
        newpnt4 = Pnt(normal[0]+(newpnt1.x+newpnt2.x+newpnt3.x)/3, normal[1]+(newpnt1.y+newpnt2.y+newpnt3.y)/3, normal[2]+(newpnt1.z+newpnt2.z+newpnt3.z)/3)
        
        newtetra = Tetra(newpnt1,newpnt2,newpnt3,newpnt4)
        newtetra.bc("top")
        solid = solid + newtetra.bc("top")
    
    if current_level > 1:
        solid = GrowMidKochSurface(solid, newpnt1, newpnt4, newpnt3, current_level-1)
        solid = GrowMidKochSurface(solid, newpnt3, newpnt4, newpnt2, current_level-1)
        solid = GrowMidKochSurface(solid, newpnt2, newpnt4, newpnt1, current_level-1)
    
    return solid

def MakeMidKochSurfaceGeo(fractal_level, h_max = 2):
    pnt1 = Pnt(0,0,0)
    pnt2 = Pnt(1,0,0)
    pnt3 = Pnt(1/2,sqrt(3)/2,0)

    seg1 = Segment(pnt1, pnt2)
    seg2 = Segment(pnt2, pnt3)
    seg3 = Segment(pnt3, pnt1)

    w = Wire([seg1, seg2, seg3])

    f = Face(w)

    solid = Prism(f,Z)
    
    for i in range(5):
        if solid.faces[i].center[2] == 0:
            solid.faces[i].name = 'bottom'
        elif solid.faces[i].center[2] < 1:
            solid.faces[i].name = 'side'
        else:
            solid.faces[i].name = 'top'
    
    solid = GrowMidKochSurface(solid, Pnt(0,0,1), Pnt(1,0,1), Pnt(1/2,sqrt(3)/2,1), fractal_level)
    geo = OCCGeometry(solid)
    mesh = Mesh(geo.GenerateMesh(maxh = h_max))
    
    l = (1/2)**fractal_level
    A_p = sqrt(3)/4 * (3-2*(3/4)**fractal_level)
    
    return mesh,l,A_p

def GrowQuadKochSurface(cube, scaling, p_min, vec1, vec2, vec3, current_level):
    # p_min, vec1, vec2 provide a square surface on which we grow fractal structure of one lower level
    # p_min is a vector
    # starting from p_min, vec1 and vec2 forms a square surface with vec1 x vec2 pointing outward
    # vec3 is an outward-pointing vector
    # vec1, vec2, vec3 has length the same as the square

    if current_level > 0:
        p_cand = [p_min+((1-scaling)/6+1/3)*vec1+((1-scaling)/6+1/3)*vec2+scaling*(1/3)*vec3, p_min+((1+scaling)/6+1/3)*vec1+((1+scaling)/6+1/3)*vec2-(1/30)*scaling*(1/3)*vec3]
        new_min = Pnt(min(p_cand[0].x,p_cand[1].x), min(p_cand[0].y,p_cand[1].y), min(p_cand[0].z,p_cand[1].z))
        new_max = Pnt(max(p_cand[0].x,p_cand[1].x), max(p_cand[0].y,p_cand[1].y), max(p_cand[0].z,p_cand[1].z))
        new_cube = Box(new_min, new_max)
        new_cube.bc("top")
        cube = cube + new_cube.bc("top")
    
    if current_level > 1:
        for i in range(3):
            for j in range(3):
                if i==1 and j==1:
                    cube = GrowQuadKochSurface(cube, scaling, p_min+((1-scaling)/6+1/3)*vec1+((1-scaling)/6+1/3)*vec2+scaling*(1/3)*vec3, (1/3)*scaling*vec1, (1/3)*scaling*vec2, (1/3)*scaling*vec3, current_level-1)
                else:
                    cube = GrowQuadKochSurface(cube, scaling, p_min+(i/3)*vec1+(j/3)*vec2, (1/3)*vec1, (1/3)*vec2, (1/3)*vec3, current_level-1)
        
        # add fractal structures on the four square surfaces surrounding the new cube
        cube = GrowQuadKochSurface(cube, scaling, p_min+((1-scaling)/6+1/3)*vec1+((1-scaling)/6+1/3)*vec2, (1/3)*scaling*vec1, (1/3)*scaling*vec3, (-(1/3)*scaling)*vec2, current_level-1)
        cube = GrowQuadKochSurface(cube, scaling, p_min+((1-scaling)/6+1/3)*vec1+((1+scaling)/6+1/3)*vec2, (-(1/3)*scaling)*vec2, ((1/3)*scaling)*vec3, (-(1/3)*scaling)*vec1, current_level-1)
        cube = GrowQuadKochSurface(cube, scaling, p_min+((1+scaling)/6+1/3)*vec1+((1+scaling)/6+1/3)*vec2, (-(1/3)*scaling)*vec1, (1/3)*scaling*vec3, (1/3)*scaling*vec2, current_level-1)
        cube = GrowQuadKochSurface(cube, scaling, p_min+((1+scaling)/6+1/3)*vec1+((1-scaling)/6+1/3)*vec2, (1/3)*scaling*vec2, (1/3)*scaling*vec3, (1/3)*scaling*vec1, current_level-1)
    
    return cube
        
def MakeQuadKochSurfaceGeo(fractal_level, scaling = 1, h_max = 0.2):
    cube =  Box(Pnt(0,0,0), Pnt(1,1,1))
    for i in range(6):
        if cube.faces[i].center[0]==0 or cube.faces[i].center[0]==1 or cube.faces[i].center[1]==0 or cube.faces[i].center[1]==1:
            cube.faces[i].name = 'side'
        elif cube.faces[i].center[2] == 0:
            cube.faces[i].name = 'bottom'
        else:
            cube.faces[i].name = 'top'
    
    cube = GrowQuadKochSurface(cube, scaling, Vec(0,0,1), Vec(1,0,0), Vec(0,1,0), Vec(0,0,1), fractal_level)
    geo = OCCGeometry(cube)
    mesh = Mesh(geo.GenerateMesh(maxh = h_max))
    
    l = (1/16)**fractal_level
    
    a = 8/9 + 5*scaling*scaling/9
    b = (1-scaling*scaling)/9
    
    A_p = a**fractal_level*(1+b/(a-1))-b/(a-1)
    
    return mesh,l,A_p

def GrowMidQuadKochSurface(cube, p_min, vec1, vec2, vec3, current_level):
    # p_min, vec1, vec2 provide a square surface on which we grow fractal structure of one lower level
    # p_min is a vector
    # starting from p_min, vec1 and vec2 forms a square surface with vec1 x vec2 pointing outward
    # vec3 is an outward-pointing vector
    # vec1, vec2, vec3 has length the same as the square

    if current_level > 0:
        p_cand = [p_min+(1/3)*vec1+(1/3)*vec2, p_min+(2/3)*vec1+(2/3)*vec2+(1/3)*vec3]
        new_min = Pnt(min(p_cand[0].x,p_cand[1].x), min(p_cand[0].y,p_cand[1].y), min(p_cand[0].z,p_cand[1].z))
        new_max = Pnt(max(p_cand[0].x,p_cand[1].x), max(p_cand[0].y,p_cand[1].y), max(p_cand[0].z,p_cand[1].z))
        new_cube = Box(new_min, new_max)
        new_cube.bc("top")
        cube = cube + new_cube.bc("top")
    
    if current_level > 1:
        for i in range(3):
            for j in range(3):
                if i==1 and j==1:
                    cube = GrowMidQuadKochSurface(cube, p_min+(1/3)*vec1+(1/3)*vec2+(1/3)*vec3, (1/3)*vec1, (1/3)*vec2, (1/3)*vec3, current_level-1)
        
        # add fractal structures on the four square surfaces surrounding the new cube
        cube = GrowMidQuadKochSurface(cube, p_min+(1/3)*vec1+(1/3)*vec2, (1/3)*vec1, (1/3)*vec3, (-1/3)*vec2, current_level-1)
        cube = GrowMidQuadKochSurface(cube, p_min+(1/3)*vec1+(2/3)*vec2, (-1/3)*vec2, (1/3)*vec3, (-1/3)*vec1, current_level-1)
        cube = GrowMidQuadKochSurface(cube, p_min+(2/3)*vec1+(2/3)*vec2, (-1/3)*vec1, (1/3)*vec3, (1/3)*vec2, current_level-1)
        cube = GrowMidQuadKochSurface(cube, p_min+(2/3)*vec1+(1/3)*vec2, (1/3)*vec2, (1/3)*vec3, (1/3)*vec1, current_level-1)
    
    return cube
        
def MakeMidQuadKochSurfaceGeo(fractal_level, h_max = 0.2):
    cube =  Box(Pnt(0,0,0), Pnt(1,1,1))
    for i in range(6):
        if cube.faces[i].center[0]==0 or cube.faces[i].center[0]==1 or cube.faces[i].center[1]==0 or cube.faces[i].center[1]==1:
            cube.faces[i].name = 'side'
        elif cube.faces[i].center[2] == 0:
            cube.faces[i].name = 'bottom'
        else:
            cube.faces[i].name = 'top'
    
    cube = GrowMidQuadKochSurface(cube, Vec(0,0,1), Vec(1,0,0), Vec(0,1,0), Vec(0,0,1), fractal_level)
    geo = OCCGeometry(cube)
    mesh = Mesh(geo.GenerateMesh(maxh = h_max))
    
    l = (1/2)**fractal_level
    A_p = 2-(5/9)**fractal_level
    
    return mesh,l,A_p

# first we define a function that will be called iteratively in MakeCSGeometry()
def GrowOneFourthQuadKochSurface(cube, p_min, vec1, vec2, vec3, current_level):
    # p_min, vec1, vec2 provide a square surface on which we grow fractal structure of one lower level
    # p_min is a vector
    # starting from p_min, vec1 and vec2 forms a square surface with vec1 x vec2 pointing outward
    # vec3 is an outward-pointing vector
    # vec1, vec2, vec3 has length the same as the square

    if current_level > 0:
        p_cand = [p_min+(1/2)*vec1+(1/2)*vec2+(1/4)*vec3, p_min+(3/4)*vec1+(3/4)*vec2-(1/30)*vec3]
        new_min = Pnt(min(p_cand[0].x,p_cand[1].x), min(p_cand[0].y,p_cand[1].y), min(p_cand[0].z,p_cand[1].z))
        new_max = Pnt(max(p_cand[0].x,p_cand[1].x), max(p_cand[0].y,p_cand[1].y), max(p_cand[0].z,p_cand[1].z))
        new_cube = Box(new_min, new_max)
        new_cube.bc("top")
        cube = cube + new_cube.bc("top")
    
    if current_level > 1:
        for i in range(4):
            for j in range(4):
                if i==2 and j==2:
                    cube = GrowOneFourthQuadKochSurface(cube, p_min+(1/2)*vec1+(1/2)*vec2+(1/4)*vec3, (1/4)*vec1, (1/4)*vec2, (1/4)*vec3, current_level-1)
                else:
                    cube = GrowOneFourthQuadKochSurface(cube, p_min+(i/4)*vec1+(j/4)*vec2, (1/4)*vec1, (1/4)*vec2, (1/4)*vec3, current_level-1)
        
        # add fractal structures on the four square surfaces surrounding the new cube
        cube = GrowOneFourthQuadKochSurface(cube, p_min+(1/2)*vec1+(1/2)*vec2, (1/4)*vec1, (1/4)*vec3, (-1/4)*vec2, current_level-1)
        cube = GrowOneFourthQuadKochSurface(cube, p_min+(1/2)*vec1+(3/4)*vec2, (-1/4)*vec2, (1/4)*vec3, (-1/4)*vec1, current_level-1)
        cube = GrowOneFourthQuadKochSurface(cube, p_min+(3/4)*vec1+(3/4)*vec2, (-1/4)*vec1, (1/4)*vec3, (1/4)*vec2, current_level-1)
        cube = GrowOneFourthQuadKochSurface(cube, p_min+(3/4)*vec1+(1/2)*vec2, (1/4)*vec2, (1/4)*vec3, (1/4)*vec1, current_level-1)
    
    return cube
        
def MakeOneFourthQuadKochSurfaceGeo(fractal_level, h_max = 0.2):
    cube =  Box(Pnt(0,0,0), Pnt(1,1,1))
    for i in range(6):
        if cube.faces[i].center[0]==0 or cube.faces[i].center[0]==1 or cube.faces[i].center[1]==0 or cube.faces[i].center[1]==1:
            cube.faces[i].name = 'side'
        elif cube.faces[i].center[2] == 0:
            cube.faces[i].name = 'bottom'
        else:
            cube.faces[i].name = 'top'
    
    cube = GrowOneFourthQuadKochSurface(cube, Vec(0,0,1), Vec(1,0,0), Vec(0,1,0), Vec(0,0,1), fractal_level)
    geo = OCCGeometry(cube)
    mesh = Mesh(geo.GenerateMesh(maxh = h_max))
    
    l = (1/16)**fractal_level
    A_p = (5/4)**fractal_level
    
    return mesh,l,A_p

if __name__ == "__main__":
    import netgen.gui
    
    geotype = input("Type one of the fractal shapes (2d, 2dquad, 2dmidquad, 3d, 3dmid, 3dquad, 3dmidquad, 3d14quad): ")
    
    fractal_level = int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
    if fractal_level<0:
        fractal_level=2
        print("WARNING: Illegal fractal level! We use 2 as the default setting.")
    
    scaling = float(input("Enter the scaling factor (0-1): "))
    if scaling<0 or scaling>1:
        scaling=1
        print("WARNING: Illegal scaling factor! We use 1 as the default setting.")
    
    h_max = float(input("Enter the allowed maximum diameter of the element: "))
    if h_max<0:
        h_max=1
        print("WARNING: Illegal h max! We use 1 as the default setting.")
    
    mesh, l, A_p = MakeGeometry(geotype, fractal_level, scaling, h_max)
    
    print("geotype:", geotype)
    print("fractal level:", fractal_level)
    print("scaling:", scaling)
    print("h max:", h_max)
    print("l:",l)
    print("(d-1)-dim area of the fractal:", A_p)
    Draw(mesh)