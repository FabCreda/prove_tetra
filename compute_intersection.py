# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:45:15 2020

@author: Fabio
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull  

####################### 2D FUNCTIONS ##################################

def edge_equation(p1,p2):
    
    '''
    STRAIGHT LINE PASSING THROUGH TWO POINTS
    ax + by + c = 0
    
    Input: p1=(x,y), p2=(x,y)
    Output: Coefficients of the straight line passing through p1 and p2
        
    '''
    
    a=p1[1]-p2[1]
    b=p2[0]-p1[0]
    c=p1[0]*p2[1]-p1[1]*p2[0]
    
    return [a,b,c]

def edges_eq_matrix(t):

    '''
    This function receives a triangle in input and returns a np.array
    with dimension 3x3: on each row there are coefficients of the equation of a side
    
    Output:   [ a, b, c     side 1
                a, b, c     side 2
                a, b, c ]   side 3
              
    '''
    eq=np.zeros((3,3))
    eq[0]=edge_equation(t[0],t[1])
    eq[1]=edge_equation(t[1],t[2])
    eq[2]=edge_equation(t[2],t[0])
    return eq

def compute_equation(eq,p,v):
    
    '''
    COEFFICIENT FOR CONSTRUCTING THE INTERSECTION POINT
    
    Input:  eq = sides equation (triangle 2),
            p=(x,y) [vertex of triangle1],
            v = unit vector related to the side (triangle 1)
    Output: coefficient for constructing the intersection point
    '''
    num=np.dot(eq[[0,1]],p)+eq[2]
    den=np.dot(eq[[0,1]],v)

    k=-num/den

    return k

def sign(eq,p):
    
    '''
    POSITION OF POINT P IN RELATION WITH A STRAIGHT LINE
    
    Input:  eq = straight line equation, 
            p=(x,y)
    Output: sign=0 if p is on the line
            sign=1 if p is in a demiplane
            sign=-1 if p is in the opposite demiplane
    '''
    
    tol=10**(-8)

    s=eq[0]*p[0]+eq[1]*p[1]+eq[2]
    
    if abs(s)<tol:
        s=0
    else:
        if s>0:
            s=1
        else:
            s=-1
    
    return s

def internal_control(t,eq,p):
    
    '''
    POINT P INTERNAL OR EXTERNAL TO A TRIANGLE
    
    Input:  t = triangle expressed by its vertices
            eq = equations of sides
            p = (x,y)
    Ouput:  1 if p is in the triangle
            0 if p is out of the triangle
    '''

    s0=sign(eq[0],t[2])*sign(eq[0],p)
    s1=sign(eq[1],t[0])*sign(eq[1],p)
    s2=sign(eq[2],t[1])*sign(eq[2],p)
    
    if s0>0 and s1>0 and s2>0:
        # p is in the triangle
        return 1 
    elif s0==0 or s1==0 or s2==0:
        return 0
    else:
        # p is out of the triangle
        return -1


def segments_control(tri,p):
    
    '''
    We find points of intersection using sides equations:
    a point P can be an intersection point for the straight lines passing through sides and not for sides themselves.
    This function controls that a point is really the intersection between two sides of two triangles
    and not for related straight lines.
    For do this, we see sides as parameterized segments.
    
    --------------------------------------------------------------------------------------------------------
    EXAMPLE.
    We have segment AB and we have to test if P is on the segment.
    Parametrization of AB:
    x = xA + t(xB-xA)
    y = yA + t(yB-yA)
    with t in [0,1]
    so
        (x-xA) / (xB-xA)  = (y-yA) / (yB-yA) = t
    If the equality is true and 0<=t<=1 ==> P is really the intersection between two sides.
    --------------------------------------------------------------------------------------------------------
    
    The function use only a triangle because we already know that p belongs to the other triangle.
        
    '''

    tol=10**(-8)
    flag=0
    
    # We control if p is on t2[1]-t2[0]
    sx=(p[0]-tri[0,0])/(tri[1,0]-tri[0,0])
    sy=(p[1]-tri[0,1])/(tri[1,1]-tri[0,1])
    if abs(sx-sy)<tol and (0-tol)<=sx<=(1+tol):
        flag=1
        
    # We control if p is on t2[2]-t2[1]
    sx=(p[0]-tri[1,0])/(tri[2,0]-tri[1,0])
    sy=(p[1]-tri[1,1])/(tri[2,1]-tri[1,1])
    if abs(sx-sy)<tol and (0-tol)<=sx<=(1+tol):
        flag=1
        
    # We control if p is on t2[0]-t2[2]
    sx=(p[0]-tri[2,0])/(tri[0,0]-tri[2,0])
    sy=(p[1]-tri[2,1])/(tri[0,1]-tri[2,1])
    if abs(sx-sy)<tol and (0-tol)<=sx<=(1+tol):
        flag=1
    
    return flag

def vertices_order(pp):
    
    '''
    This function orders polygon vertices in counterclockwise
    
    Input: np.array with dimension nx2
    Output: np.array with dimension nx2 with points order in counterclockwise
    
    Idea:
    - We put at index=0 the point with minimum y (reference point A)
    - We slide on couples of remaining points (P,Q) and we construct the associated triangle matrix (A,P,Q)
    - if det<0 ==> we have to swap the position of P and Q 
    
    '''
    # Index of the point with minimum y
    minindex=np.argmin(pp[:,1])
    
    if minindex!=0:
        # We swap position between first point and min(y)-point
        pp[[0,minindex]]=[[pp[minindex],pp[0]]] # A
    
    # Loops on points
    for j in range(1,np.size(pp,0)): # P
        for k in range(j+1,np.size(pp,0)): # Q
            
            # Matrix (A,P,Q)
            A=np.array([[pp[0,0],pp[j,0],pp[k,0]],[pp[0,1],pp[j,1],pp[k,1]],[1,1,1]])
            # Determinant control
            if np.linalg.det(A)<0:
                pp[[j,k]]=[[pp[k],pp[j]]]
                
    return pp

def triangle_intersection(t1,t2):
    
    '''
    VERTICES OF THE POLYGON OBTAINED BY INTERSECTING TWO TRIANGLES 
    '''
    
    # Edges equations of triangle 2 (we need them to find intersection points)
    eq2=edges_eq_matrix(t2)
    # Edges equations of triangle 1 (we need them only for the internal control)
    eq1=edges_eq_matrix(t1)
    
    # Unit vector of edges of triangle 1
    v=np.zeros((3,2))
    v[0]=t1[1]-t1[0]
    v[1]=t1[2]-t1[1]
    v[2]=t1[0]-t1[2]

    c=0
    pwork=np.zeros((9,2)) # We have at most 3x3=9 candidate intersection point
    
    for i in range(0,3):
    
        # Loop on vertices and unit vectors of triangle 1
        p=t1[i] # Vertex
        v_i=v[i] # Relative unit vector that starts on p
        length=np.linalg.norm(v_i)
    
        for e in eq2: # Loop on edges of triangle 2 using their equation
        
            k=compute_equation(e,p,v_i) # Coefficient for constructing the (candidate) intersection point
            new_p=v_i*k+p # Candidate intersection point
        
            if k>0 and np.linalg.norm(new_p-p)<=length: # Condition of "goodness" of the point
            
                flag=segments_control(t2,new_p)
                # We have to control that the point is really the intersection of two edges
                # and not only of the relative straight line
                if flag==1:
                    pwork[c]=new_p # We select only good points
                    c=c+1

    # We have to delete the useless tuple                
    if c>1: # If c==1, we have only an intersection point, so the intersection is degenerate
        pinter=pwork[0:c][:]

    # We have also to select vertices that are internal to the other triangle
    # because they are vertices of the resulting polygon
    if c==2 or c==3 or c==4:
    
        for p in t1: # We control if triangle 1 vertices are internal to triangle 2
            if internal_control(t2,eq2,p)==1:
                pinter=np.append(pinter,[p],axis=0)
    
        for p in t2: # vice versa
            if internal_control(t1,eq1,p)==1:
                pinter=np.append(pinter,[p],axis=0)

    # This function returns vertices in counterclockwise
    return vertices_order(pinter)

    
def draw_triangle(tri, color='r'):
    plt.plot([tri[0,0],tri[1,0]],[tri[0,1],tri[1,1]],color)
    plt.plot([tri[2,0],tri[1,0]],[tri[2,1],tri[1,1]],color)
    plt.plot([tri[0,0],tri[2,0]],[tri[0,1],tri[2,1]],color)

def draw_polygon(pinter,color='b'):

    for i in range(0,pinter.shape[0]):
        # Vertices
        plt.plot(pinter[i,0],pinter[i,1],'*'+color)
        
        # Sides
        if i==pinter.shape[0]-1:
            plt.plot([pinter[i,0],pinter[0,0]],[pinter[i,1],pinter[0,1]],color)
        else:
            plt.plot([pinter[i,0],pinter[i+1,0]],[pinter[i,1],pinter[i+1,1]],color)


####################### 3D FUNCTIONS ##################################

def face_equation(p1,p2,p3):

    '''
    PLANE PASSING THROUGH THREE POINTS
    ax + by + cz + d = 0
    
    Input: p1=(x,y,z), p2=(x,y,z), p3=(x,y,z)
    Output: Coefficients of the plane passing through p1, p2 and p3
        
    '''

    coeff=np.cross(p2-p1,p3-p1) #[a,b,c]
    d=-np.dot(coeff,p1)
    return np.append(coeff,d) #[a,b,c,d]
    # Test: print(face_equation([3,7,8],[1,0,-1],[1,2,3])). Result:[-10, 8, -4, 6] OK

def faces_eq_matrix(t):
    
    '''
    This function receives a tetrahedron in input and returns a np.array
    with dimension 4x4: on each row there are coefficients of the equation of a face
    
    Output:   [ a, b, c, d     face 1
                a, b, c, d     face 2
                a, b, c, d     face 3
                a, b, c, d ]   face 4
              
    '''
    eq=np.zeros((4,4))
    eq[0]=face_equation(t[0],t[1],t[2])
    eq[1]=face_equation(t[0],t[1],t[3])
    eq[2]=face_equation(t[1],t[2],t[3])
    eq[3]=face_equation(t[0],t[2],t[3])

    return eq

def unit_vectors(t1):
    
    '''
    Tetrahedron= [A B C D]
    This function computes the unit vector associated with each edge:
    v contains unit vectors related to AB, BC, CA in counterclockwise
    w the unit vectors related with the edges AD, BD, CD
    '''
    v=np.zeros((3,3))
    v[0]=t1[1]-t1[0]
    v[1]=t1[2]-t1[1]
    v[2]=t1[0]-t1[2]
    w=np.zeros((3,3))
    w[0]=t1[3]-t1[0]
    w[1]=t1[3]-t1[1]
    w[2]=t1[3]-t1[2]

    return v,w
    
def compute_equation3(eq,p,v):

    '''
    COEFFICIENT FOR CONSTRUCTING THE INTERSECTION POINT
    
    Input:  eq = sides equation (tetra.2),
            p=(x,y,z) [vertex of tetra.1],
            v = unit vector related to the edge (tetra.1)
    Output: coefficient for constructing the intersection point
    '''

    num=np.dot(eq[[0,1,2]],p)+eq[3]
    den=np.dot(eq[[0,1,2]],v)
    
    k=-num/den
    
    return k

def sign3(eq,p):

    '''
    POSITION OF POINT P IN RELATION WITH A PLANE
    
    Input:  eq = plane equation, 
            p=(x,y,z)
    Output: sign=0 if p is on the plane
            sign=1 if p is in a halfspace
            sign=-1 if p is in the opposite halfspace
    '''

    tol=10**(-8)
    
    s=np.dot(eq[[0,1,2]],p)+eq[3]
    
    if abs(s)<tol:
        s=0
    else:
        if s>0:
            s=1
        else:
            s=-1

    return s

def internal_control3(t,eq,p):

    '''
    POINT P INTERNAL OR EXTERNAL TO A TETRAHEDRON
    
    Input:  t = tetrahedron expressed by its vertices
            eq = equations of faces
            p = (x,y,z)
    Ouput:  1 if p is in the tetra.
            0 if p is out of the tetra.
    '''

    s0=sign3(eq[0],t[3])*sign3(eq[0],p)
    s1=sign3(eq[1],t[2])*sign3(eq[1],p)
    s2=sign3(eq[2],t[0])*sign3(eq[2],p)
    s3=sign3(eq[3],t[1])*sign3(eq[3],p)

    if s0>0 and s1>0 and s2>0 and s3>0:
        return 1
    else:
        return 0

def faces_control(tetr,eq,p):

    '''
    This function controls if a point p is really on a face of a tetra. 
    and not only on the plane that contains it.

    Input:  tetr = tetrahedron
            eq = equations of the four faces
            p = point that we have to test

    Output: 1 if the point is on a face
            0 if the point is not on a face
    '''

    flag=0
    tol=10**(-8)

    for i in range(0,4): # Loop on faces
        
        # Evaluation of the i-th face equation in p (p is on the i-th plane?)
        val=np.dot(eq[i,[0,1,2]],p)+eq[i,3]
        
        if abs(val)<tol: 
        # if p is on the plane, we have to control if p is in the triangle=face
        # (we have to consider each face individually because we have to create the triangle with its corresponding points)
        
            if i==0:
                # First, we control if the face is on a vertical plane parallel with x=0
                if p[0]==tetr[0,0]==tetr[1,0]==tetr[2,0] and min(tetr[[0,1,2],1])<=p[1]<=max(tetr[[0,1,2],1]) and min(tetr[[0,1,2],2])<=p[2]<=max(tetr[[0,1,2],2]):
                    # we create the triangle object and then we use the 2D internal_control
                    # to test if p is in the triangle or not
                    tri=np.array([tetr[0,[1,2]],tetr[1,[1,2]],tetr[2,[1,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[1,2]]) # 2D version because we are working on a triangle
                    if flag==0:
                        # We have to control if the point is on an edge
                        flag=segments_control(tri,p[[1,2]])
                    break # When we have tested p with a face is not necessary to control the others because p can be only on one plane
                    # (if p is on two planes ==> p is on an edge ... ok)
                # Then, we control if the face is on a vertical plane parallel with y=0
                elif p[1]==tetr[0,1]==tetr[1,1]==tetr[2,1] and min(tetr[[0,1,2],0])<=p[0]<=max(tetr[[0,1,2],0])and min(tetr[[0,1,2],2])<=p[2]<=max(tetr[[0,1,2],2]):
                    tri=np.array([tetr[0,[0,2]],tetr[1,[0,2]],tetr[2,[0,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[0,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[0,2]])
                    break
                else: # General case
                    tri=np.array([tetr[0],tetr[1],tetr[2]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p)                     
                    if flag==0:
                        flag=segments_control(tri,p)
                    break

            if i==1:
                if p[0]==tetr[0,0]==tetr[1,0]==tetr[3,0] and min(tetr[[0,1,3],1])<=p[1]<=max(tetr[[0,1,3],1]) and min(tetr[[0,1,3],2])<=p[2]<=max(tetr[[0,1,3],2]):
                    tri=np.array([tetr[0,[1,2]],tetr[1,[1,2]],tetr[3,[1,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[1,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[1,2]])
                    break
                elif p[1]==tetr[0,1]==tetr[1,1]==tetr[3,1] and min(tetr[[0,1,3],0])<=p[0]<=max(tetr[[0,1,3],0]) and min(tetr[[0,1,3],2])<=p[2]<=max(tetr[[0,1,3],2]):
                    tri=np.array([tetr[0,[0,2]],tetr[1,[0,2]],tetr[3,[0,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[0,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[0,2]])
                    break
                else:
                    tri=np.array([tetr[0],tetr[1],tetr[3]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p)
                    if flag==0:
                        flag=segments_control(tri,p)
                    break

            if i==2:
                if p[0]==tetr[3,0]==tetr[1,0]==tetr[2,0] and min(tetr[[3,1,2],1])<=p[1]<=max(tetr[[3,1,2],1]) and min(tetr[[3,1,2],2])<=p[2]<=max(tetr[[3,1,2],2]):
                    tri=np.array([tetr[1,[1,2]],tetr[2,[1,2]],tetr[3,[1,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[1,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[1,2]])
                    break
                elif p[1]==tetr[3,1]==tetr[1,1]==tetr[2,1] and min(tetr[[3,1,2],0])<=p[0]<=max(tetr[[3,1,2],0]) and min(tetr[[3,1,2],2])<=p[2]<=max(tetr[[3,1,2],2]):
                    tri=np.array([tetr[1,[0,2]],tetr[2,[0,2]],tetr[3,[0,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[0,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[0,2]])
                    break
                else:
                    tri=np.array([tetr[1],tetr[2],tetr[3]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p)
                    if flag==0:
                        flag=segments_control(tri,p)
                    break

            if i==3:
                if p[0]==tetr[0,0]==tetr[3,0]==tetr[2,0] and min(tetr[[0,3,2],1])<=p[1]<=max(tetr[[0,3,2],1]) and min(tetr[[0,3,2],2])<=p[2]<=max(tetr[[0,3,2],2]):
                    tri=np.array([tetr[0,[1,2]],tetr[2,[1,2]],tetr[3,[1,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[1,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[1,2]])
                    break
                elif p[1]==tetr[0,1]==tetr[3,1]==tetr[2,1] and min(tetr[[0,3,2],0])<=p[0]<=max(tetr[[0,3,2],0]) and min(tetr[[0,3,2],2])<=p[2]<=max(tetr[[0,3,2],2]):
                    tri=np.array([tetr[0,[0,2]],tetr[2,[0,2]],tetr[3,[0,2]]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p[[0,2]])
                    if flag==0:
                        flag=segments_control(tri,p[[0,2]])
                    break
                else:
                    tri=np.array([tetr[0],tetr[2],tetr[3]])
                    tri_eq=edges_eq_matrix(tri)
                    flag=internal_control(tri,tri_eq,p)
                    if flag==0:
                        flag=segments_control(tri,p)
                    break     
            
    return flag

def find_points3(t1,v1,w1,t2,eq2):

    '''
    ALGORITHM TO FIND INTERSECTION POINTS OF TWO TETRA.
    (we have to execute it two times with roles swapped!)
    The idea is the same of the triangles case.
    '''

    pinter=np.zeros((0,3))

    for i in range(0,3):
        # The loop is only on the first three points:
        # with A fixed, we test AB and AD (similary with B and C)
        p=t1[i]
        v_i=v1[i]
        length_v=np.linalg.norm(v_i)
        w_i=w1[i]
        length_w=np.linalg.norm(w_i)
    
        for e in eq2:
        
            kv=compute_equation3(e,p,v_i)
            new_pv=v_i*kv+p 

            kw=compute_equation3(e,p,w_i)
            new_pw=w_i*kw+p

        
            if np.linalg.norm(new_pv-p)<=length_v and kv>0:

                if faces_control(t2,eq2,new_pv)==1:
                    # We have to control if new_p is really an intersection point 
                    # of an edge with a face and not only a point on the plane that contains a face
                    pinter=np.vstack((pinter,new_pv))
            
            if np.linalg.norm(new_pw-p)<=length_w and kw>0:

                if faces_control(t2,eq2,new_pw)==1:            
                    pinter=np.vstack((pinter,new_pw))

    return pinter

def tetrahedra_intersection(t1,t2):

    '''
    VERTICES OF THE POLYHEDRON OBTAINED BY INTERSECTING TWO TETRAHEDRA 
    '''
    
    eq1=faces_eq_matrix(t1) # Equations of faces of tetra.1
    v1,w1=unit_vectors(t1) # Unit vectors of tetra.1

    eq2=faces_eq_matrix(t2) # Equations of faces of tetra.2
    v2,w2=unit_vectors(t2) # Unit vectors of tetra.2
    
    # We have to compute the algorithm to find intersection points
    # (we have to execute it two times with the role of tetra. swapped)
    # (with triangles is not necessary beacuse side=face=edge but face!=edge in dim=3)
    pwork1=find_points3(t1,v1,w1,t2,eq2)
    pwork2=find_points3(t2,v2,w2,t1,eq1)
    pinter=np.vstack((pwork1,pwork2))

    c=np.size(pinter,0) # Number of points found

    # We have to control if a vertex of a tetra. is contained in the other tetra.
    if c>2: # We don't consider degenerate cases
    # c=3 can be degenerate (tetra.1 and tetra.2 have a face in common) or not degenerate

        for p in t1:
            if internal_control3(t2,eq2,p)==1:
                pinter=np.vstack((pinter,p))

        for p in t2:
            if internal_control3(t1,eq1,p)==1:
                pinter=pinter=np.vstack((pinter,p))

    return np.unique(pinter,axis=0)

def draw_tetrahedron(t,color='r'):

    '''
    Calling code must have ax=plt.axes(projection='3d') 
    '''
   
    for i in range(0,4):
        for j in range(i+1,4):
            plt.plot([t[i,0],t[j,0]],[t[i,1],t[j,1]],[t[i,2],t[j,2]],color)

def draw_polyhedron(pinter,color='r'):

    '''
    Calling code must have ax=plt.axes(projection='3d') 
    '''
    
    # Vertices
    for i in range(0,pinter.shape[0]):
        plt.plot([pinter[i,0]],[pinter[i,1]],[pinter[i,2]],'*'+color)
    # Edges
    '''
    hull=ConvexHull(pinter)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        plt.plot(pinter[s, 0], pinter[s, 1], pinter[s, 2], color)
    '''
    

