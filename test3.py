#----------------------#
#- PLOT SOME EXAMPLES -#
#----------------------#
   
import numpy as np
from compute_intersection import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def example1(): # transfix

    t1=np.array([[0.,0.,0.],   
                 [1.,0.,0.],
                 [0.,1.,0.],
                 [0.,0.,1.]])     

    t2=np.array([[0.2,0.2,-0.5],
                 [1.2,0.2,-0.5],
                 [0.2,1.2,-0.5],
                 [0.2,0.2, 0.5]])

    intersection_points=tetrahedra_intersection(t1,t2)

    plt.figure()
    ax=plt.axes(projection='3d')
    draw_tetrahedron(t1,'r')
    draw_tetrahedron(t2,'g')
    draw_polyhedron(intersection_points,'b')
    plt.title('Example 1')

def example2(): # tangent star

    t1=np.array([[0.,0.,0.],   
                 [1.,0.,0.],
                 [0.,1.,0.],
                 [0.,0.,1.]])
    
    t2=np.array([[ 0.25,-0.75, 0.5],
                 [ 0.25, 0.25, 0.5],
                 [-0.75, 0.25, 0.5],
                 [ 0.25, 0.25,-0.5]])

    intersection_points=tetrahedra_intersection(t1,t2)

    plt.figure()
    ax=plt.axes(projection='3d')
    draw_tetrahedron(t1,'r')
    draw_tetrahedron(t2,'g')
    draw_polyhedron(intersection_points,'b')
    plt.title('Example 2')

def example3(): # star

    t1=np.array([[0.,0.,0.],   
                 [1.,0.,0.],
                 [0.,1.,0.],
                 [0.,0.,1.]])
     
    t2=np.array([[ 0.75,-0.5, 0.25],
                 [ 0.75, 0.5, 0.25],
                 [-0.25, 0.5, 0.25],
                 [ 0.75, 0.5,-0.75]])

    intersection_points=tetrahedra_intersection(t1,t2)

    plt.figure()
    ax=plt.axes(projection='3d')
    draw_tetrahedron(t1,'r')
    draw_tetrahedron(t2,'g')
    draw_polyhedron(intersection_points,'b')
    plt.title('Example 3')

if __name__ == "__main__":
    example1()
    example2()
    example3()
    plt.show()
