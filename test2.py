# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:33:09 2020

@author: Fabio
"""

# TEST TRIANGLES

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import compute_intersection as cint

#t1=np.array([[2,2],[7,2],[4,5]]) 
t1=np.array([[4,2],[6,4.5],[2,4]])
t2=np.array([[2,3.5],[5,1],[4,3]])
#t2=np.array([[2,1],[6,1],[4,2]])
#t2=np.array([[2,1],[7,1],[4,3]])

pinter=cint.triangle_intersection(t1,t2)

# Plot of triangle 1
cint.draw_triangle(t1,'r')
# Plot of triangle 2
cint.draw_triangle(t2,'g')
# Plot of intersection point
cint.draw_polygon(pinter,'b')
plt.show()
plt.title('Intersection')
