
# TEST TETRAHEDRA

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import compute_intersection as cint
from scipy.spatial import ConvexHull  

t1=np.array([[2,2,0],[7,2,0],[4,5,0],[5,3,4]])
t2=np.array([[2,4,2],[4,1,2],[6,4.5,2],[3.5,3,-1]])# ok stella
#t2=t1+np.array([[0,0,-2],[0,0,-2],[0,0,-2],[0,0,-2]])# ok infilzo

pinter=cint.tetrahedra_intersection(t1,t2)
print('Punti di intersezione:')
print(pinter)

ax=plt.axes(projection='3d')
cint.draw_tetrahedron(t1,'r')
cint.draw_tetrahedron(t2,'g')
cint.draw_polyhedron(pinter,'b')

plt.show()
'''
plt.figure(2)
ax=plt.axes(projection='3d')
cint.draw_polyhedron(pinter,'b--')
plt.show(2)
'''
