import numpy as np
import unittest

from compute_intersection import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TestMesh(unittest.TestCase):

    
    def test_transfix_configuration(self):

        expected_points = np.array([[0.2, 0.2, 0. ],
                                    [0.2, 0.2, 0.5],
                                    [0.2, 0.7, 0. ],
                                    [0.7, 0.2, 0. ]])
        
        t1=np.array([[0.,0.,0.],   
                     [1.,0.,0.],
                     [0.,1.,0.],
                     [0.,0.,1.]])  
   
        t2=np.array([[0.2,0.2,-0.5],
                     [1.2,0.2,-0.5],
                     [0.2,1.2,-0.5],
                     [0.2,0.2, 0.5]])
        
        intersection_points=tetrahedra_intersection(t1,t2)
        
        for p1,p2 in zip(expected_points,intersection_points):
            self.assertTrue(np.array_equal(p1,p2))
    
    def test_tangent_star_configuration(self):

        expected_points = np.array([[0.  , 0.  , 0. ],
                                    [0.  , 0.  , 0.5],
                                    [0.  , 0.25, 0. ],
                                    [0.  , 0.25, 0.5],
                                    [0.25, 0.  , 0. ],
                                    [0.25, 0.  , 0.5],
                                    [0.25, 0.25, 0. ],
                                    [0.25, 0.25, 0.5]])

        t1=np.array([[0.,0.,0.],   
                     [1.,0.,0.],
                     [0.,1.,0.],
                     [0.,0.,1.]])
      
        t2=np.array([[ 0.25,-0.75, 0.5],
                     [ 0.25, 0.25, 0.5],
                     [-0.75, 0.25, 0.5],
                     [ 0.25, 0.25,-0.5]])

        intersection_points=tetrahedra_intersection(t1,t2)
        
        for p1,p2 in zip(expected_points,intersection_points):
            self.assertTrue(np.array_equal(p1,p2))
    
    def test_star_configuration(self): 

        expected_points = np.array([[0.  , 0.25, 0.25],
                                    [0.  , 0.5 , 0.  ],
                                    [0.  , 0.5 , 0.25],
                                    [0.25, 0.  , 0.25],
                                    [0.25, 0.5 , 0.25],
                                    [0.5 , 0.  , 0.  ],
                                    [0.5 , 0.5 , 0.  ],
                                    [0.75, 0.  , 0.  ],
                                    [0.75, 0.  , 0.25],
                                    [0.75, 0.25, 0.  ]])

        t1=np.array([[0.,0.,0.],   
                     [1.,0.,0.],
                     [0.,1.,0.],
                     [0.,0.,1.]])
     
        t2=np.array([[ 0.75,-0.5, 0.25],
                     [ 0.75, 0.5, 0.25],
                     [-0.25, 0.5, 0.25],
                     [ 0.75, 0.5,-0.75]])

        intersection_points=tetrahedra_intersection(t1,t2)

        for p1,p2 in zip(expected_points,intersection_points):
            self.assertTrue(np.array_equal(p1,p2))
    

if __name__ == "__main__":

    unittest.main()

    #print(mesh.tetra_ids)
    #print(mesh.nodes_ids)
    #print(mesh.faces_ids)
    #print(mesh.edges_ids)
