# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np

G = 1

m1 = 50
m2 = 50

r1 = np.array([1,1])
r2 = np.array([-1,1])

displacement = r2-r1
abs_displacement = np.linalg.norm(displacement)

a1 = G*m2*(r2-r1)/abs_displacement**3
a2 = G*m1*(r1-r2)/abs_displacement**3

print(a1,a2)
