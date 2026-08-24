# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np

#The gravitational constant
G = 1

#Discrete time step
delta_t = 0.01

#The masses of our two massive objects
mass_vector = [50,50]

#The initial position vectors of our two massive objects
dis_vector = [np.array([1,0]), np.array([-1,0])]

#The initial velocity vectors of our two massive objects
vel_vector = [np.array([0,0]), np.array([0,0])]

def forward_euler_method(G,delta_t,mass_vector,dis_vector,vel_vector):
    
    #Calculates the absolute displacement between massive objects
    abs_displacement = np.linalg.norm(dis_vector[1]-dis_vector[0])
    
    #The calculations for the acceleration vectors for our two massive objects
    acc_vector = [G*mass_vector[0]*(dis_vector[1]-dis_vector[0])/abs_displacement**3, G*mass_vector[1]*(dis_vector[0]-dis_vector[1])/abs_displacement**3]
    
    #Using the finite difference method to find the position vectors after some time step
    dis_vector = [dis_vector[0] + vel_vector[0]*delta_t, dis_vector[1] + vel_vector[1]*delta_t]
    
    #Using the finite difference method to find the velocity vectors after some time step
    vel_vector = [vel_vector[0] + acc_vector[0]*delta_t, vel_vector[1] + acc_vector[1]*delta_t]

    return dis_vector, vel_vector

for i in range(1000):
    dis_vector, vel_vector = forward_euler_method(G, delta_t, mass_vector, dis_vector, vel_vector)
    
print(dis_vector,vel_vector)
