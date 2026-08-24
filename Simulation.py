# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np
import matplotlib.pyplot as plt

#The gravitational constant
G = 1

#Discrete time step
delta_t = 0.01

#The masses of our two massive objects
mass_vector = [30,60]

#The initial displacement vectors of our two massive objects
dis_vector = [np.array([1,0]), np.array([-1,0])]

#The initial velocity vectors of our two massive objects
vel_vector = [np.array([0,3]), np.array([0,-3])]

#Function that updates the displacement and velocity vectors
def forward_euler_method(G,delta_t,mass_vector,dis_vector,vel_vector):
    
    #Calculates the absolute displacement between massive objects
    abs_displacement = np.linalg.norm(dis_vector[1]-dis_vector[0])
    
    #The calculations for the acceleration vectors for our two massive objects
    acc_vector = [G*mass_vector[0]*(dis_vector[1]-dis_vector[0])/abs_displacement**3, G*mass_vector[1]*(dis_vector[0]-dis_vector[1])/abs_displacement**3]
    
    #Using the finite difference method to find the displacement vectors after some time step
    dis_vector = [dis_vector[0] + vel_vector[0]*delta_t, dis_vector[1] + vel_vector[1]*delta_t]
    
    #Using the finite difference method to find the velocity vectors after some time step
    vel_vector = [vel_vector[0] + acc_vector[0]*delta_t, vel_vector[1] + acc_vector[1]*delta_t]

    return dis_vector, vel_vector

#The following code stores the values of each massive objects trajectory and plots the path by both objects
trajectory1 = []
trajectory2 = []

for i in range(600):
    dis_vector, vel_vector = forward_euler_method(G, delta_t, mass_vector, dis_vector, vel_vector)
    trajectory1.append(dis_vector[0].copy())
    trajectory2.append(dis_vector[1].copy())

trajectory1 = np.array(trajectory1)
trajectory2 = np.array(trajectory2)

plt.plot(trajectory1[:,0],trajectory1[:,1])
plt.plot(trajectory2[:,0],trajectory2[:,1])

plt.axis("equal")
plt.show()