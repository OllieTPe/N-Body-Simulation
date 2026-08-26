# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#The gravitational constant
G = 1

#The time between each update
delta_t = 0.01

#The masses of the massive objects
mass_vector = [30,60]

#The initial position vectors of our two massive objects
position_vector = [np.array([1,0,0]), np.array([-1,0,0])]

#The initial velocity vectors of our two massive objects
velocity_vector = [np.array([0,3,0]), np.array([0,-3,0])]



#Function that updates the position and velocity vectors
def forward_euler_method(G, delta_t, mass_vector, position_vector, velocity_vector):

    #Calculates the absolute position between massive objects
    abs_displacement = np.linalg.norm(position_vector[1] - position_vector[0])
    
    #The calculations for the acceleration vectors for our two massive objects
    acceleration_vector = [G * mass_vector[1] * (position_vector[1] - position_vector[0]) / abs_displacement**3, G*mass_vector[0] * (position_vector[0] - position_vector[1]) / abs_displacement**3]
    
    #Using the finite difference method to find the position vectors after some time step
    position_vector = [position_vector[0] + velocity_vector[0] * delta_t, position_vector[1] + velocity_vector[1] * delta_t]
    
    #Using the finite difference method to find the velocity vectors after some time step
    velocity_vector = [velocity_vector[0] + acceleration_vector[0] * delta_t, velocity_vector[1] + acceleration_vector[1] * delta_t]

    return position_vector, velocity_vector

#The following code stores the values of each massive objects trajectory and plots the path by both objects
def run_animation(G, delta_t, mass_vector, position_vector, velocity_vector):
    trajectory1 = []
    trajectory2 = []

    for i in range(1000):
        position_vector, velocity_vector = forward_euler_method(G, delta_t, mass_vector, position_vector, velocity_vector)
        trajectory1.append(position_vector[0].copy())
        trajectory2.append(position_vector[1].copy())

    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)

    fig, ax = plt.subplots()

    ax.set_xlim(min(trajectory1[:,0].min(), trajectory2[:,0].min()) - 0.5, max(trajectory1[:,0].max(),trajectory2[:,0].max()) + 0.5)
    ax.set_ylim(min(trajectory1[:,1].min(), trajectory2[:,1].min()) - 0.5, max(trajectory1[:,1].max(),trajectory2[:,1].max()) + 0.5)

    ax.set_aspect("equal")

    body1, = ax.plot([], [], "o")
    body2, = ax.plot([], [], "o")

    def update(frame):
    
        body1.set_data([trajectory1[frame,0]], [trajectory1[frame,1]])
        body2.set_data([trajectory2[frame,0]], [trajectory2[frame,1]])
    
        return body1, body2

    animation = FuncAnimation(fig, update, frames = range(0, len(trajectory1), 2), interval = 20)

    plt.show()
    
    return animation

#animation = run_animation(G, delta_t, mass_vector, position_vector, velocity_vector)


def conservation_of_energy(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    energy_error = []
    
    k_0 = sum(mass_vector[i] * np.linalg.norm(velocity_vector[i])**2 / 2 for i in range(len(mass_vector)))
    v_0 = sum(-1* G * mass_vector[i] * mass_vector[j] / np.linalg.norm(position_vector[j] - position_vector[i]) for i in range(len(mass_vector)) for j in range(len(mass_vector)) if i < j)
    E_0 = k_0 + v_0
    
    energy_error.append(np.float64(0))
    
    for i in range(number_of_iterations):
        position_vector, velocity_vector = method(G, delta_t, mass_vector, position_vector, velocity_vector)
        
        kinetic_energy = sum(mass_vector[i] * np.linalg.norm(velocity_vector[i])**2 / 2 for i in range(len(mass_vector)))
        potential_energy = sum(-1* G * mass_vector[i] * mass_vector[j] / np.linalg.norm(position_vector[j] - position_vector[i]) for i in range(len(mass_vector)) for j in range(len(mass_vector)) if i < j)
        
        if np.isclose(E_0, 0, 1e-12):
            energy_error.append(abs(kinetic_energy + potential_energy))
            plt.ylabel("Absolute Energy Error")
        else:
            energy_error.append((kinetic_energy + potential_energy - E_0) / abs(E_0))
            plt.ylabel("Relative Energy Error")

    time = np.arange(len(energy_error)) * delta_t
    
    plt.plot(time, energy_error)

    plt.xlabel("Time")
    plt.title("Energy Conservation Using The Forward Euler Method")

    plt.show()

def conservation_of_linear_momentum(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    linear_momentum_error = []
    
    P_0 = sum(mass_vector[i] * velocity_vector[i] for i in range(len(mass_vector)))

    linear_momentum_error.append(np.float64(0))
    
    for i in range(number_of_iterations):
        position_vector, velocity_vector = method(G, delta_t, mass_vector, position_vector, velocity_vector)
        
        linear_momentum = sum(mass_vector[i] * velocity_vector[i] for i in range(len(mass_vector)))
        
        if np.isclose(np.linalg.norm(P_0), 0, atol = 1e-12):
            linear_momentum_error.append(np.linalg.norm(linear_momentum - P_0))
            plt.ylabel("Absolute Linear Momentum Error")
        else:
            linear_momentum_error.append(np.linalg.norm(linear_momentum - P_0) / np.linalg.norm(P_0))
            plt.ylabel("Relative Linear Momentum Error")
    
    time = np.arange(len(linear_momentum_error)) * delta_t
    
    plt.plot(time, linear_momentum_error)
    
    plt.xlabel("Time")
    plt.title("Linear Momentum Conservation Using The Forward Euler Method")

    plt.show()
    
def conservation_of_angular_momentum(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    angular_momentum_error = []
    
    L_0 = sum(np.cross(position_vector[i], mass_vector[i] * velocity_vector[i]) for i in range(len(mass_vector)))
    
    angular_momentum_error.append(np.float64(0))
    
    for i in range(number_of_iterations):
        position_vector, velocity_vector = method(G, delta_t, mass_vector, position_vector, velocity_vector)
        
        angular_momentum = sum(np.cross(position_vector[i], mass_vector[i] * velocity_vector[i]) for i in range(len(mass_vector)))
        
        if np.isclose(np.linalg.norm(L_0), 0, atol = 1e-12):
            angular_momentum_error.append(np.linalg.norm(angular_momentum - L_0))
            plt.ylabel("Absolute Angular Momentum Error")
        else:
            angular_momentum_error.append(np.linalg.norm(angular_momentum - L_0) / np.linalg.norm(L_0))
            plt.ylabel("Relative Angular Momentum Error")
        
    time = np.arange(len(angular_momentum_error)) * delta_t
    
    plt.plot(time, angular_momentum_error)
    
    plt.xlabel("Time")
    plt.title("Angular Momentum Conservation Using The Forward Euler Method")
    
    plt.show()
    
conservation_of_angular_momentum(forward_euler_method, 10000, G, 0.0001, mass_vector, position_vector, velocity_vector)
    





