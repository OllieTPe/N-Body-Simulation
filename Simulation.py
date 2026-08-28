# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1

delta_t = 0.01

mass_vector = np.array([1,1,1])

position_vector = np.array([[-0.97000436,0.24308753,0.0],
                            [0.97000436,-0.24308753,0.0],
                            [0.0,0.0,0.0]
                            ])

velocity_vector = np.array([[0.466203685,0.432365730,0.0],
                            [0.466203685,0.432365730,0.0],
                            [-0.932407370,-0.864731460,0.0]
                            ])




def acceleration(G, mass_vector, position_vector):
    
    acceleration_vector = np.zeros((len(mass_vector), 3))
    
    for i in range(len(mass_vector)):
        for j in range(len(mass_vector)):
            if i != j:
                acceleration_vector[i] += G * mass_vector[j] * (position_vector[j] - position_vector[i]) / np.linalg.norm(position_vector[j] - position_vector[i])**3
    
    return acceleration_vector

def forward_euler_method(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    acceleration_vector =  acceleration(G,
                                        mass_vector,
                                        position_vector)
    
    position_vector = position_vector + velocity_vector * delta_t
    
    velocity_vector = velocity_vector + acceleration_vector * delta_t
    
    return position_vector, velocity_vector

def runge_kutta_4_method(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    k_1 = np.array([
        velocity_vector, 
        acceleration(G,
                     mass_vector,
                     position_vector
                     )
        ])
    
    k_2 = np.array([
        velocity_vector + delta_t / 2 * k_1[1],
        acceleration(G,
                     mass_vector,
                     position_vector + delta_t / 2 * k_1[0]
                     )
        ])

    k_3 = np.array([
        velocity_vector + delta_t / 2 * k_2[1],
        acceleration(G,
                     mass_vector,
                     position_vector + delta_t / 2 * k_2[0]
                     )
        ])
    
    k_4 = np.array([
        velocity_vector + delta_t * k_3[1],
        acceleration(G,
                     mass_vector,
                     position_vector + delta_t * k_3[0]
                     )
        ])
    

    position_vector = position_vector + delta_t / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)[0]
    velocity_vector = velocity_vector + delta_t / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)[1]

    return position_vector, velocity_vector

def leapfrog_method(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    acceleration_vector = acceleration(G,
                                       mass_vector,
                                       position_vector
                                       )
    
    velocity_vector = velocity_vector + delta_t / 2 * acceleration_vector
    
    position_vector = position_vector + delta_t * velocity_vector

    acceleration_vector = acceleration(G,
                                       mass_vector,
                                       position_vector
                                       )
    
    velocity_vector = velocity_vector + delta_t / 2 * acceleration_vector
    
    return position_vector, velocity_vector



def run_animation(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    trajectory = []

    for i in range(10000):
        position_vector, velocity_vector = runge_kutta_4_method(G, delta_t, mass_vector, position_vector, velocity_vector)
        trajectory.append(position_vector.copy())

    trajectory = np.array(trajectory)

    fig, ax = plt.subplots()

    ax.set_xlim(
        trajectory[:, :, 0].min() - 0.5,
        trajectory[:, :, 0].max() + 0.5
        )    
    
    ax.set_ylim(
        trajectory[:, :, 1].min() - 0.5,
        trajectory[:, :, 1].max() + 0.5
        )
    
    ax.set_aspect("equal")

    bodies = []
    
    for i in range(len(mass_vector)):
        body, = ax.plot([], [], "o")
        bodies.append(body)

    def update(frame):
    
        for i in range(len(mass_vector)):
            bodies[i].set_data(
                [trajectory[frame, i, 0]],
                [trajectory[frame, i, 1]]
                )
            
        return bodies

    animation = FuncAnimation(fig, update, frames = range(0, len(trajectory), 5), interval = 20)

    plt.show()
    
    return animation

animation = run_animation(G, delta_t, mass_vector, position_vector, velocity_vector)


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
    
    if method == forward_euler_method:
        title = "Forward Euler Method"
    elif method == runge_kutta_4_method:
        title = "Runge Kutta Method"
    else:
        title = "Leapfrog Method"
    
    plt.xlabel("Time")
    plt.title(f"Angular Momentum Conservation Using The {title}")
    
    plt.show()
    


#conservation_of_energy(leapfrog_method, 1000, G, delta_t, mass_vector, position_vector, velocity_vector)
#conservation_of_linear_momentum(leapfrog_method, 1000, G, delta_t, mass_vector, position_vector, velocity_vector)
#conservation_of_angular_momentum(leapfrog_method, 1000, G, delta_t, mass_vector, position_vector, velocity_vector)



