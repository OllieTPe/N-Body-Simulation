# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

G = 1

delta_t = 0.01

mass_vector = np.array([1.0,1.0,1.0])

position_vector = np.array([[-0.97000436,0.24308753,0.0],
                            [0.97000436,-0.24308753,0.0],
                            [0.0,0.0,0.0]
                            ])

velocity_vector = np.array([[0.466203685,0.432365730,0.0],
                            [0.466203685,0.432365730,0.0],
                            [-0.932407370,-0.864731460,0.0]
                            ])



@njit
def acceleration(G, mass_vector, position_vector):
    
    acceleration_vector = np.zeros((len(mass_vector), 3))
    
    for i in range(len(mass_vector)):
        for j in range(len(mass_vector)):
            if i != j:
                displacement = position_vector[j] - position_vector[i]
                displacement_squared = np.dot(displacement,displacement)
                acceleration_vector[i] += G * mass_vector[j] * (displacement) / displacement_squared**1.5
    
    return acceleration_vector

@njit
def forward_euler_method(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    acceleration_vector =  acceleration(G,
                                        mass_vector,
                                        position_vector)
    
    position_vector = position_vector + velocity_vector * delta_t
    
    velocity_vector = velocity_vector + acceleration_vector * delta_t
    
    return position_vector, velocity_vector

@njit
def runge_kutta_4_method(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    k_1_velocity = velocity_vector
    k_1_acceleration = acceleration(G,
                                    mass_vector,
                                    position_vector
                                    )

    
    k_2_velocity = velocity_vector + delta_t / 2 * k_1_acceleration
    k_2_acceleration = acceleration(G,
                                    mass_vector,
                                    position_vector + delta_t / 2 * k_1_velocity
                                    )


    k_3_velocity = velocity_vector + delta_t / 2 * k_2_acceleration
    k_3_acceleration = acceleration(G,
                                    mass_vector,
                                    position_vector + delta_t / 2 * k_2_velocity
                                    )
    
    k_4_velocity = velocity_vector + delta_t * k_3_acceleration
    k_4_acceleration = acceleration(G,
                                    mass_vector,
                                    position_vector + delta_t * k_3_velocity
                                    )
    
    position_vector = position_vector + delta_t / 6 * (k_1_velocity + 2 * k_2_velocity + 2 * k_3_velocity + k_4_velocity)
    velocity_vector = velocity_vector + delta_t / 6 * (k_1_acceleration + 2 * k_2_acceleration + 2 * k_3_acceleration + k_4_acceleration)

    return position_vector, velocity_vector

@njit
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

#animation = run_animation(G, delta_t, mass_vector, position_vector, velocity_vector)

@njit
def total_energy(G, mass_vector, position_vector, velocity_vector):
    
    kinetic_energy = 0.0
    potential_energy = 0.0
    
    for i in range(len(mass_vector)):
        
        kinetic_energy += (mass_vector[i] * np.dot(velocity_vector[i], velocity_vector[i]) / 2)

    for i in range(len(mass_vector)):
        for j in range(i + 1, len(mass_vector)):
            
            displacement = position_vector[j] - position_vector[i]
            distance = np.sqrt(np.dot(displacement, displacement))
            
            potential_energy -= G * mass_vector[i] * mass_vector[j] / distance
            
    return kinetic_energy + potential_energy

@njit
def total_linear_momentum(mass_vector, position_vector, velocity_vector):
    
    linear_momentum = np.zeros(3)
    
    for i in range(len(mass_vector)):
        
        linear_momentum += mass_vector[i] * velocity_vector[i]
        
    return linear_momentum

@njit
def total_angular_momentum(mass_vector, position_vector, velocity_vector):
    
    angular_momentum = np.zeros(3)
    
    for i in range(len(mass_vector)):
        
        angular_momentum += np.cross(position_vector[i], mass_vector[i] * velocity_vector[i])
        
    return angular_momentum

@njit
def conservation_of_energy(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    number_of_measurements = number_of_iterations // iterations_between_updates + 1
    
    energy_error = np.empty(number_of_measurements)
    time = np.empty(number_of_measurements)
    
    energy_error[0] = 0.0
    time[0] = 0.0
    
    E_0 = total_energy(G,
                       mass_vector,
                       position_vector,
                       velocity_vector
                       )
    
    measurement = 1
    
    for i in range(number_of_iterations):
        
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )
        
        if (i + 1) % iterations_between_updates == 0:
        
            time[measurement] = (i + 1) * delta_t
        
            energy = total_energy(G,
                                  mass_vector,
                                  position_vector,
                                  velocity_vector
                                  )
        
            if abs(E_0) < 1e-12:
                energy_error[measurement] = abs(energy - E_0)
            else:
                energy_error[measurement] = (energy - E_0) / abs(E_0)
            
            measurement += 1
        
    return time, energy_error

@njit
def conservation_of_linear_momentum(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    number_of_measurements = number_of_iterations // iterations_between_updates + 1
    
    linear_momentum_error = np.empty(number_of_measurements)
    time = np.empty(number_of_measurements)
    
    linear_momentum_error[0] = 0.0
    time[0] = 0.0
    
    P_0 = total_linear_momentum(mass_vector,
                                position_vector,
                                velocity_vector
                                )
    
    P_0_norm = np.linalg.norm(P_0)
    
    measurement = 1
    
    for i in range(number_of_iterations):
        
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )
        
        if (i + 1) % iterations_between_updates == 0:
            
            time[measurement] = (i + 1) * delta_t
            
            linear_momentum = total_linear_momentum(G,
                                                    mass_vector,
                                                    position_vector,
                                                    velocity_vector
                                                    )
        
            if P_0_norm < 1e-12:
                linear_momentum_error[measurement] = np.linalg.norm(linear_momentum - P_0)
            else:
                linear_momentum_error[measurement] = np.linalg.norm(linear_momentum - P_0) / P_0_norm
                
            measurement += 1
            
    return time, linear_momentum_error

@njit
def conservation_of_angular_momentum(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    number_of_measurements = number_of_iterations // iterations_between_updates + 1
    
    angular_momentum_error = np.empty(number_of_measurements)
    time = np.empty(number_of_measurements)
    
    angular_momentum_error[0] = 0.0
    time[0] = 0.0
    
    L_0 = total_angular_momentum(mass_vector,
                                 position_vector,
                                 velocity_vector
                                 )
    
    L_0_norm = np.linalg.norm(L_0)
    
    measurement = 1
    
    for i in range(number_of_iterations):
        
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )
        
        if (i + 1) % iterations_between_updates == 0:
            
            time[measurement] = (i + 1) * delta_t
            
            angular_momentum = total_angular_momentum(G,
                                                      mass_vector,
                                                      position_vector,
                                                      velocity_vector
                                                      )
            if L_0_norm < 1e-12:
                angular_momentum_error[measurement] = np.linalg.norm(angular_momentum - L_0)
            else:
                angular_momentum_error[measurement] = np.linalg.norm(angular_momentum - L_0) / L_0_norm
                
            measurement += 1
            
    return time, angular_momentum_error


def energy_conservation_comparison(number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    methods = [runge_kutta_4_method]
    
    names = ["Runge Kutta"]
        
    for method, name in zip(methods, names):
        
        time, energy_error = conservation_of_energy(method,
                                                    number_of_iterations,
                                                    iterations_between_updates,
                                                    G,
                                                    delta_t,
                                                    mass_vector,
                                                    position_vector,
                                                    velocity_vector
                                                    )

        plt.plot(time, np.abs(energy_error), label = name)
    
    plt.xlabel("Time")
    plt.ylabel("Absolute Relative Energy Error")
    plt.yscale("log")
    plt.title(f"Energy Conservation with Δt = {delta_t}")
    plt.legend()
    plt.grid()
    plt.show()

energy_conservation_comparison(250000, 20, G, 0.05, mass_vector, position_vector, velocity_vector)












