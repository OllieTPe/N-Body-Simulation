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

mass_vector = np.array([1.0,1.0])

position_vector = np.array([[0.0,1.0,0.0],
                            [0.0,-1.0,0.0]
                            ])

velocity_vector = np.array([[0.5,0.0,0.0],
                            [-0.5,0.0,0.0]
                             ])

#mass_vector = np.array([1.0,1.0,1.0])

#position_vector = np.array([[-0.97000436,0.24308753,0.0],
 #                           [0.97000436,-0.24308753,0.0],
  #                          [0.0,0.0,0.0]
   #                         ])

#velocity_vector = np.array([[0.466203685,0.432365730,0.0],
 #                           [0.466203685,0.432365730,0.0],
  #                          [-0.932407370,-0.864731460,0.0]
   #                         ])



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



def run_animation(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    trajectory = []

    for i in range(number_of_iterations):
        position_vector, velocity_vector = method(G, delta_t, mass_vector, position_vector, velocity_vector)
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

#animation = run_animation(leapfrog_method, 10000, G, delta_t, mass_vector, position_vector, velocity_vector)

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
            
            linear_momentum = total_linear_momentum(mass_vector,
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
            
            angular_momentum = total_angular_momentum(mass_vector,
                                                      position_vector,
                                                      velocity_vector
                                                      )
            if L_0_norm < 1e-12:
                angular_momentum_error[measurement] = np.linalg.norm(angular_momentum - L_0)
            else:
                angular_momentum_error[measurement] = np.linalg.norm(angular_momentum - L_0) / L_0_norm
                
            measurement += 1
            
    return time, angular_momentum_error

def plot_graphs(conserved_quantity, total_time, measurement_interval, G, mass_vector, position_vector, velocity_vector):
    
    if conserved_quantity == conservation_of_energy:
        conserved_quantity_name = "Energy"
    elif conserved_quantity == conservation_of_linear_momentum:
        conserved_quantity_name = "Linear Momentum"
    else:
        conserved_quantity_name = "Angular Momentum"
    
    methods = [runge_kutta_4_method, leapfrog_method]
    
    method_names = ["Runge Kutta", "Leapfrog"]
    
    delta_t_times = [0.05, 0.025, 0.0125]
    
    fig, axes = plt.subplots(len(methods), len(delta_t_times), figsize = (9,6), sharex = True, sharey = "row", constrained_layout = True)
    
    for row, (method, method_name) in enumerate(zip(methods, method_names)):
        for column, delta_t in enumerate(delta_t_times):
            
            number_of_iterations = int(total_time / delta_t)
            iterations_between_updates = 1 #int(measurement_interval / delta_t)
            
            time, conserved_quantity_error = conserved_quantity(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector)
            
            ax = axes[row, column]
            
            ax.plot(time, abs(conserved_quantity_error), color = "red")
            ax.set_yscale("log")
            ax.grid(which = "both", alpha = 0.3)
            
            if row == 0:
                ax.set_title(f"Δt = {delta_t}")
            
            if column == 0:
                ax.set_ylabel(f"{method_name} Error")
                
            if row == len(methods) - 1:
                ax.set_xlabel("Time")
            
    fig.suptitle(f"{conserved_quantity_name} Conservation Comparison", fontsize = 16)        
    
    plt.show()

#plot_graphs(conservation_of_energy, 10, 1, G, mass_vector, position_vector, velocity_vector)

def orbit_graph(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    orbit = np.zeros((number_of_iterations + 1, len(mass_vector), 3))
    orbit[0] = position_vector
    
    for i in range(number_of_iterations):
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )
        
        orbit[i + 1] = position_vector.copy()
    
    fig, ax = plt.subplots()    
    
    for body in range(len(mass_vector)):
        ax.plot(orbit[:, body, 0], orbit[:, body, 1], label = f"Body {body + 1}")

    ax.set_xlim(
        orbit[:, :, 0].min() - 0.5,
        orbit[:, :, 0].max() + 0.5
        )    
    
    ax.set_ylim(
        orbit[:, :, 1].min() - 0.5,
        orbit[:, :, 1].max() + 0.5
        )
    
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()
    ax.set_title("Trajectories over 50,000 orbits")
    
    plt.show()
    
#orbit_graph(runge_kutta_4_method, 5*63259, G, delta_t, mass_vector, position_vector, velocity_vector)
    
def distance_between_bodies(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    number_of_measurements = number_of_iterations // iterations_between_updates + 1
    
    measurement = 1
    
    time = np.zeros(number_of_measurements)
    time[0] = 0.0
    distance = np.zeros(number_of_measurements)
    
    for j in range(len(mass_vector)):
        for k in range(j + 1, len(mass_vector)):
            distance[0] += np.linalg.norm(position_vector[j] - position_vector[k])
    
    for i in range(number_of_iterations):
        
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )
        
        if (i + 1) % iterations_between_updates == 0:
            
            time[measurement] = (i + 1) * delta_t
            
            for j in range(len(mass_vector)):
                for k in range(j + 1, len(mass_vector)):
                    distance[measurement] += np.linalg.norm(position_vector[k] - position_vector[j])

            measurement += 1

    fig, ax = plt.subplots()
    
    ax.plot(time / 1000, distance, label = "Distance")
    
    #ax.set_aspect("equal")
    ax.set_xlabel("Time (1000s)")
    ax.set_ylabel("Sum of Pairwise Distances")
    ax.legend()
    ax.grid()
    ax.set_title("Sum of Pairwise Distances Over Time")
    
    plt.show()

distance_between_bodies(runge_kutta_4_method, 62832000, 1000, G, delta_t, mass_vector, position_vector, velocity_vector)

def solution_error_over_time(method, number_of_iterations, G, delta_t, delta_t_ref, mass_vector, position_vector, velocity_vector):
    
    reference_solution = np.zeros((number_of_iterations + 1, len(mass_vector), 3))
    reference_solution[0] = position_vector
    simulation = np.zeros((number_of_iterations + 1, len(mass_vector), 3))
    simulation[0] = position_vector
    time = np.zeros(number_of_iterations + 1)
    
    reference_steps_per_step = int(round(delta_t / delta_t_ref))
    
    position_vector_copy = position_vector.copy()
    velocity_vector_copy = velocity_vector.copy()
    
    for i in range(number_of_iterations * reference_steps_per_step):
        
        position_vector, velocity_vector = runge_kutta_4_method(G,
                                                                delta_t_ref,
                                                                mass_vector,
                                                                position_vector,
                                                                velocity_vector
                                                                )
        
        if (i + 1) % reference_steps_per_step == 0:
            
            reference_solution[(i + 1) // reference_steps_per_step] = position_vector
    
    for j in range(number_of_iterations):
        
        position_vector_copy, velocity_vector_copy = method(G,
                                                            delta_t,
                                                            mass_vector,
                                                            position_vector_copy,
                                                            velocity_vector_copy
                                                            )
        
        simulation[j + 1] = position_vector
        time[j + 1] = delta_t * (j + 1)
    
    error = np.zeros(number_of_iterations + 1)
    
    for k in range(number_of_iterations + 1):
        
        summand = 0
        
        for l in range(len(mass_vector)):
            
            distance = simulation[k,l] - reference_solution[k,l]
            summand += np.dot(distance, distance)
        
        error[k] = np.sqrt(summand)

    fig, ax = plt.subplots()
    
    ax.plot(time, error, label = "Convergence Error")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid()
    ax.set_title("Error")
    
    plt.show()

#solution_error_over_time(runge_kutta_4_method, 10000, G, delta_t, 0.0001, mass_vector, position_vector, velocity_vector)

def convergence(total_time, G, delta_t_ref, mass_vector, position_vector, velocity_vector):
    
    delta_t_times = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    
    methods = [forward_euler_method, runge_kutta_4_method, leapfrog_method]
    
    method_names = ["Forward Euler", "Runge Kutta", "Leapfrog"]
    
    number_of_iterations_ref = int(round(total_time / delta_t_ref))
    
    position_ref = position_vector.copy()
    velocity_ref = velocity_vector.copy()
    
    fig, ax = plt.subplots()
    
    for i in range(number_of_iterations_ref):
        
        position_ref, velocity_ref = runge_kutta_4_method(G,
                                                          delta_t_ref,
                                                          mass_vector,
                                                          position_ref,
                                                          velocity_ref
                                                          )
    
    for method, method_name in zip(methods, method_names):
        
        error_at_final_time = []
        
        for delta_t in delta_t_times:
            
            
            position = position_vector.copy()
            velocity = velocity_vector.copy()
            
            number_of_iterations = int(round(total_time / delta_t))
            
            for j in range(number_of_iterations):
                
                position, velocity = method(G,
                                            delta_t,
                                            mass_vector,
                                            position,
                                            velocity
                                            )
                
                    
            summand = 0
                    
            for k in range(len(mass_vector)):
                        
                distance = position[k] - position_ref[k]
                summand += np.dot(distance, distance)
                    
            error = np.sqrt(summand)
            error_at_final_time.append(error)
            slope = (np.log(error_at_final_time[-1]) - np.log(error_at_final_time[0])) / (np.log(delta_t_times[-1]) - np.log(delta_t_times[0]))
            
        ax.plot(delta_t_times, error_at_final_time, label = f"{method_name} (p = {slope:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Error at Final Time")
    ax.legend()
    ax.grid()
    ax.set_title("Convergence Error")
    
    plt.show()

#convergence(0.1, G, 0.00000001, mass_vector, position_vector, velocity_vector)

def plot_on_same_graph(conserved_quantity, method, total_time, measurement_interval, G, mass_vector, position_vector, velocity_vector):
    
    if conserved_quantity == conservation_of_energy:
        conserved_quantity_name = "Energy"
    elif conserved_quantity == conservation_of_linear_momentum:
        conserved_quantity_name = "Linear Momentum"
    else:
        conserved_quantity_name = "Angular Momentum"
    
    if method == forward_euler_method:
        method_name = "Forward Euler"
    elif method == runge_kutta_4_method:
        method_name = "Runge Kutta"
    else:
        method_name = "Leapfrog"
    
    delta_t_times = [0.05, 0.025, 0.0125]
    
    fig, ax = plt.subplots()
    
    for delta_t in delta_t_times:
        
        number_of_iterations = int(total_time / delta_t)
        iterations_between_updates = 1
        
        time, conserved_quantity_error = conserved_quantity(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector)
            
        ax.plot(time / 1000, abs(conserved_quantity_error), label = f"Δt = {delta_t}")
        ax.set_yscale("log")

    ax.set_title(f"Conservation of {conserved_quantity_name} Using The {method_name} Method")
    ax.set_xlabel("Time (Thousands)")
    ax.set_ylabel("Absolute Relative Error")
    ax.legend()
    ax.grid(which = "both", alpha = 0.3)

    plt.show()

#plot_on_same_graph(conservation_of_energy, leapfrog_method, 10000, 1, G, mass_vector, position_vector, velocity_vector)

def long_term_comparison(conserved_quantity, method_1, method_2, total_time, delta_t, measurement_interval, G, mass_vector, position_vector, velocity_vector):
    
    if conserved_quantity == conservation_of_energy:
        conserved_quantity_name = "Energy"
    elif conserved_quantity == conservation_of_linear_momentum:
        conserved_quantity_name = "Linear Momentum"
    else:
        conserved_quantity_name = "Angular Momentum"
    
    method_name = {forward_euler_method: "Forward Euler",
                   runge_kutta_4_method: "Runge Kutta",
                   leapfrog_method: "Leapfrog"
                   }

    methods = [method_1, method_2]

    fig, ax = plt.subplots()
    
    number_of_iterations = int(total_time / delta_t)
    iterations_between_updates = int(round(measurement_interval / delta_t))
    
    for method in methods:
        
        time, conserved_quantity_error = conserved_quantity(method, number_of_iterations, iterations_between_updates, G, delta_t, mass_vector, position_vector, velocity_vector)

        ax.plot(time / 1000, abs(conserved_quantity_error), label = method_name[method])
        
    ax.set_xlabel("Time (Thousands)")
    ax.set_ylabel("Absolute Relative Error")
    ax.set_title(f"Absolute Relative {conserved_quantity_name} Error: {method_name[method_1]} vs {method_name[method_2]}")
    ax.set_yscale("log")
    ax.grid(which = "both", alpha = 0.3)
    ax.legend()
    
    plt.show()

#long_term_comparison(conserved_quantity, method_1, method_2, total_time, delta_t, measurement_interval, G, mass_vector, position_vector, velocity_vector)
























