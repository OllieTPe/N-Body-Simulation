# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:03:24 2026

@author: olive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from time import perf_counter

G = 1

delta_t = 0.01

#mass_vector = np.array([1.0,1.0])

#position_vector = np.array([[0.0,1.0,0.0],
 #                           [0.0,-1.0,0.0]
  #                          ])

#velocity_vector = np.array([[0.5,0.0,0.0],
 #                           [-0.5,0.0,0.0]
  #                           ])

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
    
    conserved_quantity_names = {conservation_of_energy: "Energy",
                                conservation_of_linear_momentum: "Linear Momentum",
                                conservation_of_angular_momentum: "Angular Momentum"
                                }
    
    method_names = {forward_euler_method: "Forward Euler",
                    runge_kutta_4_method: "Runge Kutta",
                    leapfrog_method: "Leapfrog"}
    
    
    methods = [runge_kutta_4_method, leapfrog_method]
    
    delta_t_times = [0.05, 0.025, 0.0125]
    
    fig, axes = plt.subplots(len(methods), len(delta_t_times), figsize = (9,6), sharex = True, sharey = "row", constrained_layout = True)
    
    for row, method in enumerate(methods):
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
                ax.set_ylabel(f"{method_names[method]} Error")
                
            if row == len(methods) - 1:
                ax.set_xlabel("Time")
            
    fig.suptitle(f"{conserved_quantity_names[conserved_quantity]} Conservation Comparison", fontsize = 16)        
    
    plt.show()

#plot_graphs(conservation_of_linear_momentum, 10, 1, G, mass_vector, position_vector, velocity_vector)

def orbit_graph(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    body_colours = {0: "tab:red",
                    1: "tab:blue",
                    2: "tab:orange"}
    
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
        ax.plot(orbit[:, body, 0], orbit[:, body, 1], label = f"Body {body + 1}", color = body_colours[body])

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
    ax.set_title("Trajectories Over 10,000 Orbits")
    
    plt.show()
    
#orbit_graph(runge_kutta_4_method, 632600, G, delta_t, mass_vector, position_vector, velocity_vector)
    
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
    
    ax.set_ylim(
        distance[:].min() - 0.01,
        distance[:].max() + 0.01
        )
    
    #ax.set_aspect("equal")
    ax.set_xlabel(r"Time ($10^3$ s)")
    ax.set_ylabel("Sum of Pairwise Distances")
    ax.legend()
    ax.grid()
    ax.set_title("Sum of Pairwise Distances Over Time")
    
    plt.show()

#distance_between_bodies(runge_kutta_4_method, 632600, 100, G, delta_t, mass_vector, position_vector, velocity_vector)

def solution_error_over_time(number_of_iterations, iterations_between_updates, G, delta_t, delta_t_ref, mass_vector, position_vector, velocity_vector):
    
    method_names = {forward_euler_method: "Forward Euler",
                   runge_kutta_4_method: "Runge Kutta",
                   leapfrog_method: "Leapfrog"
                   }
    
    method_colours = {forward_euler_method: "tab:blue",
                      runge_kutta_4_method: "tab:orange",
                      leapfrog_method: "tab:green"}
    
    number_of_measurements = number_of_iterations // iterations_between_updates + 1
    
    reference_solution = np.zeros((number_of_measurements, len(mass_vector), 3))
    reference_solution[0] = position_vector
    time = np.zeros(number_of_measurements)
    
    steps_per_step_reference = int(round(delta_t / delta_t_ref))
    
    methods = [runge_kutta_4_method, leapfrog_method]
    
    position_vector_copy = position_vector.copy()
    velocity_vector_copy = velocity_vector.copy()
    
    fig, ax = plt.subplots()
    
    for i in range(number_of_iterations * steps_per_step_reference):
        
        position_vector_copy, velocity_vector_copy = runge_kutta_4_method(G,
                                                                delta_t_ref,
                                                                mass_vector,
                                                                position_vector_copy,
                                                                velocity_vector_copy
                                                                )
        
        if (i + 1) % (steps_per_step_reference * iterations_between_updates) == 0:
            
            reference_solution[(i + 1) // (steps_per_step_reference * iterations_between_updates)] = position_vector_copy
    
    errors = {}
    
    for method in methods:
        
        simulation = np.zeros((number_of_measurements, len(mass_vector), 3))
        simulation[0] = position_vector
        
        position_vector_copy = position_vector.copy()
        velocity_vector_copy = velocity_vector.copy()
        
        measurement = 1
        
        for j in range(number_of_iterations):
            
            position_vector_copy, velocity_vector_copy = method(G,
                                                                delta_t,
                                                                mass_vector,
                                                                position_vector_copy,
                                                                velocity_vector_copy
                                                                )
            
            if (j + 1) % iterations_between_updates == 0:
                
                simulation[measurement] = position_vector_copy
                time[measurement] = delta_t * (j + 1)
        
                measurement += 1
        
        error = np.zeros(number_of_measurements)
        
        for k in range(number_of_measurements):
            
            summand = 0.0
            
            for l in range(len(mass_vector)):
                
                distance = simulation[k,l] - reference_solution[k,l]
                summand += np.dot(distance, distance)
            
            error[k] = np.sqrt(summand)
    
        errors[method_names[method]] = error.copy()
    
        ax.plot(time, error, label = method_names[method], color = method_colours[method])
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error")
    ax.legend()
    ax.grid()
    ax.set_title("Trajectory Error Relative to Reference Solution")
    
    plt.show()

#solution_error_over_time(100000, 10, G, delta_t, 0.0001, mass_vector, position_vector, velocity_vector)

def convergence(total_time, G, delta_t_ref, mass_vector, position_vector, velocity_vector):
    
    method_names = {forward_euler_method: "Forward Euler",
                    runge_kutta_4_method: "Runge Kutta",
                    leapfrog_method: "Leapfrog"}
    
    delta_t_times = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    methods = [forward_euler_method, runge_kutta_4_method, leapfrog_method]
    
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
    
    for method in methods:
        
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
            
        ax.plot(delta_t_times, error_at_final_time, label = f"{method_names[method]} (p = {slope:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Error at Final Time")
    ax.legend()
    ax.grid()
    ax.set_title("Convergence Error")
    
    plt.show()

#convergence(0.1, G, 0.00000001, mass_vector, position_vector, velocity_vector)

def plot_on_same_graph(conserved_quantity, total_time, measurement_interval, G, mass_vector, position_vector, velocity_vector):
    
    conserved_quantity_names = {conservation_of_energy: "Energy",
                                conservation_of_linear_momentum: "Linear Momentum",
                                conservation_of_angular_momentum: "Angular Momentum"
                                }
    
    method_names = {forward_euler_method: "Forward Euler",
                    runge_kutta_4_method: "Runge Kutta",
                    leapfrog_method: "Leapfrog"}
    
    method_colours = {forward_euler_method: "tab:blue",
                      runge_kutta_4_method: "tab:orange",
                      leapfrog_method: "tab:green"}
    
    delta_t_times = [0.01]
    
    methods = [leapfrog_method, runge_kutta_4_method]
    
    fig, ax = plt.subplots()
    
    for method in methods:
        
        for delta_t in delta_t_times:
            
            number_of_iterations = int(total_time / delta_t)
            iterations_between_updates = max(1, int(round(measurement_interval / delta_t)))
            
            time, conserved_quantity_error = conserved_quantity(method,
                                                                number_of_iterations,
                                                                iterations_between_updates,
                                                                G,
                                                                delta_t,
                                                                mass_vector,
                                                                position_vector,
                                                                velocity_vector
                                                                )
                
            ax.plot(time / 1000, abs(conserved_quantity_error), label = method_names[method], color = method_colours[method])
            

    ax.set_title(f"Relative {conserved_quantity_names[conserved_quantity]} Error")
    ax.set_xlabel(r"Time ($10^{3}$ s)")
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(which = "both", alpha = 0.3)

    plt.show()

plot_on_same_graph(conservation_of_energy, 200000, delta_t, G, mass_vector, position_vector, velocity_vector)

def long_term_comparison(conserved_quantity, method_1, method_2, total_time, delta_t, measurement_interval, G, mass_vector, position_vector, velocity_vector):
    
    conserved_quantity_names = {conservation_of_energy: "Energy",
                                conservation_of_linear_momentum: "Linear Momentum",
                                conservation_of_angular_momentum: "Angular Momentum"
                                }
    
    method_names = {forward_euler_method: "Forward Euler",
                   runge_kutta_4_method: "Runge Kutta",
                   leapfrog_method: "Leapfrog"
                   }

    methods = [method_1, method_2]

    fig, ax = plt.subplots()
    
    number_of_iterations = int(total_time / delta_t)
    iterations_between_updates = int(round(measurement_interval / delta_t))
    
    for method in methods:
        
        time, conserved_quantity_error = conserved_quantity(method,
                                                            number_of_iterations,
                                                            iterations_between_updates,
                                                            G,
                                                            delta_t,
                                                            mass_vector,
                                                            position_vector,
                                                            velocity_vector
                                                            )

        ax.plot(time, abs(conserved_quantity_error), label = method_names[method])
        
    ax.set_xlabel(r"Time (s)")
    ax.set_ylabel("Absolute Relative Error")
    ax.set_title(f"Absolute Relative {conserved_quantity_names[conserved_quantity]} Error")
    ax.grid(which = "both", alpha = 0.3)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()

#long_term_comparison(conservation_of_energy,  forward_euler_method, runge_kutta_4_method, 200, delta_t, delta_t, G, mass_vector, position_vector, velocity_vector)

def method_run_time(method, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    method(G,
           delta_t,
           mass_vector,
           position_vector,
           velocity_vector
           )
    
    start = perf_counter()
    
    for i in range(number_of_iterations):
        
        position_vector, velocity_vector = method(G,
                                                  delta_t,
                                                  mass_vector,
                                                  position_vector,
                                                  velocity_vector
                                                  )

    end = perf_counter()
    
    run_time = end - start
    
    return run_time

def plot_runtime(G, delta_t, mass_vector, position_vector, velocity_vector):
    
    method_names = {forward_euler_method: "Forward Euler",
                    runge_kutta_4_method: "Runge Kutta",
                    leapfrog_method: "Leapfrog"
                    }
    
    methods = [forward_euler_method, runge_kutta_4_method, leapfrog_method]
    
    number_of_iterations_list = np.array([100, 1000, 10000, 100000, 1000000])
    
    runtimes = np.zeros(len(number_of_iterations_list))
    
    fig, ax = plt.subplots()
    
    for method in methods:
        
        for i, number_of_iterations in enumerate(number_of_iterations_list):
        
            runtimes[i] = method_run_time(method,
                                          number_of_iterations,
                                          G,
                                          delta_t,
                                          mass_vector,
                                          position_vector,
                                          velocity_vector
                                          )
        
        plt.plot(number_of_iterations_list / 1000, runtimes, label = method_names[method])

    ax.set_xlabel(r"Number of Iterations $(10^3)$")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Computational Runtime vs Number of Iterations")
    ax.grid(which = "both", alpha = 0.3)
    ax.legend()
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    plt.show()

#plot_runtime(G, delta_t, mass_vector, position_vector, velocity_vector)

def repeated_plot_runtime(number_of_readings, number_of_iterations, G, delta_t, mass_vector, position_vector, velocity_vector):
    
    method_names = {forward_euler_method: "Forward Euler",
                    runge_kutta_4_method: "Runge Kutta",
                    leapfrog_method: "Leapfrog"
                    }
    
    methods = [forward_euler_method, runge_kutta_4_method, leapfrog_method]
    
    for method in methods:
        
        runtimes = np.zeros(number_of_readings)
        
        for j in range(number_of_readings):
            
            runtimes[j] = method_run_time(method,
                                          number_of_iterations,
                                          G,
                                          delta_t,
                                          mass_vector,
                                          position_vector,
                                          velocity_vector
                                          )

        mean = np.mean(runtimes)

        standard_deviation = np.std(runtimes, ddof = 1)

        print(f"{method_names[method]}: {mean} ± {standard_deviation} s")

#repeated_plot_runtime(5, 1000000, G, delta_t, mass_vector, position_vector, velocity_vector)
    
def accuracy_vs_computational_cost(total_time, number_of_readings, delta_t_ref):
    
    method_names = {forward_euler_method: "Forward Euler",
                   runge_kutta_4_method: "Runge Kutta",
                   leapfrog_method: "Leapfrog"
                   }
    
    number_of_iterations_ref = int(round(total_time / delta_t_ref))
    
    reference_solution = np.zeros((number_of_iterations_ref, len(mass_vector), 3))
    reference_solution[0] = position_vector
    
    methods = [forward_euler_method, runge_kutta_4_method, leapfrog_method]
    delta_t_times = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    
    position_vector_ref = position_vector.copy()
    velocity_vector_ref = velocity_vector.copy()
    
    fig, ax = plt.subplots()
    
    for i in range(number_of_iterations_ref):
        
        position_vector_ref, velocity_vector_ref = runge_kutta_4_method(G,
                                                                delta_t_ref,
                                                                mass_vector,
                                                                position_vector_ref,
                                                                velocity_vector_ref
                                                                )
        
        
        
    for method in methods:
        
        position_errors = np.zeros(len(delta_t_times))
        mean_runtimes = np.zeros(len(delta_t_times))
        
        for i, delta_t in enumerate(delta_t_times):
            
            position_vector_copy = position_vector.copy()
            velocity_vector_copy = velocity_vector.copy()
            
            number_of_iterations = int(round(total_time / delta_t))
            
            runtimes = np.zeros(number_of_readings)
            
            for j in range(number_of_readings):
                
                runtimes[j] = method_run_time(method,
                                              number_of_iterations,
                                              G,
                                              delta_t,
                                              mass_vector,
                                              position_vector,
                                              velocity_vector
                                              )
            
            mean_runtimes[i] = np.mean(runtimes)
    
            for j in range(number_of_iterations):
                
                position_vector_copy, velocity_vector_copy = method(G,
                                                                    delta_t,
                                                                    mass_vector,
                                                                    position_vector_copy,
                                                                    velocity_vector_copy
                                                                    )
            
            position_errors[i] = np.linalg.norm(position_vector_ref - position_vector_copy)
    
        print(method_names[method], mean_runtimes, position_errors, delta_t_times)
    
        ax.plot(mean_runtimes, position_errors, label = method_names[method])
    
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Position Error")
    ax.set_title("Accuracy vs Computational Cost")
    ax.grid(which = "both", alpha = 0.3)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()
    
#accuracy_vs_computational_cost(100, 100, 0.00001)

