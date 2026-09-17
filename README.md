# N-Body Simulation

Numerical simulation of a Newtonian N-body gravitational system.

## Report

A detailed write-up of the numerical methods, results and analysis is available in the [project report](N_body_Simulation.pdf).

## Overview

This project simulates the orbital trajectories of a closed Newtonian N-body gravitational system using the Forward Euler, Kick-Drift-Kick Leapfrog, and fourth-order Runge-Kutta (RK4) methods.

## Installation

Install the required Python packages:

```bash
pip install numpy matplotlib numba
```

##Numerical Integrators

- Forward Euler method
- Kick-Drift-Kick Leapfrog method
- Fourth-order Runge-Kutta (RK4) method

## Features

- Plot conserved quantities for multiple time steps using each numerical integrator.
- Plot orbital trajectories for an arbitrary number of massive bodies.
- Plot the sum of pairwise distances between an arbitrary number of massive bodies.
- Compare position error against a higher-accuracy RK4 reference solution.
- Compare conserved quantities across Forward Euler, RK4, and Leapfrog.
- Measure runtimes for each numerical integrator.
- Calculate the mean and sample standard deviation of runtimes over repeated simulations.
- Plot position error against runtime for each numerical integrator.
- Produce animations of orbital trajectories.
  
## Usage

Run the main simulation script:

```bash
python main.py
```

The main simulation script contains two example sets of initial conditions:
- a two-body circular orbit
- a three-body figure-eight orbit
The gravitational constant is set to G = 1, with a default global time step of delta_t = 0.1.

## Results

- RK4 gave the smallest energy error for the cases investigated, while Leapfrog showed bounded, oscillatory energy error characteristic of a symplectic method.
- Leapfrog and RK4 conserved linear and angular momentum to approximately \(10^{-13}\) and \(10^{-10}\), respectively, while Forward Euler showed secular error growth and eventually became numerically unstable.
- Measured convergence rates closely matched the expected first-, second-, and fourth-order behaviour of Forward Euler, Leapfrog, and RK4.
- RK4 produced substantially smaller trajectory error than Leapfrog over the tested intervals, while Forward Euler accumulated much larger long-term error.
- For one million iterations, mean runtimes over fifty runs were $3.839 \pm 0.016$, $6.070 \pm 0.026$, and $10.767 \pm 0.038$ seconds for Forward Euler, Leapfrog, and RK4, respectively.
- In the tested three-method comparison, RK4 achieved a $1.98\times10^5$ smaller position error than Leapfrog for only about $1.76\times$ the runtime.

  ## Limitations

- The model assumes Newtonian gravity and does not include relativistic effects.
- Bodies are treated as point masses.
- Collisions and physical interactions between bodies are not currently modelled.
- The fixed time-step integrators may lose accuracy during close encounters or other rapidly changing trajectories.
- Long-term trajectory prediction can be strongly affected by numerical error and sensitivity to initial conditions, particularly for chaotic systems.
- The simulations currently use a single global time step rather than adaptive time stepping.
- Computational cost increases rapidly as the number of bodies increases because gravitational interactions are evaluated pairwise.

## Possible Applications

The simulator can be used to investigate a range of gravitational systems, including:

- Stable and unstable orbital configurations.
- Two-body and three-body dynamics.
- Chaotic gravitational systems.
- Lagrange-point configurations.
- Close encounters between massive bodies.
- Approximate long-term evolution of planetary systems.
- The effect of time-step size on numerical stability and conservation laws.
- Comparisons between numerical integration methods.

## Future Work

Possible extensions to the project include:

- Implementing adaptive time-step integration.
- Adding collision detection and collision handling.
- Investigating Lagrange-point stability.
- Simulating more realistic Solar System initial conditions.
- Adding relativistic corrections for high-precision orbital simulations.
- Parallelising the gravitational force calculation for larger N-body systems.
- Implementing the numerical core in C++ to compare performance with the current Python implementation.
- Exploring more advanced symplectic integration methods.

## References

References used in the accompanying report include material on:

- Newtonian gravitational dynamics.
- Forward Euler integration.
- Fourth-order Runge-Kutta integration.
- Leapfrog and symplectic integration methods.
- Numerical error and conservation properties of time integration schemes.

A complete reference list is provided in the project report.
