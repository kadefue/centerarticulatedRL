# Autonomous Articulated Vehicle Control

This notebook explores and compares various path-tracking control algorithms for articulated vehicles (e.g., tractors, loaders) in agricultural or off-road environments.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Vehicle Model](#vehicle-model)
3.  [Control Algorithms](#control-algorithms)
4.  [Simulation Scenarios](#simulation-scenarios)
5.  [Results & Analysis](#results--analysis)
6.  [Conclusion](#conclusion)

## 1. Introduction

Articulated vehicles, commonly found in agriculture and construction, present unique control challenges due to their two-part chassis connected by a central hinge. Accurate path tracking is crucial for tasks like precision farming (row following) and material handling. This notebook implements and evaluates three primary control strategies:

-   **Pure Pursuit (PP):** A widely used geometric control method.
-   **Dr. Kadeghe Fue's Modified Pure Pursuit (Fue MPP):** An enhanced geometric method specifically adapted for articulated vehicles, often incorporating cross-track error (CTE) correction.
-   **Model Predictive Control (MPC):** An optimization-based advanced control technique.

The notebook progressively refines the `Fue Modified Pure Pursuit` algorithm, starting from a basic proportional controller and evolving into more sophisticated versions including PD, PID, vectorized P, adaptive lookahead, and even a Q-Learning (Reinforcement Learning) approach.

## 2. Vehicle Model

The core of the simulations is a kinematic model of a **center-articulated vehicle**. This model captures the unique steering mechanism where articulation (the angle between the front and rear chassis) dictates the turning radius. Key parameters include wheelbase, maximum articulation angle, and vehicle speed.

```python
class ArticulatedVehicle:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=1.5, ...):
        # ... initialization ...
    def update(self, a, target_gamma):
        # ... kinematic equations ...
        # Articulated Yaw Rate Equation
        num = self.v * math.sin(self.gamma) + LR * gamma_dot
        den = LF * math.cos(self.gamma) + LR
        yaw_dot = num / den
        # ... integration ...
```

Later iterations of the notebook introduce more advanced physical models:

-   **Traction-Aware Dynamics:** Simulating wheel slip and its effect on velocity and steering, especially in low-friction conditions (mud).
-   **Power Consumption Model:** Estimating energy usage based on rolling resistance, acceleration, and steering effort.

## 3. Control Algorithms

### A. Standard Pure Pursuit

Calculates the articulation angle (`gamma`) required to reach a lookahead point on the path. The classic formula is adapted for articulated vehicles.

### B. Dr. Kadeghe Fue's Modified Pure Pursuit (Iterative Improvement)

This notebook demonstrates the evolution of the Fue MPP approach:

-   **P-Controller:** Basic correction based on Cross-Track Error (CTE).
-   **PD-Controller:** Adds a derivative term (`Kd`) to dampen oscillations and improve response speed.
-   **PID-Controller:** Further adds an integral term (`Ki`) to eliminate steady-state error.
-   **Vectorized P:** Uses `atan(k * CTE)` to provide a non-linear, saturated correction, making it robust to large errors.
-   **Fue Vector (Hybrid):** Combines geometric PP with direct corrections for CTE and heading error, often prioritizing local alignment.
-   **Adaptive Lookahead:** Dynamically adjusts the Pure Pursuit lookahead distance based on error magnitude, allowing for aggressive turns on corners and smooth driving on straights.
-   **Curvature Feedforward:** Incorporates knowledge of upcoming path curvature to anticipate steering commands.
-   **Smart Braking/Speed Control:** Dynamically adjusts vehicle speed based on the severity of the turn or terrain conditions to maintain control and reduce error.

### C. Model Predictive Control (MPC)

An optimization-based controller that computes a sequence of control actions (acceleration and articulation angle) over a prediction horizon, minimizing a cost function that typically penalizes deviation from the path and control effort. A linearized kinematic model is used with `cvxpy` for solving the optimization problem.

### D. Reinforcement Learning (Q-Learning)

A Q-Learning agent is trained to select optimal control strategies (e.g., gain multipliers for the base controller, speed limits) based on the current state (CTE, heading error, future curvature, *and terrain slip*). This allows the vehicle to adapt its driving style to different conditions.

## 4. Simulation Scenarios

The controllers are tested on a variety of challenging paths:

-   **Sine Wave:** Smooth, continuous curves.
-   **Trapezoidal:** Features sharp, 45-degree-like turns with straight sections.
-   **Square Wave:** Extremely challenging with abrupt 90-degree turns to stress controller performance.

Later simulations introduce environmental factors:

-   **Variable Friction Soil:** Simulating mud patches to test robustness against slip.
-   **Mixed Terrain with Power Modeling:** Evaluating energy consumption and path tracking across different terrains (tarmac, rough, mud), with realistic slip effects.

## 5. Results & Analysis

Each iteration of the notebook provides plots comparing the trajectories of different controllers and quantitative metrics like:

-   **RMSE (Root Mean Square Error):** Average deviation from the path.
-   **Max Error:** The largest instantaneous deviation.
-   **Energy Consumption (kJ):** For simulations incorporating power models.

Key observations include:

-   Standard Pure Pursuit struggles with sharp turns and exhibits larger errors.
-   Fue MPP variants (P, PD, PID) progressively improve accuracy, with PD and PID showing better damping and steady-state performance.
-   Vectorized and Adaptive Fue MPP achieve superior performance on complex paths by dynamically adjusting to environmental cues.
-   MPC generally offers robust performance but can be computationally intensive and sensitive to model accuracy.
-   RL agents demonstrate the ability to learn adaptive strategies, particularly beneficial in mixed-terrain scenarios where explicit tuning for every condition is difficult.

## 6. Conclusion

This notebook illustrates the journey of developing effective control strategies for articulated vehicles. It highlights that while simple geometric methods provide a good baseline, incorporating error feedback (Fue MPP variations), predictive capabilities (MPC), and adaptive learning (RL) significantly enhances path-tracking accuracy, stability, and efficiency in complex, real-world conditions.
