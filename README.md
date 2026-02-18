# PIKO-Continuum-Robot-MPC-2025
Physics-Informed Koopman Operator (PIKO) for data-efficient real-time MPC control of an artificial-muscle-driven continuum robot, integrating static Cosserat rod simulation and experimental validation.

## Data-Efficient Real-Time Control of an Artificial-Muscle-Driven Continuum Robot  
### Physics-Informed Koopman Operator (PIKO)

This repository contains the implementation accompanying the paper:

> **Data-Efficient Real-Time Control of an Artificial-Muscle-Driven Continuum Robot with Physics-Informed Koopman Operator**  
> Jiahe Wang, Eron Ristich, Eric Weissman, Yi Ren, and Jiefeng Sun  
> 2026

---

## 🔍 Overview

Data-driven methods such as the Koopman operator enable linear control techniques (e.g., MPC) for nonlinear continuum robots.  
However, conventional Koopman approaches require large amounts of experimental data.

This work introduces **PIKO (Physics-Informed Koopman Operator)**:

- Combines experimental data with physics-based simulation
- Uses a **static Cosserat rod + muscle coupling model**
- Implements **Strang splitting** to integrate physics constraints
- Reduces experimental data requirements
- Enables real-time MPC-based trajectory tracking


## 🏗 Repository Structure
- Data Collection: 'harp_oneseg_ws', ROS 2 environment and hardware commnucation
- Koopman Training:
- Static model and JAX:

