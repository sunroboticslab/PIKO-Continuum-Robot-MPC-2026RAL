#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0
"""
offline_mpc_sim.py

1) Loads the trained Koopman model (A,B, μ/σ, delay).
2) Reads mpc_log.csv to get the very first (dx0,dz0).
3) Reads mpc_log.csv to recover the reference window (ref_x,ref_y) 
   and the actual inputs u1–u3.
4) Simulates z_{k+1} = A·z_k + B·u_norm[k] for k=0..N-1, reconstructs dx,dz.
5) Prints the reference sent at each step.
6) Plots:
     a) Reference (blue dashed) vs. actual (thin red) vs. predicted (green) trajectories,
        and marks the initial state in black.
     b) Prediction vs. actual Euclidean error over steps.
"""

import pickle
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_initial_z(dx0, dz0, delay):
    """
    Build the initial lifted state z0 by repeating the first measurement
    across all delay slots, rather than zeros.

    dx0, dz0 : scalars (first measured dx/dz)
    delay    : model’s delay (integer)

    Returns z0 of shape ((2*(delay+1))×1).
    """
    x0 = np.array([[dx0], [dz0]])              # (2×1)
    hist = deque([x0] * (delay+1), maxlen=delay+1)
    mats = list(hist)[::-1]                     # newest→oldest
    Z = np.hstack(mats)                         # 2×(delay+1)
    return Z.reshape(-1, 1)                     # (2*(delay+1))×1

def main():
    # 1) load Koopman model
    mdl = pickle.load(open('koopman_model.pkl','rb'))
    A = np.array(mdl['A'])
    B = np.array(mdl['B'])
    mu_u, sigma_u = np.array(mdl['mu_u']), np.array(mdl['sigma_u'])
    mu_x, sigma_x = np.array(mdl['mu_x']), np.array(mdl['sigma_x'])
    delay = int(mdl['delay'])
    # projection: pick first two lifted dims → [dx;dz]
    n_lift = A.shape[1]
    C = np.zeros((2, n_lift)); C[0,0]=1; C[1,1]=1

    # 2) read mpc_log.csv
    log = pd.read_csv('mpc_log.csv')
    dx0, dz0 = log.loc[0, ['dx','dz']].values

    # 3) recover reference & inputs
    actual   = log[['dx','dz']].values    # (N×2)
    u_orig = log[['u1','u2','u3']].values     # (N×3)

    # build initial lifted state
    z = build_initial_z(dx0, dz0, delay)

    # arrays for predicted dx,dz
    pred = np.zeros((len(log), 2))
    # force the very first prediction to equal the true initial measurement
    pred[0] = [dx0, dz0]

    print("Step-by-step reference sent to MPC:")
    for k in range(1,len(log)):
        print(f" Step {k:3d}: actual[:,0] = {actual[k]}")
        u_norm = (u_orig[k] - mu_u) / sigma_u
        z = A.dot(z) + B.dot(u_norm.reshape(-1,1))
        x_norm = C.dot(z).flatten()
        x_real = x_norm * sigma_x + mu_x
        pred[k] = x_real

    # actual = log[['dx','dz']].values

    # 4a) plot trajectories + initial state
    plt.figure(figsize=(6,6))
    # plt.plot(refs[:,0],  refs[:,1],  'b--', label='Reference')
    plt.plot(actual[:,0], actual[:,1], 'r-',  alpha=0.1, label='Actual')
    plt.plot(pred[:,0],   pred[:,1],   'g-',  alpha=0.8, label='Predicted')
    # mark initial state
    plt.scatter([dx0], [dz0], c='k', s=50, marker='o', label='Initial State')
    plt.axis('equal')
    plt.xlabel('dx'); plt.ylabel('dz')
    plt.title('Trajectory: ref vs actual vs predicted')
    plt.legend()
    plt.tight_layout()

    # 4b) plot prediction vs actual error
    err_pa = np.linalg.norm(pred - actual, axis=1)
    plt.figure(figsize=(8,4))
    steps = np.arange(len(log))
    plt.plot(steps, err_pa, 'm-', label='||predicted – actual||')
    plt.xlabel('Step')
    plt.ylabel('Euclidean error')
    plt.title('Prediction vs Actual Error Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
