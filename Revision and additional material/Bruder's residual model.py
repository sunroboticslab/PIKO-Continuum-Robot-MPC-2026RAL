#!/usr/bin/env python3
import csv
import sys
import pickle
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------------
def plot_data_comparison(exp_filename, sim_filename):
    """
    Plot the comparison of experimental and simulation data distributions.
    Only plot the raw trajectories (left subplot) and save as 'trajectory_comparison.pdf'.
    """
    # Load data
    exp_data, sim_data = load_data(exp_filename, sim_filename)
    
    # Extract position data
    # Experimental: [dz, dx] to match [x,y] coordinates
    X_exp = np.vstack([exp_data[:, 5], exp_data[:, 4]]).T   # shape = (N_h, 2) = [x=dz, y=dx]
    # Simulation: [x, y]
    X_sim = np.vstack([sim_data[:, 4], sim_data[:, 5]]).T   # shape = (N_f, 2) = [x, y]

    # Calculate means and standard deviations
    exp_mean_x = np.mean(X_exp[:, 0])
    exp_mean_y = np.mean(X_exp[:, 1])
    exp_std_x = np.std(X_exp[:, 0])
    exp_std_y = np.std(X_exp[:, 1])
    
    sim_mean_x = np.mean(X_sim[:, 0])
    sim_mean_y = np.mean(X_sim[:, 1])
    sim_std_x = np.std(X_sim[:, 0])
    sim_std_y = np.std(X_sim[:, 1])

    # Create figure for only the raw trajectories
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    # Plot 1: Raw trajectories
    ax1.scatter(X_exp[:, 0], X_exp[:, 1],
                c='red', s=1, alpha=0.1, label='Experiment [x=dz, y=dx]')
    ax1.scatter(X_sim[:, 0], X_sim[:, 1],
                c='blue', s=1, alpha=0.1, label='Simulation [x, y]')

    # Calculate plot limits for mean lines
    x_min = min(X_exp[:, 0].min(), X_sim[:, 0].min())
    x_max = max(X_exp[:, 0].max(), X_sim[:, 0].max())
    y_min = min(X_exp[:, 1].min(), X_sim[:, 1].min())
    y_max = max(X_exp[:, 1].max(), X_sim[:, 1].max())

    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # --- Ensure both axes have the same length and scale for the raw trajectories subplot (ax1) ---
    x_len = x_max - x_min
    y_len = y_max - y_min
    max_len = max(x_len, y_len)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_min_new = x_center - max_len / 2
    x_max_new = x_center + max_len / 2
    y_min_new = y_center - max_len / 2
    y_max_new = y_center + max_len / 2

    # Plot mean lines for experiment
    ax1.axvline(exp_mean_x, color='red', linestyle='--', alpha=0.5, 
                label=f'Exp mean x: {exp_mean_x:.2f}')
    ax1.axhline(exp_mean_y, color='red', linestyle='--', alpha=0.5,
                label=f'Exp mean y: {exp_mean_y:.2f}')

    # Plot mean lines for simulation
    ax1.axvline(sim_mean_x, color='blue', linestyle='--', alpha=0.5,
                label=f'Sim mean x: {sim_mean_x:.2f}')
    ax1.axhline(sim_mean_y, color='blue', linestyle='--', alpha=0.5,
                label=f'Sim mean y: {sim_mean_y:.2f}')

    # Add mean point markers
    ax1.plot(exp_mean_x, exp_mean_y, 'ro', markersize=8, label='Exp mean point')
    ax1.plot(sim_mean_x, sim_mean_y, 'bo', markersize=8, label='Sim mean point')

    ax1.set_title("Raw Trajectories (90% Transparency) with Mean Values")
    ax1.set_xlabel("x coordinate (dz or x)")
    ax1.set_ylabel("y coordinate (dx or y)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.set_xlim(x_min_new, x_max_new)
    ax1.set_ylim(y_min_new, y_max_new)
    ax1.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('trajectory_comparison_5step.pdf')
    plt.show()

def load_data(exp_filename, sim_filename):
    """Load and preprocess both experimental and simulation data from given filenames."""
    # Load experimental data
    exp_df = pd.read_csv(exp_filename)
    exp_data = exp_df.values  # Convert to numpy array

    # Load simulation data
    sim_df = pd.read_csv(sim_filename)
    # First convert all numeric columns to float, keeping input_type as is
    numeric_cols = ['time', 'p1', 'p2', 'p3', 'x', 'y']
    for col in numeric_cols:
        sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
    # Drop any rows with NaN values that might have resulted from conversion
    sim_df = sim_df.dropna()
    # Extract only the numeric columns for the numpy array
    sim_data = sim_df[numeric_cols].values

    print(f"\nData loading summary (FULL datasets) from {exp_filename} and {sim_filename}:")
    print(f"Experimental data shape: {exp_data.shape}")
    print(f"Simulation data shape: {sim_data.shape}")
    print(f"Simulation columns used: {numeric_cols}")

    return exp_data, sim_data

def gen_delay_traj(X, U, delay):
    """
    X: [N, D]    state sequence (D=2 for [dx,dz] or [y,x])
    U: [N, p]    input sequence (p=3)
    delay: int

    Returns:
      X_delay: [(N-delay) × (D*(delay+1))],  
      U_delay: [(N-delay) × p]
    """
    N, D = X.shape
    M = N - delay
    X_delay = np.zeros((M, D * (delay + 1)))
    for i in range(delay + 1):
        X_delay[:, D * i : D * (i + 1)] = X[delay - i : N - i, :]
    U_delay = U[delay:, :]
    return X_delay, U_delay

def iden_koop(X_delay, U_delay, mode=0, alpha=1e-3, tol=1e-4):
    """
    Identify Koopman A, B from delayed data:
      X_delay: [M × d],   d = D*(delay+1)
      U_delay: [M × p]

    mode: 0 = least‐squares, 1 = Lasso, 2 = Ridge
    Returns: A [d×d], B [d×p]
    """
    # Form regression matrices Y, Z:
    Y = np.hstack([X_delay[:-1], U_delay[:-1]])
    Z = np.hstack([X_delay[1:], U_delay[1:]])
    if mode == 0:
        K = np.linalg.lstsq(Y, Z, rcond=None)[0].T
    elif mode == 1:
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_
    else:
        clf = Ridge(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_

    d = X_delay.shape[1]        # dimension of the lifted state
    A = K[:d, :d]               # top‐left d×d
    B = K[:d, d:]               # top‐right d×p
    return A, B

def iden_koop_full(X_delay, U_delay, mode=0, alpha=1e-3, tol=1e-4):
    """
    Identify full Koopman matrix K from delayed data:
      X_delay: [M × d],   d = D*(delay+1)
      U_delay: [M × p]

    mode: 0 = least‐squares, 1 = Lasso, 2 = Ridge
    Returns: K [d×d] (full Koopman matrix)
    """
    # Form regression matrices Y, Z:
    Y = np.hstack([X_delay[:-1], U_delay[:-1]])
    Z = np.hstack([X_delay[1:], U_delay[1:]])
    if mode == 0:
        K = np.linalg.lstsq(Y, Z, rcond=None)[0].T
    elif mode == 1:
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_
    else:
        clf = Ridge(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_

    d = X_delay.shape[1]        # dimension of the lifted state
    K_full = K[:d, :d]          # full Koopman matrix (d×d)
    return K_full

def compose_koopman_models(K_f, K_h, Xd_f, Ud_f, Xd_h, Ud_h, mode=0):
    """
    Compose Koopman models using K_full = K_f @ K_h @ K_f
    and then extract A_full, B_full from the composed data.
    """
    # Compose the full Koopman matrix
    K_full = K_f @ K_h @ K_f
    
    # Combine the datasets
    Xd_combined = np.vstack([Xd_f, Xd_h])
    Ud_combined = np.vstack([Ud_f, Ud_h])
    
    # Calculate A_full and B_full from combined data using the same approach
    A_full, B_full = iden_koop(Xd_combined, Ud_combined, mode=mode)
    
    return A_full, B_full

def slice_by_number(data, n_samples):
    """
    Take exactly n_samples rows from the data.
    If n_samples > data.shape[0], return all data.
    """
    if n_samples <= 0:
        return data[:0, :]
    if n_samples >= data.shape[0]:
        return data.copy()
    return data[:n_samples, :]

def test_one_step(X_n, U_n, A, B, delay):
    """
    Given normalized experiment slice X_n ∈ ℝ^{N×2}, U_n ∈ ℝ^{N×3},
    and Koopman (A,B) where A ∈ ℝ^{d×d}, B ∈ ℝ^{d×3}, we construct
    the delayed-lifted arrays Xd, Ud with the same 'delay', then:
      z_i = Xd[i] ∈ ℝ^d
      u_i = Ud[i] ∈ ℝ^3
    The one-step prediction is: z_{i+1}^pred = A·z_i + B·u_i,
    and the true next-state is z_{i+1}^true = Xd[i+1, :2] (the first two dims).
    We return a 1‐D array err[i] = || x_{i+1}^pred − x_{i+1}^true ||₂
    for i=0…(M−2), where M = N−delay.
    """
    Xd, Ud = gen_delay_traj(X_n, U_n, delay)
    M = Xd.shape[0]
    errors = []
    for i in range(M - 1):
        z_i = Xd[i].reshape(-1, 1)                # (d×1)
        u_i = Ud[i].reshape(-1, 1)                # (3×1)
        z_next_pred = A.dot(z_i) + B.dot(u_i)     # (d×1)
        x_next_pred = z_next_pred[:2, 0]          # (dx_norm, dz_norm)
        x_next_true = Xd[i + 1, :2]               # (2,)
        err_i = np.linalg.norm(x_next_pred - x_next_true)
        errors.append(err_i)
    return np.array(errors)

def generate_exponential_sizes(max_size):
    """
    Generate exponential data sizes: 2^8, 2^9, ..., 2^n until max_size
    Also include the full dataset size if it's not already included
    """
    sizes = []
    power = 8  # Start at 2^8 = 256
    while True:
        size = 2**power
        if size > max_size:
            break
        sizes.append(size)
        power += 1
    # Add full dataset size if it's not already included
    if max_size not in sizes:
        sizes.append(max_size)
    return sizes

def train_and_save_exponential(delay=20, exp_filename=None, sim_filename=None):
    """
    Train Koopman models using exponential data sizes (2^8, 2^9, ..., 2^n)
    """
    models = {}

    print("\nTraining Koopman Models with Exponential Data Sizes:")
    print("-----------------------------------------------------")

    # Load the full datasets
    exp_data, sim_data = load_data(exp_filename, sim_filename)

    # Split experimental data into training and validation sets
    val_size = 20000
    train_size = 280000
    total_needed = val_size + train_size
    
    if exp_data.shape[0] >= total_needed:
        exp_data_train = exp_data[:train_size, :]
        exp_data_val = exp_data[train_size:train_size+val_size, :]
    elif exp_data.shape[0] > val_size:
        # If we have more than validation size but less than total needed
        exp_data_train = exp_data[:-val_size, :]
        exp_data_val = exp_data[-val_size:, :]
    else:
        # If we have less than validation size, use all data for both
        exp_data_train = exp_data
        exp_data_val = exp_data

    print(f"\nRaw data sizes:")
    print(f"Raw experimental data: {exp_data.shape[0]:,} samples (train: {exp_data_train.shape[0]}, val: {exp_data_val.shape[0]})")
    print(f"Raw simulation data: {sim_data.shape[0]:,} samples")

    # Generate exponential sizes for both datasets
    exp_sizes = generate_exponential_sizes(exp_data_train.shape[0])
    sim_sizes = generate_exponential_sizes(sim_data.shape[0])
    
    print(f"\nExperimental data sizes: {exp_sizes}")
    print(f"Simulation data sizes: {sim_sizes}")
    
    # Extract features from full data
    # columns: ts_rel, P1,P2,P3, dx, dz
    P_full_h = exp_data_train[:, 1:4]                    # (N_h × 3)
    X_full_h = np.vstack([exp_data_train[:, 5], exp_data_train[:, 4]]).T  # (N_h × 2) = [x=dz, y=dx]

    # columns: time, p1,p2,p3, x,y
    P_full_f = sim_data[:, 1:4]                    # (N_f × 3)
    X_full_f = np.vstack([sim_data[:, 4], sim_data[:, 5]]).T  # (N_f × 2) = [x, y]

    # ================
    # 1) Pure Experiment Models (Kh_only)
    # ================
    print("\n1. Training Pure Experiment Models (Kh_only):")
    
    for n_samples in exp_sizes:
        key_h = f'exp_{n_samples}'
        print(f"   Training {key_h} with {n_samples:,} samples...")
        
        P_h_sub = slice_by_number(P_full_h, n_samples)
        X_h_sub = slice_by_number(X_full_h, n_samples)

        if P_h_sub.shape[0] > delay:
            # Normalize
            mu_u_h    = P_h_sub.mean(axis=0, keepdims=True)
            sigma_u_h = P_h_sub.std(axis=0, keepdims=True)
            mu_x_h    = X_h_sub.mean(axis=0, keepdims=True)
            sigma_x_h = X_h_sub.std(axis=0, keepdims=True)

            U_n_h = (P_h_sub - mu_u_h) / sigma_u_h
            X_n_h = (X_h_sub - mu_x_h) / sigma_x_h

            Xd_h, Ud_h = gen_delay_traj(X_n_h, U_n_h, delay)
            A_h, B_h   = iden_koop(Xd_h, Ud_h)
        else:
            print(f"   Warning: {key_h} - Too few rows ({P_h_sub.shape[0]} ≤ {delay}), using zero matrices")
            d = 2 * (delay + 1)
            A_h = np.zeros((d, d))
            B_h = np.zeros((d, 3))
            mu_u_h = np.zeros((1,3)); sigma_u_h = np.ones((1,3))
            mu_x_h = np.zeros((1,2)); sigma_x_h = np.ones((1,2))

        models[key_h] = {
            'A_h':        A_h.tolist(),
            'B_h':        B_h.tolist(),
            'mu_u_h':     mu_u_h.tolist(),
            'sigma_u_h':  sigma_u_h.tolist(),
            'mu_x_h':     mu_x_h.tolist(),
            'sigma_x_h':  sigma_x_h.tolist(),
            'delay':      delay,
            'type':       'Kh_only',
            'n_samples':  P_h_sub.shape[0],
            'exp_sizes':  exp_sizes,
            'sim_sizes':  sim_sizes
        }

    # ================
    # 2) Pure Simulation Models (Kf_only)
    # ================
    print("\n2. Training Pure Simulation Models (Kf_only):")
    
    for n_samples in sim_sizes:
        key_f = f'sim_{n_samples}'
        print(f"   Training {key_f} with {n_samples:,} samples...")
        
        P_f_sub = slice_by_number(P_full_f, n_samples)
        X_f_sub = slice_by_number(X_full_f, n_samples)

        if P_f_sub.shape[0] > delay:
            # Normalize
            mu_u_f    = P_f_sub.mean(axis=0, keepdims=True)
            sigma_u_f = P_f_sub.std(axis=0, keepdims=True)
            mu_x_f    = X_f_sub.mean(axis=0, keepdims=True)
            sigma_x_f = X_f_sub.std(axis=0, keepdims=True)

            U_n_f = (P_f_sub - mu_u_f) / sigma_u_f
            X_n_f = (X_f_sub - mu_x_f) / sigma_x_f

            Xd_f, Ud_f = gen_delay_traj(X_n_f, U_n_f, delay)
            A_f, B_f   = iden_koop(Xd_f, Ud_f)
        else:
            print(f"   Warning: {key_f} - Too few rows ({P_f_sub.shape[0]} ≤ {delay}), using zero matrices")
            d = 2 * (delay + 1)
            A_f = np.zeros((d, d))
            B_f = np.zeros((d, 3))
            mu_u_f = np.zeros((1,3)); sigma_u_f = np.ones((1,3))
            mu_x_f = np.zeros((1,2)); sigma_x_f = np.ones((1,2))

        models[key_f] = {
            'A_f':        A_f.tolist(),
            'B_f':        B_f.tolist(),
            'mu_u_f':     mu_u_f.tolist(),
            'sigma_u_f':  sigma_u_f.tolist(),
            'mu_x_f':     mu_x_f.tolist(),
            'sigma_x_f':  sigma_x_f.tolist(),
            'delay':      delay,
            'type':       'Kf_only',
            'n_samples':  P_f_sub.shape[0],
            'exp_sizes':  exp_sizes,
            'sim_sizes':  sim_sizes
        }

    # ================
    # 3) Combined Models (KfXXhYY)
    # ================
    print("\n3. Training Combined Models:")
    
    for n_sim in sim_sizes:
        for n_exp in exp_sizes:
            key_comb = f'comb_{n_sim}_{n_exp}'
            print(f"   Training {key_comb} with {n_sim:,} sim + {n_exp:,} exp samples...")
            
            P_f_sub = slice_by_number(P_full_f, n_sim)
            X_f_sub = slice_by_number(X_full_f, n_sim)
            P_h_sub = slice_by_number(P_full_h, n_exp)
            X_h_sub = slice_by_number(X_full_h, n_exp)

            # Train Kf on simulation slice
            if P_f_sub.shape[0] > delay:
                mu_u_f    = P_f_sub.mean(axis=0, keepdims=True)
                sigma_u_f = P_f_sub.std(axis=0, keepdims=True)
                mu_x_f    = X_f_sub.mean(axis=0, keepdims=True)
                sigma_x_f = X_f_sub.std(axis=0, keepdims=True)

                U_n_f = (P_f_sub - mu_u_f) / sigma_u_f
                X_n_f = (X_f_sub - mu_x_f) / sigma_x_f
                Xd_f, Ud_f = gen_delay_traj(X_n_f, U_n_f, delay)
                A_f, B_f   = iden_koop(Xd_f, Ud_f)
            else:
                print(f"   Warning: {key_comb} - Too few simulation rows ({P_f_sub.shape[0]} ≤ {delay}), using zero matrices")
                d = 2 * (delay + 1)
                A_f = np.zeros((d, d))
                B_f = np.zeros((d, 3))
                mu_u_f = np.zeros((1, 3)); sigma_u_f = np.ones((1, 3))
                mu_x_f = np.zeros((1, 2)); sigma_x_f = np.ones((1, 2))

            # Train Kh on experiment slice
            if P_h_sub.shape[0] > delay:
                mu_u_h    = P_h_sub.mean(axis=0, keepdims=True)
                sigma_u_h = P_h_sub.std(axis=0, keepdims=True)
                mu_x_h    = X_h_sub.mean(axis=0, keepdims=True)
                sigma_x_h = X_h_sub.std(axis=0, keepdims=True)

                U_n_h = (P_h_sub - mu_u_h) / sigma_u_h
                X_n_h = (X_h_sub - mu_x_h) / sigma_x_h
                Xd_h, Ud_h = gen_delay_traj(X_n_h, U_n_h, delay)
                A_h, B_h   = iden_koop(Xd_h, Ud_h)
            else:
                print(f"   Warning: {key_comb} - Too few experiment rows ({P_h_sub.shape[0]} ≤ {delay}), using zero matrices")
                d = 2 * (delay + 1)
                A_h = np.zeros((d, d))
                B_h = np.zeros((d, 3))
                mu_u_h = np.zeros((1, 3)); sigma_u_h = np.ones((1, 3))
                mu_x_h = np.zeros((1, 2)); sigma_x_h = np.ones((1, 2))

            # Calculate full Koopman matrices from data
            K_f = iden_koop_full(Xd_f, Ud_f, mode=0)  # Full Koopman matrix from simulation data
            K_h = iden_koop_full(Xd_h, Ud_h, mode=0)  # Full Koopman matrix from experimental data

            # Compose Koopman models using K_full = K_f @ K_h @ K_f
            A_full, B_full = compose_koopman_models(K_f, K_h, Xd_f, Ud_f, Xd_h, Ud_h, mode=0)

            models[key_comb] = {
                'A_f':        A_f.tolist(),
                'B_f':        B_f.tolist(),
                'mu_u_f':     mu_u_f.tolist(),
                'sigma_u_f':  sigma_u_f.tolist(),
                'mu_x_f':     mu_x_f.tolist(),
                'sigma_x_f':  sigma_x_f.tolist(),
                'A_h':        A_h.tolist(),
                'B_h':        B_h.tolist(),
                'mu_u_h':     mu_u_h.tolist(),
                'sigma_u_h':  sigma_u_h.tolist(),
                'mu_x_h':     mu_x_h.tolist(),
                'sigma_x_h':  sigma_x_h.tolist(),
                'A_full':     A_full.tolist(),
                'B_full':     B_full.tolist(),
                'delay':      delay,
                'type':       'combined',
                'n_sim_samples': P_f_sub.shape[0],
                'n_exp_samples': P_h_sub.shape[0],
                'exp_sizes':  exp_sizes,
                'sim_sizes':  sim_sizes
            }

    # Save all models
    with open('koopman_models_V2.pkl', 'wb') as f:
        pickle.dump(models, f)

    print(f"\n✅ Trained and saved {len(models)} Koopman models in 'koopman_models_V2.pkl'.")
    
    return models, exp_sizes, sim_sizes, exp_data_val

def compute_model_errors(models, exp_sizes, sim_sizes, exp_filename, sim_filename, exp_data_val=None):
    """
    Compute and return a dictionary of mean errors for all models.
    Use exp_data_val (validation set) for error calculation if provided.
    Uses 5-step prediction error.
    """
    prediction_horizon = 5
    exp_data, sim_data = load_data(exp_filename, sim_filename)
    # Use validation set if provided
    if exp_data_val is not None:
        exp_data_eval = exp_data_val
    else:
        exp_data_eval = exp_data
    P_full_h = exp_data_eval[:, 1:4]
    X_full_h = np.vstack([exp_data_eval[:, 5], exp_data_eval[:, 4]]).T
    model_err_means = {}
    for key, m in models.items():
        delay = m['delay']
        P_h_sub = slice_by_number(P_full_h, max(exp_sizes))
        X_h_sub = slice_by_number(X_full_h, max(exp_sizes))
        if P_h_sub.shape[0] <= delay + prediction_horizon:
            continue
        if m['type'] == 'Kf_only':
            mu_u = np.array(m['mu_u_f']).reshape(1, 3)
            sigma_u = np.array(m['sigma_u_f']).reshape(1, 3)
            mu_x = np.array(m['mu_x_f']).reshape(1, 2)
            sigma_x = np.array(m['sigma_x_f']).reshape(1, 2)
            A = np.array(m['A_f'])
            B = np.array(m['B_f'])
        elif m['type'] == 'Kh_only':
            mu_u = np.array(m['mu_u_h']).reshape(1, 3)
            sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
            mu_x = np.array(m['mu_x_h']).reshape(1, 2)
            sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
            A = np.array(m['A_h'])
            B = np.array(m['B_h'])
        else:
            mu_u = np.array(m['mu_u_h']).reshape(1, 3)
            sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
            mu_x = np.array(m['mu_x_h']).reshape(1, 2)
            sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
            A = np.array(m['A_full'])
            B = np.array(m['B_full'])
        U_n = (P_h_sub - mu_u) / sigma_u
        X_n = (X_h_sub - mu_x) / sigma_x
        Xd, Ud = gen_delay_traj(X_n, U_n, delay)
        M = Xd.shape[0]
        errors = np.zeros(M - prediction_horizon)
        for i_step in range(M - prediction_horizon):
            z_pred = Xd[i_step].reshape(-1, 1)
            # Rollout prediction_horizon steps
            for k in range(prediction_horizon):
                u_k = Ud[i_step + k].reshape(-1, 1)
                z_pred = A.dot(z_pred) + B.dot(u_k)
            x_pred = z_pred[:2, 0]
            x_true = Xd[i_step + prediction_horizon, :2]
            errors[i_step] = np.sum(np.abs(x_pred - x_true))
        mean_err = np.mean(errors)
        model_err_means[key] = mean_err
    return model_err_means

def plot_error_surface(models, exp_sizes, sim_sizes, exp_filename, sim_filename, model_err_means=None):
    """
    Plot the error surface for combined models with pure experimental and simulation lines.
    Accepts precomputed model_err_means if provided.
    """
    if model_err_means is None:
        model_err_means = compute_model_errors(models, exp_sizes, sim_sizes, exp_filename, sim_filename)
    # Load fresh experimental data for error evaluation
    exp_data, sim_data = load_data(exp_filename, sim_filename)
    
    # Extract features
    P_full_h = exp_data[:, 1:4]                    # (N_h × 3)
    X_full_h = np.vstack([exp_data[:, 5], exp_data[:, 4]]).T  # (N_h × 2) = [x=dz, y=dx]
    
    P_full_f = sim_data[:, 1:4]                    # (N_f × 3)
    X_full_f = np.vstack([sim_data[:, 4], sim_data[:, 5]]).T  # (N_f × 2) = [x, y]

    print("\nComputing prediction errors for all models...")
    
    # Store error means for all models
    # model_err_means = {} # This line is now redundant as model_err_means is passed
    
    # Evaluate all models
    # for key, m in models.items(): # This loop is now redundant as model_err_means is passed
    #     delay = m['delay']
        
    #     # Use experimental data for error evaluation
    #     P_h_sub = slice_by_number(P_full_h, max(exp_sizes))
    #     X_h_sub = slice_by_number(X_full_h, max(exp_sizes))
        
    #     if P_h_sub.shape[0] <= delay + 1:
    #         continue
            
    #     if m['type'] == 'Kf_only':
    #         mu_u = np.array(m['mu_u_f']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_f']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_f']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_f']).reshape(1, 2)
    #         A = np.array(m['A_f'])
    #         B = np.array(m['B_f'])
    #     elif m['type'] == 'Kh_only':
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_h'])
    #         B = np.array(m['B_h'])
    #     else:  # Combined model
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_full'])
    #         B = np.array(m['B_full'])
            
    #     U_n = (P_h_sub - mu_u) / sigma_u
    #     X_n = (X_h_sub - mu_x) / sigma_x
    #     Xd, Ud = gen_delay_traj(X_n, U_n, delay)
    #     M = Xd.shape[0]
        
    #     # One-step prediction error
    #     errors = np.zeros(M - 1)
    #     for i_step in range(M - 1):
    #         z = Xd[i_step].reshape(-1, 1)
    #         u_i = Ud[i_step].reshape(-1, 1)
    #         z_next = A.dot(z) + B.dot(u_i)
    #         x_pred = z_next[:2, 0]
    #         x_true = Xd[i_step + 1, :2]
    #         errors[i_step] = np.sum(np.abs(x_pred - x_true))  # L1 norm
            
    #     mean_err = np.mean(errors)
    #     model_err_means[key] = mean_err

    # Create the error surface plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.ticker as ticker

    # Prepare data for surface plot
    exp_counts = np.array(exp_sizes)
    sim_counts = np.array(sim_sizes)
    
    # Use log2 for x and y axes
    exp_counts_log2 = np.log2(exp_counts)
    sim_counts_log2 = np.log2(sim_counts)
    
    # Create meshgrid for surface
    P_EXP, P_SIM = np.meshgrid(exp_counts_log2, sim_counts_log2)
    
    # Fill Z matrix with combined model errors
    Z = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
    for i, n_sim in enumerate(sim_sizes):
        for j, n_exp in enumerate(exp_sizes):
            key = f'comb_{n_sim}_{n_exp}'
            if key in model_err_means:
                Z[i, j] = model_err_means[key]
    
    # Use log10 for error (z axis)
    Z_log10 = np.log10(Z)

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    # Plot pure experiment surface (extended across all simulation sizes)
    pure_exp_errs = []
    for n_exp in exp_sizes:
        key = f'exp_{n_exp}'
        pure_exp_errs.append(model_err_means.get(key, np.nan))
    P_EXP_pure, P_SIM_pure = np.meshgrid(exp_counts_log2, sim_counts_log2)
    Z_exp_pure = np.tile(np.log10(pure_exp_errs).reshape(1, -1), (len(sim_sizes), 1))
    surf_exp = ax.plot_surface(
        P_EXP_pure, P_SIM_pure, Z_exp_pure - 0.02,
        color='red', alpha=0.4, zorder=1, edgecolor='darkred', linewidth=0.5
    )

    # Plot pure simulation surface (extended across all experiment sizes)
    pure_sim_errs = []
    for n_sim in sim_sizes:
        key = f'sim_{n_sim}'
        pure_sim_errs.append(model_err_means.get(key, np.nan))
    P_EXP_pure, P_SIM_pure = np.meshgrid(exp_counts_log2, sim_counts_log2)
    Z_sim_pure = np.tile(np.log10(pure_sim_errs).reshape(-1, 1), (1, len(exp_sizes)))
    surf_sim = ax.plot_surface(
        P_EXP_pure, P_SIM_pure, Z_sim_pure - 0.02,
        color='blue', alpha=0.4, zorder=1, edgecolor='darkblue', linewidth=0.5
    )

    # Plot the combined model surface (on top, more opaque, with thick black edge lines)
    surf = ax.plot_surface(
        P_EXP, P_SIM, Z_log10,
        cmap='viridis', edgecolor='k', linewidth=2, antialiased=True, alpha=0.8, zorder=2
    )

    # Add a thick wireframe overlay for the combined surface
    ax.plot_wireframe(P_EXP, P_SIM, Z_log10, color='k', linewidth=1.5, alpha=0.7, zorder=3)

    # Highlight intersection edges
    pure_exp_errs = []
    for n_exp in exp_sizes:
        key = f'exp_{n_exp}'
        pure_exp_errs.append(model_err_means.get(key, np.nan))
    ax.plot(exp_counts_log2, [sim_counts_log2[0]]*len(exp_counts_log2), np.log10(pure_exp_errs), 'r-', lw=4, zorder=4)
    ax.plot([exp_counts_log2[0]]*len(sim_counts_log2), sim_counts_log2, np.log10(pure_sim_errs), 'b-', lw=4, zorder=4)

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=10, label='log10(Mean |error|)')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Combined Models'),
        Patch(facecolor='red', alpha=0.4, label='Pure Experiment (Extended)'),
        Patch(facecolor='blue', alpha=0.4, label='Pure Simulation (Extended)')
    ]
    ax.legend(handles=legend_elements)

    # Set axis labels and title
    ax.set_xlabel('Experimental Data (log2 scale)')
    ax.set_ylabel('Simulation Data (log2 scale)')
    ax.set_zlabel('Mean Error (log10 scale)')
    ax.set_title('Error Surfaces: Combined Models vs Pure Experimental/Simulation Surfaces')

    # Reduce number of ticks and add grid
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.zaxis.set_major_locator(plt.MaxNLocator(6))
    ax.grid(True)
    plt.show()
    fig.savefig('error_surface.pdf')

def plot_error_curves_2d(models, exp_sizes, sim_sizes, exp_filename, sim_filename, model_err_means=None):
    """
    Plot 2D error curves for all available simulation data sizes.
    Accepts precomputed model_err_means if provided.
    """
    if model_err_means is None:
        model_err_means = compute_model_errors(models, exp_sizes, sim_sizes, exp_filename, sim_filename)
    # Load fresh experimental data for error evaluation
    exp_data, sim_data = load_data(exp_filename, sim_filename)
    
    # Extract features
    P_full_h = exp_data[:, 1:4]                    # (N_h × 3)
    X_full_h = np.vstack([exp_data[:, 5], exp_data[:, 4]]).T  # (N_h × 2) = [x=dz, y=dx]
    
    P_full_f = sim_data[:, 1:4]                    # (N_f × 3)
    X_full_f = np.vstack([sim_data[:, 4], sim_data[:, 5]]).T  # (N_f × 2) = [x, y]

    print("\nComputing prediction errors for 2D curve plots...")
    
    # Store error means for all models
    # model_err_means = {} # This line is now redundant as model_err_means is passed
    
    # Evaluate all models
    # for key, m in models.items(): # This loop is now redundant as model_err_means is passed
    #     delay = m['delay']
        
    #     # Use experimental data for error evaluation
    #     P_h_sub = slice_by_number(P_full_h, max(exp_sizes))
    #     X_h_sub = slice_by_number(X_full_h, max(exp_sizes))
        
    #     if P_h_sub.shape[0] <= delay + 1:
    #         continue
            
    #     if m['type'] == 'Kf_only':
    #         mu_u = np.array(m['mu_u_f']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_f']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_f']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_f']).reshape(1, 2)
    #         A = np.array(m['A_f'])
    #         B = np.array(m['B_f'])
    #     elif m['type'] == 'Kh_only':
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_h'])
    #         B = np.array(m['B_h'])
    #     else:  # Combined model
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_full'])
    #         B = np.array(m['B_full'])
            
    #     U_n = (P_h_sub - mu_u) / sigma_u
    #     X_n = (X_h_sub - mu_x) / sigma_x
    #     Xd, Ud = gen_delay_traj(X_n, U_n, delay)
    #     M = Xd.shape[0]
        
    #     # One-step prediction error
    #     errors = np.zeros(M - 1)
    #     for i_step in range(M - 1):
    #         z = Xd[i_step].reshape(-1, 1)
    #         u_i = Ud[i_step].reshape(-1, 1)
    #         z_next = A.dot(z) + B.dot(u_i)
    #         x_pred = z_next[:2, 0]
    #         x_true = Xd[i_step + 1, :2]
    #         errors[i_step] = np.sum(np.abs(x_pred - x_true))  # L1 norm
            
    #     mean_err = np.mean(errors)
    #     model_err_means[key] = mean_err

    # Define specific simulation sizes to plot
    # target_sim_sizes = [2**5, 2**7, 2**9, 2**11, 2**13, 2**15, 2**17] # This line is now redundant
    # Add full simulation dataset size if it exists # This line is now redundant
    # full_sim_size = sim_data.shape[0] # This line is now redundant
    # if full_sim_size not in target_sim_sizes: # This line is now redundant
    #     target_sim_sizes.append(full_sim_size) # This line is now redundant
    
    # Filter to only include sizes that exist in our data # This line is now redundant
    # available_sim_sizes = [size for size in target_sim_sizes if size <= full_sim_size] # This line is now redundant
    
    # Create subplots
    # n_plots = len(available_sim_sizes) # This line is now redundant
    cols = 3
    rows = (len(sim_sizes) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sim_size in enumerate(sim_sizes):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get experimental error curve (pure experiment models)
        exp_errors = []
        exp_sizes_available = []
        for n_exp in exp_sizes:
            key = f'exp_{n_exp}'
            if key in model_err_means:
                exp_errors.append(model_err_means[key])
                exp_sizes_available.append(n_exp)
        
        # Get simulation error (pure simulation model for this size)
        sim_key = f'sim_{sim_size}'
        sim_error = model_err_means.get(sim_key, np.nan)
        
        # Get combined model errors for this simulation size
        comb_errors = []
        exp_sizes_comb = []
        for n_exp in exp_sizes:
            key = f'comb_{sim_size}_{n_exp}'
            if key in model_err_means:
                comb_errors.append(model_err_means[key])
                exp_sizes_comb.append(n_exp)
        
        # Plot experimental error curve (red)
        if exp_errors:
            ax.semilogy(exp_sizes_available, exp_errors, 'r-', linewidth=2, 
                       label='Pure Experiment', marker='o', markersize=4)
        
        # Plot simulation error line (blue dashed)
        if not np.isnan(sim_error):
            ax.axhline(y=sim_error, color='blue', linestyle='--', linewidth=2,
                      label=f'Pure Simulation ({sim_size:,} samples)')
        
        # Plot combined model curve (black)
        if comb_errors:
            ax.semilogy(exp_sizes_comb, comb_errors, 'k-', linewidth=2,
                       label='Combined Models', marker='s', markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Experimental Data Size')
        ax.set_ylabel('Mean Error (L1 norm)')
        ax.set_title(f'Simulation Data Size: {sim_size:,}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis as well
        ax.set_xscale('log', base=2)
        
        # Add text with simulation error value
        if not np.isnan(sim_error):
            ax.text(0.02, 0.98, f'Sim Error: {sim_error:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(sim_sizes), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('error_curves_2d_simulation_slice.pdf')
    plt.show()

def plot_error_curves_2d_simulation_slice(models, exp_sizes, sim_sizes, exp_filename, sim_filename, model_err_means=None):
    """
    Plot 2D error curves for all available experimental data sizes.
    Accepts precomputed model_err_means if provided.
    """
    if model_err_means is None:
        model_err_means = compute_model_errors(models, exp_sizes, sim_sizes, exp_filename, sim_filename)
    # Load fresh experimental data for error evaluation
    exp_data, sim_data = load_data(exp_filename, sim_filename)
    
    # Extract features
    P_full_h = exp_data[:, 1:4]                    # (N_h × 3)
    X_full_h = np.vstack([exp_data[:, 5], exp_data[:, 4]]).T  # (N_h × 2) = [x=dz, y=dx]
    
    P_full_f = sim_data[:, 1:4]                    # (N_f × 3)
    X_full_f = np.vstack([sim_data[:, 4], sim_data[:, 5]]).T  # (N_f × 2) = [x, y]

    print("\nComputing prediction errors for 2D simulation slice plots...")
    
    # Store error means for all models
    # model_err_means = {} # This line is now redundant as model_err_means is passed
    
    # Evaluate all models
    # for key, m in models.items(): # This loop is now redundant as model_err_means is passed
    #     delay = m['delay']
        
    #     # Use experimental data for error evaluation
    #     P_h_sub = slice_by_number(P_full_h, max(exp_sizes))
    #     X_h_sub = slice_by_number(X_full_h, max(exp_sizes))
        
    #     if P_h_sub.shape[0] <= delay + 1:
    #         continue
            
    #     if m['type'] == 'Kf_only':
    #         mu_u = np.array(m['mu_u_f']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_f']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_f']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_f']).reshape(1, 2)
    #         A = np.array(m['A_f'])
    #         B = np.array(m['B_f'])
    #     elif m['type'] == 'Kh_only':
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_h'])
    #         B = np.array(m['B_h'])
    #     else:  # Combined model
    #         mu_u = np.array(m['mu_u_h']).reshape(1, 3)
    #         sigma_u = np.array(m['sigma_u_h']).reshape(1, 3)
    #         mu_x = np.array(m['mu_x_h']).reshape(1, 2)
    #         sigma_x = np.array(m['sigma_x_h']).reshape(1, 2)
    #         A = np.array(m['A_full'])
    #         B = np.array(m['B_full'])
            
    #     U_n = (P_h_sub - mu_u) / sigma_u
    #     X_n = (X_h_sub - mu_x) / sigma_x
    #     Xd, Ud = gen_delay_traj(X_n, U_n, delay)
    #     M = Xd.shape[0]
        
    #     # One-step prediction error
    #     errors = np.zeros(M - 1)
    #     for i_step in range(M - 1):
    #         z = Xd[i_step].reshape(-1, 1)
    #         u_i = Ud[i_step].reshape(-1, 1)
    #         z_next = A.dot(z) + B.dot(u_i)
    #         x_pred = z_next[:2, 0]
    #         x_true = Xd[i_step + 1, :2]
    #         errors[i_step] = np.sum(np.abs(x_pred - x_true))  # L1 norm
            
    #     mean_err = np.mean(errors)
    #     model_err_means[key] = mean_err

    # Define specific experimental sizes to plot
    # target_exp_sizes = [2**5, 2**7, 2**9, 2**11, 2**13, 2**15, 2**17] # This line is now redundant
    # Add full experimental dataset size if it exists # This line is now redundant
    # full_exp_size = exp_data.shape[0] # This line is now redundant
    # if full_exp_size not in target_exp_sizes: # This line is now redundant
    #     target_exp_sizes.append(full_exp_size) # This line is now redundant
    
    # Filter to only include sizes that exist in our data # This line is now redundant
    # available_exp_sizes = [size for size in target_exp_sizes if size <= full_exp_size] # This line is now redundant
    
    # Create subplots
    # n_plots = len(available_exp_sizes) # This line is now redundant
    cols = 4
    rows = (len(exp_sizes) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, exp_size in enumerate(exp_sizes):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get simulation error curve (pure simulation models)
        sim_errors = []
        sim_sizes_available = []
        for n_sim in sim_sizes:
            key = f'sim_{n_sim}'
            if key in model_err_means:
                sim_errors.append(model_err_means[key])
                sim_sizes_available.append(n_sim)
        
        # Get experimental error (pure experimental model for this size)
        exp_key = f'exp_{exp_size}'
        exp_error = model_err_means.get(exp_key, np.nan)
        
        # Get combined model errors for this experimental size
        comb_errors = []
        sim_sizes_comb = []
        for n_sim in sim_sizes:
            key = f'comb_{n_sim}_{exp_size}'
            if key in model_err_means:
                comb_errors.append(model_err_means[key])
                sim_sizes_comb.append(n_sim)
        
        # Plot simulation error curve (blue)
        if sim_errors:
            ax.semilogy(sim_sizes_available, sim_errors, 'b-', linewidth=2, 
                       label='Pure Simulation', marker='o', markersize=4)
        
        # Plot experimental error line (red dashed)
        if not np.isnan(exp_error):
            ax.axhline(y=exp_error, color='red', linestyle='--', linewidth=2,
                      label=f'Pure Experiment ({exp_size:,} samples)')
        
        # Plot combined model curve (black)
        if comb_errors:
            ax.semilogy(sim_sizes_comb, comb_errors, 'k-', linewidth=2,
                       label='Combined Models', marker='s', markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Simulation Data Size')
        ax.set_ylabel('Mean Error (L1 norm)')
        ax.set_title(f'Experimental Data Size: {exp_size:,}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis as well
        ax.set_xscale('log', base=2)
        
        # Add text with experimental error value
        if not np.isnan(exp_error):
            ax.text(0.02, 0.98, f'Exp Error: {exp_error:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(exp_sizes), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('error_curves_2d_experimental_slice.pdf')
    plt.show()

# ============================================================================
# Bruder's Method Comparison Functions
# ============================================================================

def identify_K_full_with_inputs(X_delay, U_delay, mode=0, alpha=1e-3, tol=1e-4):
    """
    Identify the full Koopman matrix K including inputs: shape (d+p, d+p)
    where d = D*(delay+1) is the lifted state dimension and p is input dimension.
    
    Args:
        X_delay: [M × d] - delayed state coordinates
        U_delay: [M × p] - delayed input coordinates
        mode: 0 = least-squares, 1 = Lasso, 2 = Ridge
        alpha: regularization parameter
        tol: tolerance
        
    Returns:
        K_full: [(d+p) × (d+p)] - full Koopman matrix mapping [X_delay, U_delay] -> [X_delay_next, U_delay_next]
    """
    # Form regression matrices: Y = [X_delay[:-1], U_delay[:-1]], Z = [X_delay[1:], U_delay[1:]]
    Y = np.hstack([X_delay[:-1], U_delay[:-1]])  # Shape: (M-1, d+p)
    Z = np.hstack([X_delay[1:], U_delay[1:]])   # Shape: (M-1, d+p)
    
    d = X_delay.shape[1]  # dimension of lifted state
    p = U_delay.shape[1]  # dimension of input
    total_dim = d + p
    
    if mode == 0:
        K = np.linalg.lstsq(Y, Z, rcond=None)[0].T  # Shape: (d+p, d+p)
    elif mode == 1:
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=alpha, fit_intercept=False, tol=tol, max_iter=10000)
        clf.fit(Y, Z)
        K = clf.coef_  # Shape: (d+p, d+p)
    else:
        clf = Ridge(alpha=alpha, fit_intercept=False, tol=tol, max_iter=10000)
        clf.fit(Y, Z)
        K = clf.coef_  # Shape: (d+p, d+p)
    
    # Ensure K has the correct shape
    if K.shape != (total_dim, total_dim):
        # Pad or truncate if necessary
        K_full = np.zeros((total_dim, total_dim))
        min_rows = min(K.shape[0], total_dim)
        min_cols = min(K.shape[1], total_dim)
        K_full[:min_rows, :min_cols] = K[:min_rows, :min_cols]
        return K_full
    
    return K

def normalize_with_exp_stats(X_raw, U_raw, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h):
    """
    Normalize raw data with experimental normalization (mu, sigma).
    
    Args:
        X_raw: (N, D) - raw state data
        U_raw: (N, p) - raw input data
        mu_x_h: (1, D) - experimental state mean
        sigma_x_h: (1, D) - experimental state std
        mu_u_h: (1, p) - experimental input mean
        sigma_u_h: (1, p) - experimental input std
        
    Returns:
        X_n: (N, D) - normalized state data
        U_n: (N, p) - normalized input data
    """
    # Avoid dividing by zero
    sigma_x_h_safe = np.where(sigma_x_h < 1e-10, 1.0, sigma_x_h)
    sigma_u_h_safe = np.where(sigma_u_h < 1e-10, 1.0, sigma_u_h)
    X_n = (X_raw - mu_x_h) / sigma_x_h_safe
    U_n = (U_raw - mu_u_h) / sigma_u_h_safe
    return X_n, U_n

def extract_A_B_from_Kfull(K_full, d, p):
    """
    Extract A (d×d) and B (d×p) from full Koopman matrix K_full (d+p, d+p).
    
    Args:
        K_full: (d+p, d+p) - full Koopman matrix
        d: int - state dimension
        p: int - input dimension
        
    Returns:
        A: (d, d) - state transition matrix
        B: (d, p) - input matrix
    """
    A = K_full[:d, :d]
    B = K_full[:d, d:d+p]
    return A, B

def evaluate_K_on_exp_validation(K_full, exp_val_data, delay, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h, 
                                 prediction_horizon=5, max_exp_size=None):
    """
    Compute mean prediction error for a full Koopman matrix K_full on experimental validation data.
    Uses the same error metric as compute_model_errors (L1 norm, 5-step prediction).
    Matches PIKO's error computation exactly.
    
    Args:
        K_full: (d+p, d+p) or (d, d) - full Koopman matrix
        exp_val_data: (N, 6) - experimental validation data [ts_rel, P1, P2, P3, dx, dz]
        delay: int - delay coordinate parameter
        mu_x_h, sigma_x_h: (1, 2) - experimental state normalization
        mu_u_h, sigma_u_h: (1, 3) - experimental input normalization
        prediction_horizon: int - prediction steps ahead (default 5 to match compute_model_errors)
        max_exp_size: int - maximum experimental size to use for slicing (matches compute_model_errors)
        
    Returns:
        mean_err: float - mean L1 error
    """
    # Prepare validation experimental data (raw) - same as compute_model_errors
    P_full_h = exp_val_data[:, 1:4]                    # (N_h × 3)
    X_full_h = np.vstack([exp_val_data[:, 5], exp_val_data[:, 4]]).T  # (N_h × 2) = [x=dz, y=dx]

    # Slice data if max_exp_size is provided (to match compute_model_errors behavior)
    if max_exp_size is not None:
        P_h_sub = slice_by_number(P_full_h, max_exp_size)
        X_h_sub = slice_by_number(X_full_h, max_exp_size)
    else:
        P_h_sub = P_full_h
        X_h_sub = X_full_h

    # Normalize validation data using experimental stats (same as compute_model_errors)
    # Note: compute_model_errors uses model's normalization, but here we use experimental normalization
    # which is consistent with how Bruder method works
    U_n = (P_h_sub - mu_u_h) / np.where(sigma_u_h < 1e-10, 1.0, sigma_u_h)
    X_n = (X_h_sub - mu_x_h) / np.where(sigma_x_h < 1e-10, 1.0, sigma_x_h)

    Xd, Ud = gen_delay_traj(X_n, U_n, delay)
    d_dim = Xd.shape[1]  # d
    p = U_n.shape[1]  # p = 3

    # Extract A, B from K_full
    if K_full.shape[0] == (d_dim + p):
        # Full matrix including inputs
        A, B = extract_A_B_from_Kfull(K_full, d_dim, p)
    elif K_full.shape[0] == d_dim:
        # Only state part, assume B is zero
        A = K_full[:d_dim, :d_dim]
        B = np.zeros((d_dim, p))
    else:
        # Try to extract what we can
        min_dim = min(K_full.shape[0], d_dim)
        A = K_full[:min_dim, :min_dim]
        if K_full.shape[1] >= d_dim + p:
            B = K_full[:min_dim, d_dim:d_dim+p]
        else:
            B = np.zeros((min_dim, p))
        # Pad if necessary
        if A.shape[0] < d_dim:
            A_padded = np.zeros((d_dim, d_dim))
            A_padded[:A.shape[0], :A.shape[1]] = A
            A = A_padded
        if B.shape[0] < d_dim:
            B_padded = np.zeros((d_dim, p))
            B_padded[:B.shape[0], :B.shape[1]] = B
            B = B_padded

    M = Xd.shape[0]
    errors = np.zeros(M - prediction_horizon)
    
    for i_step in range(M - prediction_horizon):
        z_pred = Xd[i_step].reshape(-1, 1)
        # Rollout prediction_horizon steps
        for k in range(prediction_horizon):
            u_k = Ud[i_step + k].reshape(-1, 1)
            z_pred = A.dot(z_pred) + B.dot(u_k)
        
        # Extract predicted and true normalized states
        x_pred = z_pred[:2, 0]
        x_true = Xd[i_step + prediction_horizon, :2]
        
        # Use L1 norm (same as V2)
        errors[i_step] = np.sum(np.abs(x_pred - x_true))

    mean_err = np.mean(errors) if len(errors) > 0 else np.nan
    return mean_err

def compute_bruder_surfaces_and_plot_combined(sim_filename, exp_filename, sim_sizes, exp_sizes, delay,
                                              exp_data_val, exp_data_train,
                                              gammas=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99],
                                              model_err_means=None,
                                              save_pdf='comparison_surfaces_bruder.pdf'):
    """
    Computes Bruder blend surfaces and plots them with pure simulation and pure experiment surfaces.
    Uses the same axis scaling as plot_error_surface in V2:
    - x-axis: log2(exp_counts)
    - y-axis: log2(sim_counts)
    - z-axis: log10(error)
    
    Labels are placed in the same column (vertically aligned).
    
    Args:
        sim_filename: simulation data filename
        exp_filename: experimental data filename
        sim_sizes: list of simulation data sizes
        exp_sizes: list of experimental data sizes
        delay: delay coordinate parameter
        exp_data_val: experimental validation data
        exp_data_train: experimental training data (for normalization)
        gammas: list of blending factors for Bruder method
        model_err_means: dict with keys 'sim_{n_sim}', 'exp_{n_exp}' for pure model errors
        save_pdf: output filename
    """
    print("\n" + "="*80)
    print("Computing Bruder's Method Comparison Surfaces")
    print("="*80)

    # --- 1) Build experimental normalization from exp_data_train ---
    P_full_h_train = exp_data_train[:, 1:4]
    X_full_h_train = np.vstack([exp_data_train[:, 5], exp_data_train[:, 4]]).T
    mu_u_h = P_full_h_train.mean(axis=0, keepdims=True)
    sigma_u_h = P_full_h_train.std(axis=0, keepdims=True)
    mu_x_h = X_full_h_train.mean(axis=0, keepdims=True)
    sigma_x_h = X_full_h_train.std(axis=0, keepdims=True)
    
    # Handle zero standard deviations
    sigma_u_h = np.where(sigma_u_h < 1e-10, 1.0, sigma_u_h)
    sigma_x_h = np.where(sigma_x_h < 1e-10, 1.0, sigma_x_h)
    
    print(f"Experimental normalization computed from {exp_data_train.shape[0]:,} training samples")

    # --- 2) Preload sim and exp full raw arrays ---
    exp_full, sim_full = load_data(exp_filename, sim_filename)
    
    print(f"Loaded datasets: exp shape={exp_full.shape}, sim shape={sim_full.shape}")

    # --- 3) Build/calc K_sim and K_exp caches ---
    cache_K_sim = {}
    cache_K_exp = {}

    print(f"\nComputing K_sim matrices (in experimental-normalized coordinates)...")
    for n_sim in sim_sizes:
        if n_sim > sim_full.shape[0]:
            print(f"  Skipping sim_size={n_sim} (exceeds available data: {sim_full.shape[0]})")
            continue
            
        sim_data_raw = sim_full[:n_sim, :]
        
        # Extract features: simulation format is time,p1,p2,p3,x,y
        P_sim_raw = sim_data_raw[:, 1:4]
        X_sim_raw = np.vstack([sim_data_raw[:, 4], sim_data_raw[:, 5]]).T

        if P_sim_raw.shape[0] <= delay + 10:
            print(f"  Skipping sim_size={n_sim} (too few samples: {P_sim_raw.shape[0]} <= {delay + 10})")
            continue

        Xn_sim, Un_sim = normalize_with_exp_stats(X_sim_raw, P_sim_raw, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h)
        Xd_sim, Ud_sim = gen_delay_traj(Xn_sim, Un_sim, delay)
        K_sim = identify_K_full_with_inputs(Xd_sim, Ud_sim, mode=0)
        cache_K_sim[n_sim] = K_sim
        print(f"  Computed K_sim for n_sim={n_sim:,} (shape: {K_sim.shape})")

    print(f"\nComputing K_exp matrices (in experimental-normalized coordinates)...")
    for n_exp in exp_sizes:
        if n_exp > exp_data_train.shape[0]:
            print(f"  Skipping exp_size={n_exp} (exceeds available training data: {exp_data_train.shape[0]})")
            continue
            
        exp_data_raw = exp_data_train[:n_exp, :]
        P_exp_raw = exp_data_raw[:, 1:4]
        X_exp_raw = np.vstack([exp_data_raw[:, 5], exp_data_raw[:, 4]]).T

        if P_exp_raw.shape[0] <= delay + 10:
            print(f"  Skipping exp_size={n_exp} (too few samples: {P_exp_raw.shape[0]} <= {delay + 10})")
            continue

        Xn_exp, Un_exp = normalize_with_exp_stats(X_exp_raw, P_exp_raw, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h)
        Xd_exp, Ud_exp = gen_delay_traj(Xn_exp, Un_exp, delay)
        K_exp = identify_K_full_with_inputs(Xd_exp, Ud_exp, mode=0)
        cache_K_exp[n_exp] = K_exp
        print(f"  Computed K_exp for n_exp={n_exp:,} (shape: {K_exp.shape})")

    # --- 4) Evaluate Bruder blends --- 
    gamma_results = {gamma: np.full((len(sim_sizes), len(exp_sizes)), np.nan) for gamma in gammas}

    print(f"\nEvaluating Bruder blended models for {len(gammas)} gamma values...")
    for i_sim, n_sim in enumerate(sim_sizes):
        if n_sim not in cache_K_sim:
            continue
        for j_exp, n_exp in enumerate(exp_sizes):
            if n_exp not in cache_K_exp:
                continue
                
            K_sim = cache_K_sim[n_sim]
            K_exp = cache_K_exp[n_exp]

            # Align shapes if needed (pad/truncate)
            if K_sim.shape != K_exp.shape:
                maxdim = max(K_sim.shape[0], K_exp.shape[0])
                K_sim_pad = np.zeros((maxdim, maxdim))
                K_exp_pad = np.zeros((maxdim, maxdim))
                K_sim_pad[:K_sim.shape[0], :K_sim.shape[1]] = K_sim
                K_exp_pad[:K_exp.shape[0], :K_exp.shape[1]] = K_exp
                K_sim = K_sim_pad
                K_exp = K_exp_pad

            for gamma in gammas:
                K_bruder = (1.0 - gamma) * K_sim + gamma * K_exp
                mean_err = evaluate_K_on_exp_validation(K_bruder, exp_data_val, delay,
                                                        mu_x_h, sigma_x_h, mu_u_h, sigma_u_h,
                                                        prediction_horizon=5)
                gamma_results[gamma][i_sim, j_exp] = mean_err
                
        if (i_sim + 1) % max(1, len(sim_sizes) // 5) == 0:
            print(f"  Progress: {i_sim+1}/{len(sim_sizes)} simulation sizes completed")

    # --- 5) Prepare pure simulation and pure experimental surfaces ---
    # Pure simulation surface: extend across all experimental sizes (constant for each sim_size)
    Z_pure_sim = None
    if model_err_means is not None:
        pure_sim_errs = []
        for n_sim in sim_sizes:
            key = f'sim_{n_sim}'
            if key in model_err_means:
                pure_sim_errs.append(model_err_means[key])
            else:
                pure_sim_errs.append(np.nan)
        
        # Create extended surface: each row (sim_size) has constant error across all exp_sizes
        Z_pure_sim = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
        for i_sim, err in enumerate(pure_sim_errs):
            if not np.isnan(err):
                Z_pure_sim[i_sim, :] = err  # Extend across all exp_sizes
        print(f"Pure simulation surface prepared: {np.sum(~np.isnan(Z_pure_sim))} valid points")
    
    # Pure experimental surface: extend across all simulation sizes (constant for each exp_size)
    Z_pure_exp = None
    if model_err_means is not None:
        pure_exp_errs = []
        for n_exp in exp_sizes:
            key = f'exp_{n_exp}'
            if key in model_err_means:
                pure_exp_errs.append(model_err_means[key])
            else:
                pure_exp_errs.append(np.nan)
        
        # Create extended surface: each column (exp_size) has constant error across all sim_sizes
        Z_pure_exp = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
        for j_exp, err in enumerate(pure_exp_errs):
            if not np.isnan(err):
                Z_pure_exp[:, j_exp] = err  # Extend across all sim_sizes
        print(f"Pure experimental surface prepared: {np.sum(~np.isnan(Z_pure_exp))} valid points")

    # --- 6) Prepare mesh in log-space for plotting (same as V2) ---
    exp_counts = np.array(exp_sizes)
    sim_counts = np.array(sim_sizes)
    
    # Use log2 for x and y axes (same as V2)
    exp_counts_log2 = np.log2(exp_counts)
    sim_counts_log2 = np.log2(sim_counts)
    
    # Create meshgrid for surface
    P_EXP, P_SIM = np.meshgrid(exp_counts_log2, sim_counts_log2)

    # helper safe log10
    def safe_log10(Z):
        Zsafe = np.array(Z, dtype=float)
        Zsafe[~np.isfinite(Zsafe)] = np.nan
        small = 1e-12
        Zsafe = np.where(Zsafe <= 0, small, Zsafe)
        return np.log10(Zsafe)

    # --- 7) Create 3D plot with all surfaces (same style as V2) ---
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot pure experiment surface (extended across all simulation sizes) - same as V2
    if Z_pure_exp is not None:
        Z_exp_pure_log = safe_log10(Z_pure_exp)
        surf_exp = ax.plot_surface(
            P_EXP, P_SIM, Z_exp_pure_log - 0.02,
            color='red', alpha=0.4, zorder=1, edgecolor='darkred', linewidth=0.5
        )

    # Plot pure simulation surface (extended across all experiment sizes) - same as V2
    if Z_pure_sim is not None:
        Z_sim_pure_log = safe_log10(Z_pure_sim)
        surf_sim = ax.plot_surface(
            P_EXP, P_SIM, Z_sim_pure_log - 0.02,
            color='blue', alpha=0.4, zorder=1, edgecolor='darkblue', linewidth=0.5
        )

    # Plot Bruder surfaces for each gamma
    # Use green colormap that goes from bright (small gamma) to dark (large gamma)
    # Using Greens colormap: light green -> dark green
    # Equal split: map each gamma to an equal portion of the range [0.4, 0.9]
    import matplotlib.cm as cm
    n_gammas = len(gammas)
    # Equal split: each gamma gets an equal portion of [0.4, 0.9]
    # First gamma (index 0) -> 0.4, last gamma (index n-1) -> 0.9
    colors = [cm.Greens(0.4 + (0.9 - 0.4) * idx / (n_gammas - 1)) for idx in range(n_gammas)]
    
    for idx, gamma in enumerate(gammas):
        Z_gamma = gamma_results[gamma]
        Z_gamma_log = safe_log10(Z_gamma)
        color = colors[idx]
        
        # Plot surface with same style as combined model in V2
        surf = ax.plot_surface(
            P_EXP, P_SIM, Z_gamma_log,
            color=color, alpha=0.6, zorder=2, edgecolor='k', linewidth=0.5, antialiased=True
        )
        
        # Add label in the same column (vertically aligned) - use a fixed x position
        # Place labels at the rightmost x position (max exp size)
        x_label = exp_counts_log2[-1]
        # Distribute y positions evenly
        y_positions = np.linspace(sim_counts_log2[0], sim_counts_log2[-1], len(gammas))
        y_label = y_positions[idx]
        
        # Get z value at this position
        i_pos = len(sim_sizes) - 1  # Last row
        j_pos = len(exp_sizes) - 1  # Last column
        z_label = Z_gamma_log[i_pos, j_pos]
        
        # Add text label
        ax.text(x_label, y_label, z_label + 0.1, f'γ={gamma:.2f}', 
               color=color, fontsize=10, weight='bold', zorder=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.8, linewidth=1.5))

    # Set axis labels and title (same as V2)
    ax.set_xlabel('Experimental Data (log2 scale)', fontsize=12)
    ax.set_ylabel('Simulation Data (log2 scale)', fontsize=12)
    ax.set_zlabel('Mean Error (log10 scale)', fontsize=12)
    ax.set_title('Bruder\'s Method: K_bruder = (1-γ)K_sim + γ*K_exp', fontsize=14, fontweight='bold')

    # Reduce number of ticks and add grid (same as V2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.zaxis.set_major_locator(plt.MaxNLocator(6))
    ax.grid(True)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = []
    if Z_pure_exp is not None:
        legend_elements.append(Patch(facecolor='red', alpha=0.4, label='Pure Experiment (Extended)'))
    if Z_pure_sim is not None:
        legend_elements.append(Patch(facecolor='blue', alpha=0.4, label='Pure Simulation (Extended)'))
    for idx, gamma in enumerate(gammas):
        color = colors[idx]
        legend_elements.append(Patch(facecolor=color, alpha=0.6, label=f'Bruder γ={gamma:.2f}'))
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    fig.savefig(save_pdf, dpi=300, bbox_inches='tight')
    print(f"\n✅ Bruder comparison surfaces saved to {save_pdf}")
    plt.show()

    return {
        'gamma_results': gamma_results,
        'sim_sizes': sim_sizes,
        'exp_sizes': exp_sizes,
        'mu_x_h': mu_x_h, 'sigma_x_h': sigma_x_h, 'mu_u_h': mu_u_h, 'sigma_u_h': sigma_u_h,
        'K_sim_cache': cache_K_sim,
        'K_exp_cache': cache_K_exp
    }

def plot_bruder_individual_comparisons(gamma_results, Z_pure_sim, Z_pure_exp, sim_sizes, exp_sizes,
                                       gammas=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99],
                                       save_pdf='bruder_individual_comparisons.pdf'):
    """
    Plot individual comparisons for each gamma value in a 3x2 subplot layout.
    Each subplot shows the Bruder surface for that gamma compared to pure experiment and pure simulation.
    
    Args:
        gamma_results: dict with keys as gamma values, values as error matrices
        Z_pure_sim: pure simulation error matrix (extended surface)
        Z_pure_exp: pure experimental error matrix (extended surface)
        sim_sizes: list of simulation data sizes
        exp_sizes: list of experimental data sizes
        gammas: list of gamma values to plot
        save_pdf: output filename
    """
    print("\n" + "="*80)
    print("Generating Individual Bruder Comparison Subplots (3x2)")
    print("="*80)
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm
    
    # Prepare mesh in log-space (same as V2)
    exp_counts = np.array(exp_sizes)
    sim_counts = np.array(sim_sizes)
    exp_counts_log2 = np.log2(exp_counts)
    sim_counts_log2 = np.log2(sim_counts)
    P_EXP, P_SIM = np.meshgrid(exp_counts_log2, sim_counts_log2)
    
    # Helper function for safe log10
    def safe_log10(Z):
        Zsafe = np.array(Z, dtype=float)
        Zsafe[~np.isfinite(Zsafe)] = np.nan
        small = 1e-12
        Zsafe = np.where(Zsafe <= 0, small, Zsafe)
        return np.log10(Zsafe)
    
    # Get colors for gamma surfaces (same as combined plot - green gradient)
    # Equal split: map each gamma to an equal portion of the range [0.4, 0.9]
    n_gammas = len(gammas)
    # Equal split: each gamma gets an equal portion of [0.4, 0.9]
    colors = [cm.Greens(0.4 + (0.9 - 0.4) * idx / (n_gammas - 1)) for idx in range(n_gammas)]
    
    # Create 3x2 subplot figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot pure surfaces for reference (will be used in all subplots)
    Z_exp_pure_log = safe_log10(Z_pure_exp) if Z_pure_exp is not None else None
    Z_sim_pure_log = safe_log10(Z_pure_sim) if Z_pure_sim is not None else None
    
    # Find global z limits for consistent scaling across all subplots
    z_min = np.inf
    z_max = -np.inf
    
    # Check pure surfaces
    if Z_exp_pure_log is not None:
        z_min = min(z_min, np.nanmin(Z_exp_pure_log))
        z_max = max(z_max, np.nanmax(Z_exp_pure_log))
    if Z_sim_pure_log is not None:
        z_min = min(z_min, np.nanmin(Z_sim_pure_log))
        z_max = max(z_max, np.nanmax(Z_sim_pure_log))
    
    # Check all gamma surfaces
    for gamma in gammas:
        if gamma in gamma_results:
            Z_gamma_log = safe_log10(gamma_results[gamma])
            z_min = min(z_min, np.nanmin(Z_gamma_log))
            z_max = max(z_max, np.nanmax(Z_gamma_log))
    
    # Add some padding
    z_range = z_max - z_min
    z_min -= 0.1 * z_range
    z_max += 0.1 * z_range
    
    # Plot each gamma in a subplot
    for idx, gamma in enumerate(gammas):
        row = idx // 2  # 0, 0, 1, 1, 2, 2
        col = idx % 2   # 0, 1, 0, 1, 0, 1
        ax = fig.add_subplot(3, 2, idx + 1, projection='3d')
        
        color = colors[idx]
        Z_gamma = gamma_results[gamma]
        Z_gamma_log = safe_log10(Z_gamma)
        
        # Plot pure experiment surface (red, behind)
        if Z_exp_pure_log is not None:
            ax.plot_surface(
                P_EXP, P_SIM, Z_exp_pure_log - 0.02,
                color='red', alpha=0.4, zorder=1, edgecolor='darkred', linewidth=0.5
            )
        
        # Plot pure simulation surface (blue, behind)
        if Z_sim_pure_log is not None:
            ax.plot_surface(
                P_EXP, P_SIM, Z_sim_pure_log - 0.02,
                color='blue', alpha=0.4, zorder=1, edgecolor='darkblue', linewidth=0.5
            )
        
        # Plot Bruder surface for this gamma (on top)
        ax.plot_surface(
            P_EXP, P_SIM, Z_gamma_log,
            color=color, alpha=0.7, zorder=2, edgecolor='k', linewidth=0.5, antialiased=True
        )
        
        # Set consistent axis limits
        ax.set_xlim(exp_counts_log2[0], exp_counts_log2[-1])
        ax.set_ylim(sim_counts_log2[0], sim_counts_log2[-1])
        ax.set_zlim(z_min, z_max)
        
        # Set labels
        ax.set_xlabel('Exp Data (log2)', fontsize=9)
        ax.set_ylabel('Sim Data (log2)', fontsize=9)
        ax.set_zlabel('Error (log10)', fontsize=9)
        ax.set_title(f'Bruder γ={gamma:.2f}', fontsize=11, fontweight='bold')
        
        # Reduce number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(True, alpha=0.3)
        
        # Add legend for first subplot only
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.4, label='Pure Experiment'),
                Patch(facecolor='blue', alpha=0.4, label='Pure Simulation'),
                Patch(facecolor=color, alpha=0.7, label=f'Bruder γ={gamma:.2f}')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    plt.suptitle('Individual Bruder Method Comparisons: Each γ vs Pure Models', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_pdf, dpi=300, bbox_inches='tight')
    print(f"\n✅ Individual Bruder comparison subplots saved to {save_pdf}")
    plt.show()

def find_optimal_gamma_and_plot(sim_filename, exp_filename, sim_sizes, exp_sizes, delay,
                                 exp_data_val, exp_data_train,
                                 model_err_means=None,
                                 gamma_resolution=0.01,
                                 save_pdf='optimal_gamma_analysis.pdf'):
    """
    For each (sim_size, exp_size) combination, find the optimal gamma (0 to 1) that minimizes error.
    Plot:
    1. Top: 3D surface showing pure exp, pure sim, and optimal gamma error surface
    2. Bottom: Contour plot showing the distribution of optimal gamma values
    
    Args:
        sim_filename: simulation data filename
        exp_filename: experimental data filename
        sim_sizes: list of simulation data sizes
        exp_sizes: list of experimental data sizes
        delay: delay coordinate parameter
        exp_data_val: experimental validation data
        exp_data_train: experimental training data (for normalization)
        model_err_means: dict with keys 'sim_{n_sim}', 'exp_{n_exp}' for pure model errors
        gamma_resolution: step size for gamma search (default 0.01, so 0, 0.01, 0.02, ..., 1.0)
        save_pdf: output filename
    """
    print("\n" + "="*80)
    print("Finding Optimal Gamma for Each (Sim, Exp) Combination")
    print("="*80)
    
    # --- 1) Build experimental normalization from exp_data_train ---
    P_full_h_train = exp_data_train[:, 1:4]
    X_full_h_train = np.vstack([exp_data_train[:, 5], exp_data_train[:, 4]]).T
    mu_u_h = P_full_h_train.mean(axis=0, keepdims=True)
    sigma_u_h = P_full_h_train.std(axis=0, keepdims=True)
    mu_x_h = X_full_h_train.mean(axis=0, keepdims=True)
    sigma_x_h = X_full_h_train.std(axis=0, keepdims=True)
    
    # Handle zero standard deviations
    sigma_u_h = np.where(sigma_u_h < 1e-10, 1.0, sigma_u_h)
    sigma_x_h = np.where(sigma_x_h < 1e-10, 1.0, sigma_x_h)
    
    print(f"Experimental normalization computed from {exp_data_train.shape[0]:,} training samples")
    
    # --- 2) Preload sim and exp full raw arrays ---
    exp_full, sim_full = load_data(exp_filename, sim_filename)
    print(f"Loaded datasets: exp shape={exp_full.shape}, sim shape={sim_full.shape}")
    
    # --- 3) Build/calc K_sim and K_exp caches ---
    cache_K_sim = {}
    cache_K_exp = {}
    
    print(f"\nComputing K_sim matrices (in experimental-normalized coordinates)...")
    for n_sim in sim_sizes:
        if n_sim > sim_full.shape[0]:
            print(f"  Skipping sim_size={n_sim} (exceeds available data: {sim_full.shape[0]})")
            continue
            
        sim_data_raw = sim_full[:n_sim, :]
        
        # Extract features: simulation format is time,p1,p2,p3,x,y
        P_sim_raw = sim_data_raw[:, 1:4]
        X_sim_raw = np.vstack([sim_data_raw[:, 4], sim_data_raw[:, 5]]).T
        
        if P_sim_raw.shape[0] <= delay + 10:
            print(f"  Skipping sim_size={n_sim} (too few samples: {P_sim_raw.shape[0]} <= {delay + 10})")
            continue
        
        Xn_sim, Un_sim = normalize_with_exp_stats(X_sim_raw, P_sim_raw, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h)
        Xd_sim, Ud_sim = gen_delay_traj(Xn_sim, Un_sim, delay)
        K_sim = identify_K_full_with_inputs(Xd_sim, Ud_sim, mode=0)
        cache_K_sim[n_sim] = K_sim
        print(f"  Computed K_sim for n_sim={n_sim:,} (shape: {K_sim.shape})")
    
    print(f"\nComputing K_exp matrices (in experimental-normalized coordinates)...")
    for n_exp in exp_sizes:
        if n_exp > exp_data_train.shape[0]:
            print(f"  Skipping exp_size={n_exp} (exceeds available training data: {exp_data_train.shape[0]})")
            continue
            
        exp_data_raw = exp_data_train[:n_exp, :]
        P_exp_raw = exp_data_raw[:, 1:4]
        X_exp_raw = np.vstack([exp_data_raw[:, 5], exp_data_raw[:, 4]]).T
        
        if P_exp_raw.shape[0] <= delay + 10:
            print(f"  Skipping exp_size={n_exp} (too few samples: {P_exp_raw.shape[0]} <= {delay + 10})")
            continue
        
        Xn_exp, Un_exp = normalize_with_exp_stats(X_exp_raw, P_exp_raw, mu_x_h, sigma_x_h, mu_u_h, sigma_u_h)
        Xd_exp, Ud_exp = gen_delay_traj(Xn_exp, Un_exp, delay)
        K_exp = identify_K_full_with_inputs(Xd_exp, Ud_exp, mode=0)
        cache_K_exp[n_exp] = K_exp
        print(f"  Computed K_exp for n_exp={n_exp:,} (shape: {K_exp.shape})")
    
    # --- 4) Find optimal gamma for each combination ---
    # Generate gamma values to test: 0, gamma_resolution, 2*gamma_resolution, ..., 1.0
    gamma_values = np.arange(0.0, 1.0 + gamma_resolution, gamma_resolution)
    print(f"\nTesting {len(gamma_values)} gamma values from 0 to 1 (resolution={gamma_resolution})...")
    
    # Storage for optimal gamma and optimal error
    optimal_gamma = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
    optimal_error = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
    
    total_combinations = len(sim_sizes) * len(exp_sizes)
    current_combination = 0
    
    for i_sim, n_sim in enumerate(sim_sizes):
        if n_sim not in cache_K_sim:
            continue
        for j_exp, n_exp in enumerate(exp_sizes):
            if n_exp not in cache_K_exp:
                continue
            
            current_combination += 1
            if current_combination % max(1, total_combinations // 10) == 0:
                print(f"  Progress: {current_combination}/{total_combinations} combinations tested")
            
            K_sim = cache_K_sim[n_sim]
            K_exp = cache_K_exp[n_exp]
            
            # Align shapes if needed
            if K_sim.shape != K_exp.shape:
                maxdim = max(K_sim.shape[0], K_exp.shape[0])
                K_sim_pad = np.zeros((maxdim, maxdim))
                K_exp_pad = np.zeros((maxdim, maxdim))
                K_sim_pad[:K_sim.shape[0], :K_sim.shape[1]] = K_sim
                K_exp_pad[:K_exp.shape[0], :K_exp.shape[1]] = K_exp
                K_sim = K_sim_pad
                K_exp = K_exp_pad
            
            # Test all gamma values and find the one with minimum error
            best_gamma = np.nan
            best_error = np.inf
            
            for gamma in gamma_values:
                K_bruder = (1.0 - gamma) * K_sim + gamma * K_exp
                # Use max(exp_sizes) for slicing to match compute_model_errors behavior
                mean_err = evaluate_K_on_exp_validation(K_bruder, exp_data_val, delay,
                                                       mu_x_h, sigma_x_h, mu_u_h, sigma_u_h,
                                                       prediction_horizon=5,
                                                       max_exp_size=max(exp_sizes))
                
                if not np.isnan(mean_err) and mean_err < best_error:
                    best_error = mean_err
                    best_gamma = gamma
            
            if not np.isinf(best_error):
                optimal_gamma[i_sim, j_exp] = best_gamma
                optimal_error[i_sim, j_exp] = best_error
    
    print(f"\n✅ Optimal gamma found for {np.sum(~np.isnan(optimal_gamma))} combinations")
    print(f"   Gamma range: [{np.nanmin(optimal_gamma):.3f}, {np.nanmax(optimal_gamma):.3f}]")
    print(f"   Error range: [{np.nanmin(optimal_error):.6f}, {np.nanmax(optimal_error):.6f}]")
    
    # --- 5) Compute pure surfaces using the same evaluation method ---
    # This ensures consistency: gamma=0 should match pure sim, gamma=1 should match pure exp
    print(f"\nComputing pure simulation and experimental surfaces using same evaluation method...")
    
    Z_pure_sim = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
    Z_pure_exp = np.full((len(sim_sizes), len(exp_sizes)), np.nan)
    
    # Pure simulation: gamma = 0 (K_bruder = K_sim)
    for i_sim, n_sim in enumerate(sim_sizes):
        if n_sim not in cache_K_sim:
            continue
        
        K_sim = cache_K_sim[n_sim]
        # Evaluate pure simulation (gamma=0) for all exp_sizes
        # Note: pure sim error doesn't depend on exp_size, but we compute it for consistency
        for j_exp, n_exp in enumerate(exp_sizes):
            if n_exp not in cache_K_exp:
                continue
            
            K_exp = cache_K_exp[n_exp]
            
            # Align shapes if needed
            if K_sim.shape != K_exp.shape:
                maxdim = max(K_sim.shape[0], K_exp.shape[0])
                K_sim_pad = np.zeros((maxdim, maxdim))
                K_sim_pad[:K_sim.shape[0], :K_sim.shape[1]] = K_sim
                K_sim = K_sim_pad
            
            # Pure simulation: gamma = 0
            K_pure_sim = (1.0 - 0.0) * K_sim + 0.0 * K_exp  # = K_sim
            # Use max(exp_sizes) for slicing to match compute_model_errors behavior
            mean_err = evaluate_K_on_exp_validation(K_pure_sim, exp_data_val, delay,
                                                   mu_x_h, sigma_x_h, mu_u_h, sigma_u_h,
                                                   prediction_horizon=5,
                                                   max_exp_size=max(exp_sizes))
            if not np.isnan(mean_err):
                Z_pure_sim[i_sim, j_exp] = mean_err
    
    # Pure experimental: gamma = 1 (K_bruder = K_exp)
    for j_exp, n_exp in enumerate(exp_sizes):
        if n_exp not in cache_K_exp:
            continue
        
        K_exp = cache_K_exp[n_exp]
        # Evaluate pure experimental (gamma=1) for all sim_sizes
        for i_sim, n_sim in enumerate(sim_sizes):
            if n_sim not in cache_K_sim:
                continue
            
            K_sim = cache_K_sim[n_sim]
            
            # Align shapes if needed
            if K_sim.shape != K_exp.shape:
                maxdim = max(K_sim.shape[0], K_exp.shape[0])
                K_exp_pad = np.zeros((maxdim, maxdim))
                K_exp_pad[:K_exp.shape[0], :K_exp.shape[1]] = K_exp
                K_exp = K_exp_pad
            
            # Pure experimental: gamma = 1
            K_pure_exp = (1.0 - 1.0) * K_sim + 1.0 * K_exp  # = K_exp
            # Use max(exp_sizes) for slicing to match compute_model_errors behavior
            mean_err = evaluate_K_on_exp_validation(K_pure_exp, exp_data_val, delay,
                                                  mu_x_h, sigma_x_h, mu_u_h, sigma_u_h,
                                                  prediction_horizon=5,
                                                  max_exp_size=max(exp_sizes))
            if not np.isnan(mean_err):
                Z_pure_exp[i_sim, j_exp] = mean_err
    
    print(f"Pure simulation surface: {np.sum(~np.isnan(Z_pure_sim))} valid points")
    print(f"Pure experimental surface: {np.sum(~np.isnan(Z_pure_exp))} valid points")
    
    # Verify consistency: check that optimal_error matches pure surfaces at boundaries
    print(f"\nVerifying consistency at boundaries...")
    for i_sim, n_sim in enumerate(sim_sizes):
        for j_exp, n_exp in enumerate(exp_sizes):
            if np.isnan(optimal_gamma[i_sim, j_exp]):
                continue
            
            # When optimal gamma = 0, error should match pure sim
            if abs(optimal_gamma[i_sim, j_exp] - 0.0) < 1e-6:
                if not np.isnan(Z_pure_sim[i_sim, j_exp]):
                    diff = abs(optimal_error[i_sim, j_exp] - Z_pure_sim[i_sim, j_exp])
                    if diff > 1e-6:
                        print(f"  Warning: gamma=0 at (sim={n_sim}, exp={n_exp}): "
                              f"optimal_error={optimal_error[i_sim, j_exp]:.6f} != "
                              f"pure_sim={Z_pure_sim[i_sim, j_exp]:.6f} (diff={diff:.6f})")
            
            # When optimal gamma = 1, error should match pure exp
            if abs(optimal_gamma[i_sim, j_exp] - 1.0) < 1e-6:
                if not np.isnan(Z_pure_exp[i_sim, j_exp]):
                    diff = abs(optimal_error[i_sim, j_exp] - Z_pure_exp[i_sim, j_exp])
                    if diff > 1e-6:
                        print(f"  Warning: gamma=1 at (sim={n_sim}, exp={n_exp}): "
                              f"optimal_error={optimal_error[i_sim, j_exp]:.6f} != "
                              f"pure_exp={Z_pure_exp[i_sim, j_exp]:.6f} (diff={diff:.6f})")
    
    # --- 6) Prepare mesh in log-space for plotting ---
    # Filter to only show 2^8 to 2^18 (log2 values 8 to 18)
    exp_counts = np.array(exp_sizes)
    sim_counts = np.array(sim_sizes)
    exp_counts_log2 = np.log2(exp_counts)
    sim_counts_log2 = np.log2(sim_counts)
    
    # Filter indices for 2^8 to 2^18 range
    exp_mask = (exp_counts_log2 >= 8) & (exp_counts_log2 <= 18)
    sim_mask = (sim_counts_log2 >= 8) & (sim_counts_log2 <= 18)
    
    # Apply filters
    exp_counts_log2_filtered = exp_counts_log2[exp_mask]
    sim_counts_log2_filtered = sim_counts_log2[sim_mask]
    exp_counts_filtered = exp_counts[exp_mask]
    sim_counts_filtered = sim_counts[sim_mask]
    
    # Filter optimal_gamma and optimal_error matrices
    optimal_gamma_filtered = optimal_gamma[np.ix_(sim_mask, exp_mask)]
    optimal_error_filtered = optimal_error[np.ix_(sim_mask, exp_mask)]
    
    # Create meshgrid with filtered data
    P_EXP, P_SIM = np.meshgrid(exp_counts_log2_filtered, sim_counts_log2_filtered)
    
    # Helper function for safe log10
    def safe_log10(Z):
        Zsafe = np.array(Z, dtype=float)
        Zsafe[~np.isfinite(Zsafe)] = np.nan
        small = 1e-12
        Zsafe = np.where(Zsafe <= 0, small, Zsafe)
        return np.log10(Zsafe)
    
    # --- 7) Create figure with two subplots: 3D surface (top) and contour (bottom) ---
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    
    # Top subplot: 3D surface plot
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    
    # Filter pure surfaces to match the filtered range
    Z_pure_exp_filtered = None
    Z_pure_sim_filtered = None
    if Z_pure_exp is not None:
        Z_pure_exp_filtered = Z_pure_exp[np.ix_(sim_mask, exp_mask)]
    if Z_pure_sim is not None:
        Z_pure_sim_filtered = Z_pure_sim[np.ix_(sim_mask, exp_mask)]
    
    # Plot pure experiment surface (red, behind) - use filtered data
    if Z_pure_exp_filtered is not None:
        Z_exp_pure_log = safe_log10(Z_pure_exp_filtered)
        ax1.plot_surface(
            P_EXP, P_SIM, Z_exp_pure_log - 0.02,
            color='red', alpha=0.4, zorder=1, edgecolor='darkred', linewidth=0.5
        )
    
    # Plot pure simulation surface (blue, behind) - use filtered data
    if Z_pure_sim_filtered is not None:
        Z_sim_pure_log = safe_log10(Z_pure_sim_filtered)
        ax1.plot_surface(
            P_EXP, P_SIM, Z_sim_pure_log - 0.02,
            color='blue', alpha=0.4, zorder=1, edgecolor='darkblue', linewidth=0.5
        )
    
    # Plot optimal gamma error surface (green, on top) - use filtered data
    Z_optimal_log = safe_log10(optimal_error_filtered)
    ax1.plot_surface(
        P_EXP, P_SIM, Z_optimal_log,
        color='green', alpha=0.7, zorder=2, edgecolor='darkgreen', linewidth=0.5, antialiased=True
    )
    
    ax1.set_xlabel('Experimental Data (log2 scale)', fontsize=12)
    ax1.set_ylabel('Simulation Data (log2 scale)', fontsize=12)
    ax1.set_zlabel('Mean Error (log10 scale)', fontsize=12)
    ax1.set_title('Optimal Gamma Error Surface vs Pure Models', fontsize=14, fontweight='bold')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.grid(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    if Z_pure_exp is not None:
        legend_elements.append(Patch(facecolor='red', alpha=0.4, label='Pure Experiment'))
    if Z_pure_sim is not None:
        legend_elements.append(Patch(facecolor='blue', alpha=0.4, label='Pure Simulation'))
    legend_elements.append(Patch(facecolor='green', alpha=0.7, label='Optimal Gamma'))
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Bottom subplot: Discrete grid/heatmap of optimal gamma values (like a coefficient matrix)
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Create discrete heatmap using pcolormesh to show grid cells without interpolation
    # Create edges for pcolormesh (need one more point than data)
    # Handle non-uniform spacing by computing midpoints between adjacent points
    exp_edges = np.zeros(len(exp_counts_log2_filtered) + 1)
    exp_edges[0] = exp_counts_log2_filtered[0] - (exp_counts_log2_filtered[1] - exp_counts_log2_filtered[0]) / 2 if len(exp_counts_log2_filtered) > 1 else exp_counts_log2_filtered[0] - 0.5
    for i in range(1, len(exp_counts_log2_filtered)):
        exp_edges[i] = (exp_counts_log2_filtered[i-1] + exp_counts_log2_filtered[i]) / 2
    exp_edges[-1] = exp_counts_log2_filtered[-1] + (exp_counts_log2_filtered[-1] - exp_counts_log2_filtered[-2]) / 2 if len(exp_counts_log2_filtered) > 1 else exp_counts_log2_filtered[-1] + 0.5
    
    sim_edges = np.zeros(len(sim_counts_log2_filtered) + 1)
    sim_edges[0] = sim_counts_log2_filtered[0] - (sim_counts_log2_filtered[1] - sim_counts_log2_filtered[0]) / 2 if len(sim_counts_log2_filtered) > 1 else sim_counts_log2_filtered[0] - 0.5
    for i in range(1, len(sim_counts_log2_filtered)):
        sim_edges[i] = (sim_counts_log2_filtered[i-1] + sim_counts_log2_filtered[i]) / 2
    sim_edges[-1] = sim_counts_log2_filtered[-1] + (sim_counts_log2_filtered[-1] - sim_counts_log2_filtered[-2]) / 2 if len(sim_counts_log2_filtered) > 1 else sim_counts_log2_filtered[-1] + 0.5
    
    EXP_EDGES, SIM_EDGES = np.meshgrid(exp_edges, sim_edges)
    
    # Use pcolormesh for discrete grid visualization (no interpolation, no edges)
    im = ax2.pcolormesh(EXP_EDGES, SIM_EDGES, optimal_gamma_filtered, 
                        cmap='RdYlGn', vmin=0, vmax=1, 
                        shading='flat', edgecolors='none')
    
    # Add text annotations showing gamma values at each grid point
    for i_sim in range(len(sim_counts_log2_filtered)):
        for j_exp in range(len(exp_counts_log2_filtered)):
            if not np.isnan(optimal_gamma_filtered[i_sim, j_exp]):
                gamma_val = optimal_gamma_filtered[i_sim, j_exp]
                # Position text at the center of each grid cell
                x_pos = exp_counts_log2_filtered[j_exp]
                y_pos = sim_counts_log2_filtered[i_sim]
                # Choose text color based on background (white for dark, black for light)
                text_color = 'white' if gamma_val < 0.5 else 'black'
                ax2.text(x_pos, y_pos, f'{gamma_val:.2f}', 
                        ha='center', va='center', fontsize=9, 
                        color=text_color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='none', alpha=0.3))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Optimal Gamma Value', fontsize=12)
    
    # Set ticks to show actual grid points (centers of cells)
    ax2.set_xticks(exp_counts_log2_filtered)
    ax2.set_xticklabels([f'2^{int(x)}' for x in exp_counts_log2_filtered], fontsize=9, rotation=45, ha='right')
    ax2.set_yticks(sim_counts_log2_filtered)
    ax2.set_yticklabels([f'2^{int(y)}' for y in sim_counts_log2_filtered], fontsize=9)
    
    # Set axis limits to show all grid cells
    ax2.set_xlim(exp_edges[0], exp_edges[-1])
    ax2.set_ylim(sim_edges[0], sim_edges[-1])
    
    # Make subplot square
    ax2.set_aspect('equal', adjustable='box')
    
    ax2.set_xlabel('Experimental Data (log2 scale)', fontsize=12)
    ax2.set_ylabel('Simulation Data (log2 scale)', fontsize=12)
    ax2.set_title('Optimal Gamma Distribution (Discrete Grid - Only Known Values Shown)', fontsize=14, fontweight='bold')
    ax2.grid(False)  # Disable grid
    
    plt.tight_layout()
    fig.savefig(save_pdf, dpi=300, bbox_inches='tight')
    print(f"\n✅ Optimal gamma analysis saved to {save_pdf}")
    plt.show()
    
    return {
        'optimal_gamma': optimal_gamma,
        'optimal_error': optimal_error,
        'Z_pure_sim': Z_pure_sim,
        'Z_pure_exp': Z_pure_exp,
        'sim_sizes': sim_sizes,
        'exp_sizes': exp_sizes,
        'mu_x_h': mu_x_h,
        'sigma_x_h': sigma_x_h,
        'mu_u_h': mu_u_h,
        'sigma_u_h': sigma_u_h
    }

def plot_error_slices(optimal_results, save_pdf='error_slices.pdf'):
    """
    Plot error slices for specific exp and sim values.
    
    Creates 4 subplots:
    1. exp = 2^10 (varying sim) - pure exp, pure sim, optimal gamma error
    2. exp = 2^11 (varying sim) - pure exp, pure sim, optimal gamma error
    3. sim = 2^11 (varying exp) - pure exp, pure sim, optimal gamma error
    4. sim = 2^12 (varying exp) - pure exp, pure sim, optimal gamma error
    
    Args:
        optimal_results: dictionary returned from find_optimal_gamma_and_plot
        save_pdf: output filename
    """
    print("\n" + "="*80)
    print("Plotting Error Slices")
    print("="*80)
    
    # Extract data from results
    optimal_error = optimal_results['optimal_error']
    Z_pure_sim = optimal_results['Z_pure_sim']
    Z_pure_exp = optimal_results['Z_pure_exp']
    sim_sizes = np.array(optimal_results['sim_sizes'])
    exp_sizes = np.array(optimal_results['exp_sizes'])
    
    # Helper function for safe log10
    def safe_log10(Z):
        Zsafe = np.array(Z, dtype=float)
        Zsafe[~np.isfinite(Zsafe)] = np.nan
        small = 1e-12
        Zsafe = np.where(Zsafe <= 0, small, Zsafe)
        return np.log10(Zsafe)
    
    # Find indices for target values
    target_exp_10 = 2**10
    target_exp_11 = 2**11
    target_sim_11 = 2**11
    target_sim_12 = 2**12
    
    # Find indices (handle cases where exact values might not exist)
    idx_exp_10 = np.where(exp_sizes == target_exp_10)[0]
    idx_exp_11 = np.where(exp_sizes == target_exp_11)[0]
    idx_sim_11 = np.where(sim_sizes == target_sim_11)[0]
    idx_sim_12 = np.where(sim_sizes == target_sim_12)[0]
    
    if len(idx_exp_10) == 0:
        print(f"Warning: exp = 2^10 not found in exp_sizes. Available: {exp_sizes}")
        return
    if len(idx_exp_11) == 0:
        print(f"Warning: exp = 2^11 not found in exp_sizes. Available: {exp_sizes}")
        return
    if len(idx_sim_11) == 0:
        print(f"Warning: sim = 2^11 not found in sim_sizes. Available: {sim_sizes}")
        return
    if len(idx_sim_12) == 0:
        print(f"Warning: sim = 2^12 not found in sim_sizes. Available: {sim_sizes}")
        return
    
    idx_exp_10 = idx_exp_10[0]
    idx_exp_11 = idx_exp_11[0]
    idx_sim_11 = idx_sim_11[0]
    idx_sim_12 = idx_sim_12[0]
    
    print(f"Found indices: exp_10={idx_exp_10}, exp_11={idx_exp_11}, sim_11={idx_sim_11}, sim_12={idx_sim_12}")
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Slices: Pure Models vs Optimal Gamma (Bruder)', fontsize=16, fontweight='bold')
    
    # Subplot 1: exp = 2^10 (varying sim)
    ax1 = axes[0, 0]
    sim_log2 = np.log2(sim_sizes)
    
    # Extract slices (rows for varying sim, fixed exp column)
    pure_exp_slice_10 = Z_pure_exp[:, idx_exp_10]
    pure_sim_slice_10 = Z_pure_sim[:, idx_exp_10]
    optimal_slice_10 = optimal_error[:, idx_exp_10]
    
    # Filter out NaN values for plotting
    valid_mask_10 = ~(np.isnan(pure_exp_slice_10) | np.isnan(pure_sim_slice_10) | np.isnan(optimal_slice_10))
    
    ax1.plot(sim_log2[valid_mask_10], safe_log10(pure_exp_slice_10[valid_mask_10]), 
             'r-', linewidth=2, label='Pure Experiment', marker='o', markersize=5)
    ax1.plot(sim_log2[valid_mask_10], safe_log10(pure_sim_slice_10[valid_mask_10]), 
             'b-', linewidth=2, label='Pure Simulation', marker='s', markersize=5)
    ax1.plot(sim_log2[valid_mask_10], safe_log10(optimal_slice_10[valid_mask_10]), 
             'g-', linewidth=2, label='Optimal Gamma (Bruder)', marker='^', markersize=5)
    
    ax1.set_xlabel('Simulation Data (log2 scale)', fontsize=11)
    ax1.set_ylabel('log10(Error)', fontsize=11)
    ax1.set_title(f'Fixed Exp = 2^10 (varying Sim)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sim_log2[valid_mask_10][::max(1, len(sim_log2[valid_mask_10])//8)])
    ax1.set_xticklabels([f'2^{int(x)}' for x in sim_log2[valid_mask_10][::max(1, len(sim_log2[valid_mask_10])//8)]], 
                        rotation=45, ha='right', fontsize=8)
    
    # Subplot 2: exp = 2^11 (varying sim)
    ax2 = axes[0, 1]
    
    pure_exp_slice_11 = Z_pure_exp[:, idx_exp_11]
    pure_sim_slice_11 = Z_pure_sim[:, idx_exp_11]
    optimal_slice_11 = optimal_error[:, idx_exp_11]
    
    valid_mask_11 = ~(np.isnan(pure_exp_slice_11) | np.isnan(pure_sim_slice_11) | np.isnan(optimal_slice_11))
    
    ax2.plot(sim_log2[valid_mask_11], safe_log10(pure_exp_slice_11[valid_mask_11]), 
             'r-', linewidth=2, label='Pure Experiment', marker='o', markersize=5)
    ax2.plot(sim_log2[valid_mask_11], safe_log10(pure_sim_slice_11[valid_mask_11]), 
             'b-', linewidth=2, label='Pure Simulation', marker='s', markersize=5)
    ax2.plot(sim_log2[valid_mask_11], safe_log10(optimal_slice_11[valid_mask_11]), 
             'g-', linewidth=2, label='Optimal Gamma (Bruder)', marker='^', markersize=5)
    
    ax2.set_xlabel('Simulation Data (log2 scale)', fontsize=11)
    ax2.set_ylabel('log10(Error)', fontsize=11)
    ax2.set_title(f'Fixed Exp = 2^11 (varying Sim)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sim_log2[valid_mask_11][::max(1, len(sim_log2[valid_mask_11])//8)])
    ax2.set_xticklabels([f'2^{int(x)}' for x in sim_log2[valid_mask_11][::max(1, len(sim_log2[valid_mask_11])//8)]], 
                        rotation=45, ha='right', fontsize=8)
    
    # Subplot 3: sim = 2^11 (varying exp)
    ax3 = axes[1, 0]
    exp_log2 = np.log2(exp_sizes)
    
    # Extract slices (columns for varying exp, fixed sim row)
    pure_exp_slice_sim11 = Z_pure_exp[idx_sim_11, :]
    pure_sim_slice_sim11 = Z_pure_sim[idx_sim_11, :]
    optimal_slice_sim11 = optimal_error[idx_sim_11, :]
    
    valid_mask_sim11 = ~(np.isnan(pure_exp_slice_sim11) | np.isnan(pure_sim_slice_sim11) | np.isnan(optimal_slice_sim11))
    
    ax3.plot(exp_log2[valid_mask_sim11], safe_log10(pure_exp_slice_sim11[valid_mask_sim11]), 
             'r-', linewidth=2, label='Pure Experiment', marker='o', markersize=5)
    ax3.plot(exp_log2[valid_mask_sim11], safe_log10(pure_sim_slice_sim11[valid_mask_sim11]), 
             'b-', linewidth=2, label='Pure Simulation', marker='s', markersize=5)
    ax3.plot(exp_log2[valid_mask_sim11], safe_log10(optimal_slice_sim11[valid_mask_sim11]), 
             'g-', linewidth=2, label='Optimal Gamma (Bruder)', marker='^', markersize=5)
    
    ax3.set_xlabel('Experimental Data (log2 scale)', fontsize=11)
    ax3.set_ylabel('log10(Error)', fontsize=11)
    ax3.set_title(f'Fixed Sim = 2^11 (varying Exp)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(exp_log2[valid_mask_sim11][::max(1, len(exp_log2[valid_mask_sim11])//8)])
    ax3.set_xticklabels([f'2^{int(x)}' for x in exp_log2[valid_mask_sim11][::max(1, len(exp_log2[valid_mask_sim11])//8)]], 
                        rotation=45, ha='right', fontsize=8)
    
    # Subplot 4: sim = 2^12 (varying exp)
    ax4 = axes[1, 1]
    
    pure_exp_slice_sim12 = Z_pure_exp[idx_sim_12, :]
    pure_sim_slice_sim12 = Z_pure_sim[idx_sim_12, :]
    optimal_slice_sim12 = optimal_error[idx_sim_12, :]
    
    valid_mask_sim12 = ~(np.isnan(pure_exp_slice_sim12) | np.isnan(pure_sim_slice_sim12) | np.isnan(optimal_slice_sim12))
    
    ax4.plot(exp_log2[valid_mask_sim12], safe_log10(pure_exp_slice_sim12[valid_mask_sim12]), 
             'r-', linewidth=2, label='Pure Experiment', marker='o', markersize=5)
    ax4.plot(exp_log2[valid_mask_sim12], safe_log10(pure_sim_slice_sim12[valid_mask_sim12]), 
             'b-', linewidth=2, label='Pure Simulation', marker='s', markersize=5)
    ax4.plot(exp_log2[valid_mask_sim12], safe_log10(optimal_slice_sim12[valid_mask_sim12]), 
             'g-', linewidth=2, label='Optimal Gamma (Bruder)', marker='^', markersize=5)
    
    ax4.set_xlabel('Experimental Data (log2 scale)', fontsize=11)
    ax4.set_ylabel('log10(Error)', fontsize=11)
    ax4.set_title(f'Fixed Sim = 2^12 (varying Exp)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(exp_log2[valid_mask_sim12][::max(1, len(exp_log2[valid_mask_sim12])//8)])
    ax4.set_xticklabels([f'2^{int(x)}' for x in exp_log2[valid_mask_sim12][::max(1, len(exp_log2[valid_mask_sim12])//8)]], 
                        rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(save_pdf, dpi=300, bbox_inches='tight')
    print(f"\n✅ Error slices saved to {save_pdf}")
    plt.show()

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    DELAY = 20

    # === Select simulation and experimental data files ===
    # List of available simulation and experimental files
    simulation_files = [
        'Data/merged_simulation_results_grid8.csv',   # 0
        'Data/merged_simulation_results_grid27.csv',  # 1
        'Data/merged_simulation_results_init4.csv',   # 2
        'Data/merged_simulation_results_init7.csv',   # 3
        'Data/merged_simulation_results_init13.csv',  # 4
        'Data/merged_simulation_results_init19.csv',  # 5
    ]
    experimental_files = [
        'Data/raw_data_dynamic.csv',  # 0
        'Data/raw_data_static.csv',   # 1
        'Data/raw_data_0731.csv',   # 2
    ]

    # === Choose which files to use (edit these variables to select) ===
    sim_filename = simulation_files[4]  # e.g., 'merged_simulation_results_grid8.csv'
    exp_filename = experimental_files[2]  # e.g., 'raw_data_dynamic.csv'

    print(f"Using simulation file: {sim_filename}")
    print(f"Using experimental file: {exp_filename}")

    # Load data and prepare for Bruder calculation
    print("\nLoading data and preparing for Bruder's method calculation...")
    exp_data_full, sim_data_full = load_data(exp_filename, sim_filename)
    
    # Split experimental data into training and validation sets
    val_size = 20000
    train_size = 280000
    if exp_data_full.shape[0] >= val_size + train_size:
        exp_data_train = exp_data_full[:train_size, :]
        exp_data_val = exp_data_full[train_size:train_size+val_size, :]
    elif exp_data_full.shape[0] > val_size:
        exp_data_train = exp_data_full[:-val_size, :]
        exp_data_val = exp_data_full[-val_size:, :]
    else:
        exp_data_train = exp_data_full
        exp_data_val = exp_data_full
    
    # Generate exponential sizes for both datasets
    exp_sizes = generate_exponential_sizes(exp_data_train.shape[0])
    sim_sizes = generate_exponential_sizes(sim_data_full.shape[0])
    
    print(f"Experimental data sizes: {exp_sizes}")
    print(f"Simulation data sizes: {sim_sizes}")
    
    # Calculate Bruder's method results
    print("\nCalculating Bruder's method results...")
    bruder_results = compute_bruder_surfaces_and_plot_combined(
        sim_filename=sim_filename,
        exp_filename=exp_filename,
        sim_sizes=sim_sizes,
        exp_sizes=exp_sizes,
        delay=DELAY,
        exp_data_val=exp_data_val,
        exp_data_train=exp_data_train,
        gammas=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99],
        model_err_means=None,  # Not needed for Bruder calculation
        save_pdf='comparison_surfaces_bruder.pdf'
    )

    print("\n✅ Bruder's method calculation completed!") 