#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0
"""
error_evaluation.py

1) Load raw_data.csv and plot:
     a) P1,P2,P3 vs. time_relative
     b) (x2-x1) vs. (z2-z1)

2) Compute μ/σ, identify Koopman (A_full,B_full) on full set,
   save to koopman_model.pkl, plot eigenvalues.

3) Split 90/10 retrain on 90%, one-step errors pretrained vs retrained.

4) Visualize example trajectories (dx, dz) true vs predicted for retrained.

5) Compute and plot mean multi-step (1…20) prediction error on validation.

6) **New**: pick a random delay-coordinate start, simulate 1 000 steps
   using the pretrained (A_full,B_full) and the recorded u_norm sequence,
   then plot real vs predicted (dx,dz) long-horizon.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge, Lasso
import time
from collections import deque
from svgpathtools import svg2paths


def gen_delay_traj(X, U, delay):
    N, D = X.shape
    Z = np.zeros((N - delay, D * (delay + 1)))
    for i in range(delay + 1):
        Z[:, D*i:D*(i+1)] = X[delay-i : N-i, :]
    return Z, U[delay:, :]

def iden_koop(X_delay, U_delay, mode=0, alpha=1e-3):
    Y = np.hstack([X_delay[:-1], U_delay[:-1]])
    Z = np.hstack([X_delay[ 1:], U_delay[ 1:]])
    if mode == 0:
        K_dt = np.linalg.lstsq(Y, Z, rcond=None)[0].T
    elif mode == 1:
        clf = Lasso(alpha=alpha, fit_intercept=False)
        clf.fit(Y, Z)
        K_dt = clf.coef_
    else:
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(Y, Z)
        K_dt = clf.coef_

    D_state = X_delay.shape[1]
    A = K_dt[:D_state, :D_state]
    B = K_dt[:D_state, D_state:]
    return A, B

def test_one_step(X, U, A, B, delay):
    Z, U_del = gen_delay_traj(X, U, delay)
    D = X.shape[1]
    errors = []
    for k in range(Z.shape[0] - 1):
        z_pred = A @ Z[k] + B @ U_del[k]
        errors.append(np.linalg.norm(Z[k+1, :D] - z_pred[:D]))
    return np.array(errors)
def build_initial_z(dx0, dz0, delay):
    # repeat the first measurement across all delay slots
    x0 = np.array([[dx0],[dz0]])
    hist = deque([x0] * (delay+1), maxlen=delay+1)
    mats = list(hist)[::-1]
    Z = np.hstack(mats)
    return Z.reshape(-1,1)

def main():
    # 1) Load & plot raw data
    df = pd.read_csv("raw_data_0731.csv")
    # Read ALL paths and the SVG canvas size
    svg_file = r'/root/DataCollection/harp_oneseg_ws/src/data_collection/data_collection/RAL3.svg'

    paths, attributes = svg2paths(svg_file)
    path = paths[0]
    start_pt = np.array([0.005, 0.0])
    Scale = 0.1 / 400.0
    N = 9000

    points = [path.point(t) for t in np.linspace(0, 1, N)]
    x = np.array([Scale * p.real - .05 for p in points])
    y = np.array([Scale * p.imag - .05 for p in points])
    y = -y  # Optional: flip Y

    # Offset by current start point
    x = x + start_pt[0]
    y = y + start_pt[1]

    traj_svg = np.column_stack([x, y])

    # Flip the trajectory direction: reverse the SVG trajectory
    # traj_svg = traj_svg[::-1]  # Reverse the trajectory

    # Hold the starting point (now the original end point) for 1000 steps
    hold_start = np.repeat(traj_svg[0:1, :], 1000, axis=0)
    # Hold the ending point (now the original start point) for 1000 steps
    hold_end   = np.repeat(traj_svg[-1:, :], 1000, axis=0)

    # Final trajectory: [hold_start, flipped SVG trajectory, hold_end]
    # Now the robot will follow from the original end point back to the original start point
    trajectory = np.vstack([hold_start, traj_svg, hold_end])


    df = df[:100000]   ## change the training datasets
    ts = df["ts_rel"].values
    P  = df[["P1","P2","P3"]].values
    dx = df["dx"].values
    dz = df["dz"].values

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(ts, P[:,0], label="P1")
    ax[0].plot(ts, P[:,1], label="P2")
    ax[0].plot(ts, P[:,2], label="P3")
    ax[0].set(xlabel="time rel (s)", ylabel="Pressure", title="Inputs vs Time")
    ax[0].legend()
    ax[1].plot(dx, dz, "-o", markersize=3)
    ax[1].plot(trajectory[:,0], trajectory[:,1], "-", markersize=2)
    ax[1].plot(dx[0], dz[0], 'r*', markersize=10)
    ax[1].set(xlabel="dx", ylabel="dz", title="Output Trajectory")
    plt.tight_layout(); plt.show()

    # 2) Full‐data Koopman identification + save to .pkl
    
    delay = 20
    mu_u    = P.mean(axis=0, keepdims=True)
    sigma_u = P.std(axis=0,  keepdims=True)
    X_all   = np.vstack([dx, dz]).T
    mu_x    = X_all.mean(axis=0, keepdims=True)
    sigma_x = X_all.std(axis=0,  keepdims=True)

    U_norm = (P     - mu_u) / sigma_u
    X_norm = (X_all - mu_x) / sigma_x

    Xall_del, Uall_del = gen_delay_traj(X_norm, U_norm, delay)
    A_full, B_full     = iden_koop(Xall_del, Uall_del, mode=0)

    with open("koopman_model.pkl", "wb") as f:
        pickle.dump({
            "A":         A_full,
            "B":         B_full,
            "mu_u":      mu_u.tolist(),
            "sigma_u":   sigma_u.tolist(),
            "mu_x":      mu_x.tolist(),
            "sigma_x":   sigma_x.tolist(),
            "delay":     delay
        }, f)
    print("✅ Saved Koopman model to koopman_model.pkl")

    # 2b) eigenvalues
    eigs = np.linalg.eigvals(A_full)
    plt.figure(figsize=(6,6))
    plt.scatter(eigs.real, eigs.imag, marker='o')
    circle = plt.Circle((0,0),1,color='r',fill=False,linestyle='--')
    plt.gca().add_patch(circle)
    plt.xlabel("Real Part"); plt.ylabel("Imag Part")
    plt.title("Eigenvalues of A_full (unit circle)")
    plt.grid(True); plt.axis('equal'); plt.show()

    # 3) Split 90/10 & retrain on 90%
    N       = len(df)
    ntrain  = int(0.9 * N)
    Utr, Ute = U_norm[:ntrain], U_norm[ntrain:]
    Xtr, Xte = X_norm[:ntrain], X_norm[ntrain:]

    Xtr_del, Utr_del = gen_delay_traj(Xtr, Utr, delay)
    A_new, B_new     = iden_koop(Xtr_del, Utr_del, mode=0)

    # 4) One‐step errors on hold‐out
    err_full = test_one_step(Xte, Ute, A_full, B_full, delay)
    err_new  = test_one_step(Xte, Ute, A_new,  B_new,  delay)

    # plt.figure(figsize=(8,4))
    # plt.plot(err_full, 'r-',  label="Pretrained (all data)")
    # plt.plot(err_new,  'b--', label="Retrained (90%)")
    # plt.xlabel("Index (validation set)")
    # plt.ylabel("One‐step error")
    # plt.legend()
    # plt.title("Koopman One‐Step Prediction Errors")
    # plt.grid(True); plt.tight_layout(); plt.show()

    # print(f"Avg error (full data):   {err_full.mean():.5f}")
    # print(f"Avg error (90% retrain): {err_new.mean():.5f}")

    # # 5) Multi‐step (1…20) prediction error
    # D = X_norm.shape[1]
    # Zte, Ute_del = gen_delay_traj(Xte, Ute, delay)
    # max_hor = 10
    # multi_err = np.zeros(max_hor)
    # for h in range(1, max_hor+1):
    #     errs = []
    #     for k in range(Zte.shape[0] - h):
    #         pred = Zte[k].copy()
    #         for t in range(1, h+1):
    #             pred = A_new @ pred + B_new @ Ute_del[k+t-1]
    #         errs.append(np.linalg.norm(Zte[k+h,:D] - pred[:D]))
    #     multi_err[h-1] = np.mean(errs)

    # plt.figure(figsize=(6,4))
    # plt.plot(np.arange(1, max_hor+1), multi_/root/DataCollection/harp_oneseg_ws/history data/leaking/raw_data_0520.csverr, 'o-')
    # plt.xlabel("Prediction horizon (steps)")
    # plt.ylabel("Mean multi‐step error")
    # plt.title("Multi‐step Prediction Accuracy (Retrained)")
    # plt.grid(True); plt.tight_layout(); plt.show()

    # 6) Long-horizon (1 000‐step) simulation from a random start
    np.random.seed(42)
    M = X_norm.shape[0] - delay
    Tpred = 1000
    if M <= Tpred:
        print(f"Not enough data for a {Tpred}-step simulation")
        return

    start = np.random.randint(0, M - Tpred)
    Z_all_del, U_all_del = gen_delay_traj(X_norm, U_norm, delay)
    z = Z_all_del[start].reshape(-1,1)

    pred_big   = np.zeros((Tpred, 2))
    actual_big = X_norm[delay+start : delay+start+Tpred]

    for i in range(Tpred):
        u_n = U_all_del[start + i]
        z   = A_full.dot(z) + B_full.dot(u_n.reshape(-1,1))
        x_n = z[:2,0]
        pred_big[i] = x_n * sigma_x.flatten() + mu_x.flatten()

    actual_big = actual_big * sigma_x + mu_x

    plt.figure(figsize=(6,6))
    plt.plot(actual_big[:,0], actual_big[:,1], 'b-', alpha=0.5, label='Actual')
    plt.plot(pred_big[:,0],   pred_big[:,1],   'ro', alpha=0.8,markersize=1, label='Predicted')
    plt.xlabel("dx"); plt.ylabel("dz")
    plt.title(f"Long-horizon ({Tpred}-step) Prediction")
    plt.legend(); plt.axis('equal'); plt.tight_layout(); plt.show()

    # 6b) animate long‐horizon prediction step by step
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(6,6))
    # ax.set_title(f"Long‐horizon ({Tpred}‐step) Prediction (animated)")
    # ax.set_xlabel("dx")
    # ax.set_ylabel("dz")
    # ax.axis('equal')

    # # set fixed axes limits to cover both actual and predicted
    # xmin = min(actual_big[:,0].min(), pred_big[:,0].min()) - 0.01
    # xmax = max(actual_big[:,0].max(), pred_big[:,0].max()) + 0.01
    # ymin = min(actual_big[:,1].min(), pred_big[:,1].min()) - 0.01
    # ymax = max(actual_big[:,1].max(), pred_big[:,1].max()) + 0.01
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)

    # # create two Line2D objects we will update
    # actual_line, = ax.plot([], [], 'b-', alpha=0.5, label='Actual')
    # pred_line,   = ax.plot([], [], 'ro', alpha=0.8, markersize=1,  label='Predicted')
    # ax.legend(loc='upper right')

    # # buffers for the points drawn so far
    # act_xs, act_ys = [], []
    # pred_xs, pred_ys = [], []

    # for i in range(Tpred):
    #     # append next true and predicted point
    #     act_xs.append(actual_big[i,0])
    #     act_ys.append(actual_big[i,1])
    #     pred_xs.append(pred_big[i,0])
    #     pred_ys.append(pred_big[i,1])
        
    #     # update line data
    #     actual_line.set_data(act_xs, act_ys)
    #     pred_line.set_data(pred_xs, pred_ys)
        
    #     # redraw
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
        
    #     # wait a tiny bit so it looks like a video
    #     time.sleep(0.01)

    # plt.ioff()
    # plt.show()

    # # 7) Long‐horizon simulation on the recorded MPC run
    # log = pd.read_csv("mpc_log.csv")
    # act_mpc     = log[['dx','dz']].values      # true from the run
    # u_mpc_orig  = log[['u1','u2','u3']].values  # actual controls issued
    # steps_mpc   = min(1000, len(log))

    # # build initial lifted state from the very first dx,dz in MPC log
    # dx0, dz0 = act_mpc[0]
    # z = build_initial_z(dx0, dz0, delay)
    # # projection: pick first two lifted dims → [dx;dz]
    # n_lift = A_full.shape[1]
    # C = np.zeros((2, n_lift)); C[0,0]=1; C[1,1]=1

    # # preallocate predicted
    # pred_mpc = np.zeros((steps_mpc, 2))

    # # simulate
    # for i in range(steps_mpc):
    #     # normalize this step’s recorded u
    #     u_norm = (u_mpc_orig[i] - mu_u) / sigma_u
    #     # advance Koopman linear model
    #     z = A_full.dot(z) + B_full.dot(u_norm.reshape(-1,1))
    #     # project & denormalize
    #     x_n     = (C.dot(z)).flatten()
    #     pred_mpc[i] = x_n * sigma_x.flatten() + mu_x.flatten()

    # # animate the MPC‐log trajectory
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(6,6))
    # ax.set_title(f"MPC‐Log Long‐Horizon ({steps_mpc}-step) Prediction")
    # ax.set_xlabel("dx"); ax.set_ylabel("dz"); ax.axis('equal')
    # # matching axis limits
    # xmin = min(act_mpc[:steps_mpc,0].min(), pred_mpc[:,0].min()) - 0.01
    # xmax = max(act_mpc[:steps_mpc,0].max(), pred_mpc[:,0].max()) + 0.01
    # ymin = min(act_mpc[:steps_mpc,1].min(), pred_mpc[:,1].min()) - 0.01
    # ymax = max(act_mpc[:steps_mpc,1].max(), pred_mpc[:,1].max()) + 0.01
    # ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # actual_line, = ax.plot([], [], 'b-', alpha=0.5, label='MPC-Run Actual')
    # pred_line,   = ax.plot([], [], 'ro', alpha=0.8, label='Predicted')
    # ax.legend(loc='upper right')

    # act_xs, act_ys   = [], []
    # pred_xs, pred_ys = [], []

    # for i in range(steps_mpc):
    #     act_xs.append(act_mpc[i,0]);   act_ys.append(act_mpc[i,1])
    #     pred_xs.append(pred_mpc[i,0]); pred_ys.append(pred_mpc[i,1])
    #     actual_line.set_data(act_xs, act_ys)
    #     pred_line.set_data(pred_xs, pred_ys)
    #     fig.canvas.draw(); fig.canvas.flush_events()
    #     time.sleep(0.01)

    # plt.ioff()
    # plt.show()
    # ## ←— END OF NEW BLOCK —→

    # 7) Plot distribution of ramp_time from pressure_params.csv
    try:
        params = pd.read_csv("/root/DataCollection/harp_oneseg_ws/pressure_params.csv")
        if 'ramp_time' in params.columns:
            plt.figure(figsize=(6,4))
            plt.hist(params['ramp_time'], bins=20, edgecolor='black', alpha=0.7)
            plt.xlabel("Ramp Time (s)")
            plt.ylabel("Count")
            plt.title("Distribution of Ramp Times Used During Data Collection")
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️  'ramp_time' column not found in pressure_params.csv")
    except FileNotFoundError:
        print("⚠️  pressure_params.csv not found; skipping ramp_time distribution plot")


if __name__ == "__main__":
    main()
