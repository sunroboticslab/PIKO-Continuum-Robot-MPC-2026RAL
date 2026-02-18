import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge, Lasso
from collections import deque
import time

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
    x0 = np.array([[dx0],[dz0]])
    hist = deque([x0] * (delay+1), maxlen=delay+1)
    mats = list(hist)[::-1]
    Z = np.hstack(mats)
    return Z.reshape(-1,1)

def main():
    # 1) Load & normalize data
    df = pd.read_csv("raw_data_dynamic.csv")
    ts = df["ts_rel"].values
    P  = df[["P1","P2","P3"]].values
    dx = df["dx"].values
    dz = df["dz"].values
    delay = 20
    X_all   = np.vstack([dx, dz]).T
    mu_u    = P.mean(axis=0, keepdims=True)
    sigma_u = P.std(axis=0,  keepdims=True)
    mu_x    = X_all.mean(axis=0, keepdims=True)
    sigma_x = X_all.std(axis=0,  keepdims=True)
    U_norm = (P     - mu_u) / sigma_u
    X_norm = (X_all - mu_x) / sigma_x

    # Save means/stdev for use later
    stats = dict(mu_u=mu_u, sigma_u=sigma_u, mu_x=mu_x, sigma_x=sigma_x, delay=delay)

    # 2) Train and save Koopman models at different fractions
    fracs = [ 0.1, 0.2, 0.6, 0.8, 0.9, 1 ]  # 2, 5, 10, 20, 40, 60, 80, 90, 100  0.01, 0.02, 0.05, 0.075, 0.1, 1
    models = []
    errors = []

    plt.figure(figsize=(14,8))
    for i, frac in enumerate(fracs):
        N = int(frac * len(df))
        # Training set
        Xtr, Utr = X_norm[:N], U_norm[:N]
        Xtr_del, Utr_del = gen_delay_traj(Xtr, Utr, delay)
        A, B = iden_koop(Xtr_del, Utr_del, mode=0)
        model = {"A": A, "B": B, **stats}
        models.append(model)
        # Save model
        with open(f"Koopman_{int(frac*100)}pct_model.pkl", "wb") as f:
            pickle.dump(model, f)
        # Eigenvalue plot
        eigs = np.linalg.eigvals(A)
        plt.subplot(2,3,i+1)
        plt.scatter(eigs.real, eigs.imag, marker='o')
        circle = plt.Circle((0,0),1,color='r',fill=False,linestyle='--')
        plt.gca().add_patch(circle)
        plt.xlabel("Re"); plt.ylabel("Im")
        plt.title(f"Eig: {int(frac*100)}% data")
        plt.grid(True); plt.axis('equal')
    plt.suptitle("Koopman A Eigenvalues vs. Dataset Fraction")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3) Prediction error for each fraction (20-90%), evaluate on remaining data
    fracs_pred = fracs[:-1]  # exclude 100%
    error_means = []
    for i, frac in enumerate(fracs_pred):
        N = int(frac * len(df))
        Xtr, Utr = X_norm[:N], U_norm[:N]
        Xte, Ute = X_norm[N:], U_norm[N:]
        Xte_del, Ute_del = gen_delay_traj(Xte, Ute, delay)
        A, B = models[i]["A"], models[i]["B"]
        if Xte_del.shape[0] > 1:
            err = test_one_step(Xte, Ute, A, B, delay)
            error_means.append(np.mean(err))
        else:
            error_means.append(np.nan)  # Not enough validation data

    plt.figure(figsize=(6,4))
    plt.plot([int(f*100) for f in fracs_pred], error_means, 'o-', lw=2)
    plt.xlabel('Training data used (%)')
    plt.ylabel('Validation error (mean one-step)')
    plt.title('Prediction Error vs. Training Fraction')
    plt.grid(True)
    plt.show()

    # 4) Long-horizon predictions (1,000 steps) for each model, overlaid
    np.random.seed(43)
    M = X_norm.shape[0] - delay
    Tpred = 100
    if M <= Tpred:
        print(f"Not enough data for a {Tpred}-step simulation")
        return
    start = np.random.randint(0, M - Tpred)
    Z_all_del, U_all_del = gen_delay_traj(X_norm, U_norm, delay)
    actual_big = X_norm[delay+start : delay+start+Tpred] * sigma_x + mu_x

    fig, axes = plt.subplots(3,2,figsize=(14,10),sharex=True,sharey=True)
    axes = axes.flatten()
    for i, frac in enumerate(fracs):
        model = models[i]
        A, B = model["A"], model["B"]
        z = Z_all_del[start].reshape(-1,1)
        pred_big = np.zeros((Tpred, 2))
        for j in range(Tpred):
            u_n = U_all_del[start + j]
            z   = A.dot(z) + B.dot(u_n.reshape(-1,1))
            x_n = z[:2,0]
            pred_big[j] = x_n * sigma_x.flatten() + mu_x.flatten()
        ax = axes[i]
        ax.plot(actual_big[:,0], actual_big[:,1], 'b-', alpha=0.5, label='Actual')
        ax.plot(pred_big[:,0],   pred_big[:,1],   'ro', alpha=0.8,markersize=1, label='Predicted')
        ax.set_title(f"{int(frac*100)}% data")
        ax.set_xlabel("dx"); ax.set_ylabel("dz")
        # ax.set_aspect('equal', adjustable='box')
        if i == 0:
            ax.legend()
    plt.suptitle("1000-step Prediction Trajectories (different training fractions)")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
