#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray

import numpy as np
import pickle
import csv
from sklearn.linear_model import Ridge, Lasso

def gen_delay_traj(X, U, delay):
    """
    X: [N, D] state sequence
    U: [N, p] input sequence
    delay: number of delays

    Returns:
      Z: [(N–delay) x (D*(delay+1))] stacked delayed X
      U_del: [(N–delay) x p] aligned inputs
    """
    N, D = X.shape
    Z = np.zeros((N - delay, D * (delay + 1)))
    for i in range(delay + 1):
        Z[:, D*i:D*(i+1)] = X[delay - i : N - i, :]
    return Z, U[delay:, :]

def iden_koop(X_delay, U_delay, mode=0, alpha=1e-3, tol=1e-4):
    """
    Identify Koopman A,B from delayed data.
      X_delay: [M x (D*(delay+1))]
      U_delay: [M x p]
    mode: 0=least‐squares, 1=Lasso, 2=Ridge
    """
    Y = np.hstack([X_delay[:-1], U_delay[:-1]])
    Z = np.hstack([X_delay[ 1:], U_delay[ 1:]])
    if mode == 0:
        K = np.linalg.lstsq(Y, Z, rcond=None)[0].T
    elif mode == 1:
        clf = Lasso(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_
    else:
        clf = Ridge(alpha=alpha, fit_intercept=False, tol=tol)
        clf.fit(Y, Z)
        K = clf.coef_

    D_state = X_delay.shape[1]
    A = K[:D_state, :D_state]
    B = K[:D_state, D_state:]
    return A, B

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.first_ts      = None
        self.latest_opt    = None
        self._all          = []
        self.max_rows      = 300_000

        self.create_subscription(
            Float64MultiArray,
            'regulator_input',
            self.reg_callback, 10
        )
        self.create_subscription(
            Float32MultiArray,
            '/optitrack/rigid_body_array',
            self.opt_callback, 10
        )
        self.get_logger().info("🔍 DataCollector ready…")

    def opt_callback(self, msg: Float32MultiArray):
        self.latest_opt = msg.data

    def reg_callback(self, msg: Float64MultiArray):
        ts, P1, P2, P3 = msg.data
        if self.first_ts is None:
            self.first_ts = ts

        if self.latest_opt is None:
            self.get_logger().warning("No OptiTrack data yet; skipping this sample.")
            return

        o  = self.latest_opt
        x1, z1 = float(o[1]), float(o[3])
        x2, z2 = float(o[9]), float(o[11])
        self._all.append([ts, P1, P2, P3, x1, z1, x2, z2])

        # log percentage collected so far
        pct = len(self._all) / self.max_rows * 100
        self.get_logger().info(f"{pct:.1f}% of samples collected")

        if len(self._all) >= self.max_rows:
            self.get_logger().info(f"Reached {self.max_rows} samples → processing")
            self.process_and_train()
            rclpy.shutdown()

    def process_and_train(self):
        data = np.array(self._all)
        ts     = data[:,0]
        P      = data[:,1:4]
        x1, z1 = data[:,4], data[:,5]
        x2, z2 = data[:,6], data[:,7]

        # 1) Save raw_data.csv
        ts_rel = ts - self.first_ts
        dx, dz = x2 - x1, z2 - z1
        raw    = np.column_stack([ts_rel, P, dx, dz])
        with open('raw_data.csv','w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['ts_rel','P1','P2','P3','dx','dz'])
            w.writerows(raw)
        self.get_logger().info("➤ raw_data.csv saved")

        # 2) Compute μ/σ and normalize
        U        = P
        X        = np.column_stack([dx, dz])
        mu_u     = U.mean(axis=0);    sigma_u = U.std(axis=0)
        mu_x     = X.mean(axis=0);    sigma_x = X.std(axis=0)
        U_n      = (U - mu_u) / sigma_u
        X_n      = (X - mu_x) / sigma_x

        # 3) Build delay coords & identify Koopman on full set
        delay    = 5
        Xd, Ud   = gen_delay_traj(X_n, U_n, delay)
        A_full, B_full = iden_koop(Xd, Ud, mode=0)

        # 4) Save model + stats
        out = {
            'A':       A_full.tolist(),
            'B':       B_full.tolist(),
            'mu_u':    mu_u.tolist(),
            'sigma_u': sigma_u.tolist(),
            'mu_x':    mu_x.tolist(),
            'sigma_x': sigma_x.tolist(),
            'delay':   delay
        }
        with open('koopman_model.pkl','wb') as f:
            pickle.dump(out, f)
        self.get_logger().info("➤ koopman_model.pkl saved")

    def plot_data(self):
        # no-op for now
        pass

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    finally:
        if node._all:
            node.get_logger().info("Early shutdown → saving collected data")
            node.process_and_train()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
