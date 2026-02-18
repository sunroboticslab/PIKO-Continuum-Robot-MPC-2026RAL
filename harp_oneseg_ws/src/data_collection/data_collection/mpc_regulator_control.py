#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0
"""
mpc_regulator_control.py

1) Wait for first OptiTrack reading, record (dx0,dz0).
2) Build a "guideline" from (dx0,dz0) to circle start, then the circle.
3) Runs Koopman‐MPC at 10 Hz (QRQP) to track that trajectory, with Δu cost.
4) Logs & prints control signals.
"""

import time, math, csv, pickle
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial
import numpy as np
import casadi as cs
from svgpathtools import svg2paths
from svgpathtools import svg2paths2

# --- KMPC class with Δu cost ---
class KMPC:
    def __init__(self, A, B, C, Q, R, look_ahead, u_max=65, u_min=0, solver='qrqp'):
        self.A, self.B, self.C = A, B, C
        self.Q, self.R         = Q, R
        self.T                = look_ahead
        self.u_max, self.u_min = u_max, u_min
        self.n, self.m, self.N = A.shape[1], B.shape[1], C.shape[0]

        opti = cs.Opti('conic') if solver in ['qrqp','qpoases','osqp','proxqp'] else cs.Opti()

        # decision vars
        z = opti.variable(self.n, self.T+1)
        u = opti.variable(self.m, self.T)

        # parameters
        z0     = opti.parameter(self.n, 1)
        xref   = opti.parameter(self.N, self.T+1)
        u_prev = opti.parameter(self.m, 1)

        # build cost
        cost = 0
        xpred = C @ z
        # state‐tracking cost
        for k in range(1, self.T+1):
            e = xpred[:,k] - xref[:,k]
            cost += e.T @ Q @ e
        # Δu‐cost
        du0 = u[:,0] - u_prev
        cost += du0.T @ R @ du0
        for k in range(self.T-1):
            duk = u[:,k+1] - u[:,k]
            cost += duk.T @ R @ duk

        # dynamics & bounds
        opti.subject_to(z[:,0] == z0)
        for k in range(self.T):
            opti.subject_to(z[:,k+1] == A @ z[:,k] + B @ u[:,k])
            opti.subject_to(opti.bounded(self.u_min, u[:,k], self.u_max))

        opti.minimize(cost)
        opti.solver('qrqp', {'print_header':False,'print_info':False,'print_iter':False})

        self._opti = opti
        self.vars  = dict(z=z, u=u, z0=z0, xref=xref, u_prev=u_prev)

    def solve(self, ref, z_init, u_prev_val, warm=None):
        opti = self._opti
        v    = self.vars
        if warm is not None:
            opti.set_initial(warm.value_variables())
        opti.set_value(v['z0'],     z_init)
        opti.set_value(v['xref'],   ref)
        opti.set_value(v['u_prev'], u_prev_val)

        try:
            sol = opti.solve()
            return sol.value(v['u'])[:,0], sol
        except RuntimeError as e:
            print("MPC solve failed:", e)
            return np.zeros(self.m), None


class RegulatorControl(Node):
    def __init__(self):
        super().__init__('mpc_regulator_control')

        # will fill in on first opti‐track callback
        self.trajectory  = None
        self.step        = 0
        self.u_prev      = np.zeros((3,1))
        self.log         = []

        # load Koopman + stats
        with open('koopman_model.pkl','rb') as f:
            data = pickle.load(f)
        self.mu_u, self.sigma_u = np.array(data['mu_u']), np.array(data['sigma_u'])
        self.mu_x, self.sigma_x = np.array(data['mu_x']), np.array(data['sigma_x'])
        self.delay = int(data['delay'])
        A = np.array(data['A']); B = np.array(data['B'])
        n_lift = A.shape[1]
        C = np.zeros((2,n_lift)); C[0,0]=1; C[1,1]=1
        # # ───────────────────────────────────────────────────────────────
        # # STEP A) Load the "big" pickle, which contains all 35 Koopman models
        # # ───────────────────────────────────────────────────────────────
        # with open('koopman_models.pkl', 'rb') as f:
        #     all_models = pickle.load(f)

        # # ────────────────────────────────────────────────────────────────────
        # # STEP B) Choose exactly which Koopman variant you want to use.
        # #          Valid keys follow the pattern "kfXXhYY", where XX ∈ {00,20,40,60,80,100}
        # #          is the percentage of simulation data (Kf) and YY ∈ {00,20,40,60,80,100}
        # #          is the percentage of experiment data (Kh). For example:
        # #            'kf00h20' →  0% sim, 20% exp (Kh_only on 20% of exp)
        # #            'kf20h00' → 20% sim,  0% exp (Kf_only on 20% of sim)
        # #            'kf20h40' → 20% sim, 40% exp (combined)
        # #            'kf100h100' → 100% sim, 100% exp (combined on full datasets)
        # #
        # #          Here's the full list of 35 possible keys:
        # #            • Kf_only  : 'kf20h00', 'kf40h00', 'kf60h00', 'kf80h00', 'kf100h00'
        # #            • Kh_only  : 'kf00h20', 'kf00h40', 'kf00h60', 'kf00h80', 'kf00h100'
        # #            • Combined : 'kf20h20', 'kf20h40', 'kf20h60', 'kf20h80', 'kf20h100',
        # #                         'kf40h20', 'kf40h40', 'kf40h60', 'kf40h80', 'kf40h100',
        # #                         'kf60h20', 'kf60h40', 'kf60h# STEP B) Choose exactly which key you want to use.
        # #          For example: "kf20h40" means 20% of sim data + 40% of exp data.
        # # → Edit the line below to pick your desired variant:
        # chosen_key = 'kf20h40'
        # # ────────────────────────────────────────────────────────────────────

        # if chosen_key not in all_models:
        #     raise KeyError(f"Could not find key '{chosen_key}' in koopman_models.pkl")

        # m = all_models[chosen_key]

        # # ───────────────────────────────────────────────────────────────
        # # STEP C) Depending on m['type'], extract the correct A, B, mu, sigma, delay
        # # ───────────────────────────────────────────────────────────────
        # # All three possibilities:
        # #   • Kf_only  → use (A_f, B_f)  plus (mu_u_f, sigma_u_f), (mu_x_f, sigma_x_f)
        # #   • Kh_only  → use (A_h, B_h)  plus (mu_u_h, sigma_u_h), (mu_x_h, sigma_x_h)
        # #   • combined → use (A_full, B_full) plus (mu_u_h, sigma_u_h), (mu_x_h, sigma_x_h)
        # #
        # model_type = m['type']

        # if model_type == 'Kf_only':
        #     # Pure‐simulation Koopman (predicts [y,x] from sim data)
        #     A_data = np.array(m['A_f'])
        #     B_data = np.array(m['B_f'])
        #     mu_u   = np.array(m['mu_u_f']).reshape(1, 3)
        #     sigma_u= np.array(m['sigma_u_f']).reshape(1, 3)
        #     mu_x   = np.array(m['mu_x_f']).reshape(1, 2)
        #     sigma_x= np.array(m['sigma_x_f']).reshape(1, 2)

        # elif model_type == 'Kh_only':
        #     # Pure‐experiment Koopman (predicts [dx,dz] from exp data)
        #     A_data = np.array(m['A_h'])
        #     B_data = np.array(m['B_h'])
        #     mu_u   = np.array(m['mu_u_h']).reshape(1, 3)
        #     sigma_u= np.array(m['sigma_u_h']).reshape(1, 3)
        #     mu_x   = np.array(m['mu_x_h']).reshape(1, 2)
        #     sigma_x= np.array(m['sigma_x_h']).reshape(1, 2)

        # else:  # model_type == 'combined'
        #     # Combined Koopman (Strang‐split) (predicts [dx,dz] from exp data)
        #     A_data = np.array(m['A_full'])
        #     B_data = np.array(m['B_full'])
        #     mu_u   = np.array(m['mu_u_h']).reshape(1, 3)
        #     sigma_u= np.array(m['sigma_u_h']).reshape(1, 3)
        #     mu_x   = np.array(m['mu_x_h']).reshape(1, 2)
        #     sigma_x= np.array(m['sigma_x_h']).reshape(1, 2)

        # delay_val = int(m['delay'])

        # # ───────────────────────────────────────────────────────────────
        # # STEP D) Store them exactly as before in self.mu_u, self.sigma_u, etc.
        # # ───────────────────────────────────────────────────────────────
        # self.mu_u,    self.sigma_u = mu_u,    sigma_u
        # self.mu_x,    self.sigma_x = mu_x,    sigma_x
        # self.delay                  = delay_val
        # A = A_data.copy()
        # B = B_data.copy()
        # n_lift = A.shape[1]

        # # Build C so that C·z = [dx; dz] or [y; x], depending on model_type.
        # # In both "Kh_only" and "combined" we want to extract the first two lifted states (dx, dz).
        # # In the "Kf_only" scenario, the first two lifted coordinates correspond to [y, x] by construction.
        # C = np.zeros((2, n_lift))
        # C[0,0] = 1
        # C[1,1] = 1



        # rolling buffer
        self.x_hist = None  # will initialize after first OptiTrack reading

        # normalized bounds
        u_min = ((0.0 - self.mu_u)/self.sigma_u).reshape(-1,1)
        u_max = ((65.0 - self.mu_u)/self.sigma_u).reshape(-1,1)

        # MPC w/ Δu cost
        Q, R, T = np.eye(2)*10, np.eye(3)*1.5, 2  ## at least 2
        self.mpc = KMPC(A, B, C, Q, R, T,
                        u_max=u_max, u_min=u_min,
                        solver='qrqp')

        # Arduino
        self.ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)
        time.sleep(2)
        self.get_logger().info("✅ Arduino connected")

        # OptiTrack subscriber
        self.create_subscription(
            Float32MultiArray,
            '/optitrack/rigid_body_array',
            self.opt_callback,
            10
        )

    def opt_callback(self, msg: Float32MultiArray):
        o = msg.data
        dx0, dz0 = (o[9] - o[1]), (o[11] - o[3])

        if self.trajectory is None:
            start_pt = np.array([dx0, dz0])
            
            ##### --- ciricle and guide line --- #####
            # # 1) start point
            # mid_pt = np.array([dx0, dz0])  # [r+dx0, dz0]

            # # 2) small "guide" to entry of circle
            # r, ncount = 0.015, 1000
            # guide1 = np.linspace(start_pt, mid_pt, 500, endpoint=False)
            # guide2 = np.linspace(mid_pt, [r+mid_pt[0], mid_pt[1]], 500, endpoint=False)

            # # 3) one circle (offset by start)
            # angles = np.linspace(0, 2*math.pi, ncount, endpoint=False)
            # circle = np.stack([r*np.cos(angles)+mid_pt[0], r*np.sin(angles)+ mid_pt[1]], axis=1)

            # # 4) repeat circle 10×
            # circle_10x = np.vstack([circle]*3)

            # # 5) full trajectory & start loop
            # self.trajectory  = np.vstack([guide1,guide2, circle_10x])

            ##### --- ASU word --- #####

            # # Load SVG and extract paths
            # paths, attributes = svg2paths(r'/root/DataCollection/harp_oneseg_ws/src/data_collection/data_collection/ASUlogo.svg')
            # path = paths[0]
            # start_pt = np.array([0.015, -0.010])
            # Scale = 0.1 / 350.0
            # N = 9000

            # points = [path.point(t) for t in np.linspace(0, 1, N)]
            # x = np.array([Scale * p.real - .05 for p in points])
            # y = np.array([Scale * p.imag - .05 for p in points])
            # y = -y  # Optional: flip Y

            # # Offset by current start point
            # x = x + start_pt[0]
            # y = y + start_pt[1]

            # traj_svg = np.column_stack([x, y])

            # # Flip the trajectory direction: reverse the SVG trajectory
            # traj_svg = traj_svg[::-1]  # Reverse the trajectory

            # # Hold the starting point (now the original end point) for 1000 steps
            # hold_start = np.repeat(traj_svg[0:1, :], 1000, axis=0)
            # # Hold the ending point (now the original start point) for 1000 steps
            # hold_end   = np.repeat(traj_svg[-1:, :], 1000, axis=0)

            # # Final trajectory: [hold_start, flipped SVG trajectory, hold_end]
            # # Now the robot will follow from the original end point back to the original start point
            # self.trajectory = np.vstack([hold_start, traj_svg, hold_end])


            ##### --- RAL word --- #####
            # Read ALL paths and the SVG canvas size
            svg_file = r'/root/DataCollection/harp_oneseg_ws/src/data_collection/data_collection/RAL3.svg'

            paths, attributes = svg2paths(svg_file)
            path = paths[0]
            start_pt = np.array([0.005, 0.0])
            Scale = 0.1 / 400.0
            N = 6000

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

            # Final trajectory: [hold_start, flipped SVG trajectory, hold_end]
            # Now the robot will follow from the original end point back to the original start point
            self.trajectory = np.vstack([hold_start, traj_svg, hold_end])



            self.total_steps = len(self.trajectory)
            self.step = 0
            self.create_timer(0.01, self.control_loop)

        # always update
        self.last_opt = msg.data

    def control_loop(self):
        if self.step >= self.total_steps:
            self.ser.write(b"A,0.00,0.00,0.00,Z\n")
            self._save_log()
            rclpy.shutdown()
            return

        # build normalized reference window
        idx = self.step % len(self.trajectory)
        ref = np.zeros((2, self.mpc.T+1))
        for k in range(self.mpc.T+1):
            p = self.trajectory[(idx+k) % len(self.trajectory)]
            ref[:,k] = (p - self.mu_x)/self.sigma_x

        # read & normalize measurement
        o = self.last_opt
        dx, dz = (o[9]-o[1]), (o[11]-o[3])
        x_n = ((np.array([dx,dz]) - self.mu_x)/self.sigma_x).reshape(-1,1)

        # initialize x_hist on first real measurement
        if self.x_hist is None:
            self.x_hist = deque([x_n.copy()] * (self.delay+1), maxlen=self.delay+1)
        else:
            self.x_hist.append(x_n)

        # update delay‐buffer
        hist = list(self.x_hist)[::-1]  # newest→oldest
        z0 = np.vstack([x.flatten() for x in hist]).reshape(-1,1)

        # solve MPC with Δu cost
        u_norm, _ = self.mpc.solve(ref, z0, self.u_prev)
        u_orig    = (u_norm*self.sigma_u + self.mu_u).flatten()

        # save for next Δu
        self.u_prev = u_norm.reshape(-1,1)

        # send to Arduino
        cmd = f"A,{u_orig[0]:.2f},{u_orig[1]:.2f},{u_orig[2]:.2f},Z\n"
        self.ser.write(cmd.encode())
        self.get_logger().info(f"➡️ {cmd.strip()}")

        # log & advance
        self.log.append([
            self.step,
            *self.trajectory[idx],
            dx, dz,
            float(u_orig[0]), float(u_orig[1]), float(u_orig[2])
        ])
        self.step += 1

    def _save_log(self):
        with open('mpc_log.csv','w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['step','ref_x','ref_y','dx','dz','u1','u2','u3'])
            w.writerows(self.log)
        self.get_logger().info("💾 Saved mpc_log.csv")


def main(args=None):
    rclpy.init(args=args)
    node = RegulatorControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    main()