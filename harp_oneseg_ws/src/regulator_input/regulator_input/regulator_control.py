#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0
"""
regulator_control.py

1) Streams smooth randomized ramp-and-hold pressure profiles to the robot.
2) Publishes [timestamp, P1, P2, P3] on 'regulator_input'.
3) Sends identical commands to the Arduino.
4) Randomizes ramp_time = 1 + U(0,5) *for each cycle* and sets hold_time = 5.
5) On shutdown, saves the full list of ramp_times, the hold_time,
   and the entire pressure sequence.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import serial
import time
import numpy as np
import csv
import atexit

class RegulatorControl(Node):
    def __init__(self):
        super().__init__('regulator_control')

        # --- Serial to Arduino ---
        self.ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=1)
        time.sleep(2)
        self.get_logger().info("Connected to Arduino on /dev/ttyACM0")

        # --- ROS Publisher ---
        self.pub = self.create_publisher(Float64MultiArray, 'regulator_input', 10)

        # Control parameters
        self.n_regulators = 3
        self.pressure_max = 65.0
        self.taus         = 3000   # number of random transitions
        # for *each* of the taus+1 cycles we pick a new ramp_time
        # self.ramp_times   = 1.0 + np.random.uniform(0.0, 5.0, size=(self.taus+1,)) ## static
        # self.ramp_times   = 0.2 + np.random.uniform(0.0, 1.0, size=(self.taus+1,))  ## dynamic
        self.ramp_times   = 0.2 + np.random.uniform(0.0, 1.0, size=(self.taus+1,))
        self.hold_time    = 2.5
        self.dt           = 1.0 / 100.0
        self.current_index = 0

        # Precompute per-cycle durations and their cumulative start times
        cycle_durations = self.ramp_times + self.hold_time
        self.cycle_starts = np.concatenate(([0.0], np.cumsum(cycle_durations)))
        total_time = cycle_durations.sum() #  self.ramp_times + self.hold_time
        self.N = int(total_time / self.dt)

        # Generate the full pressure sequence
        self.Uf = self.gen_pressure_data()

        # Register a save on shutdown
        atexit.register(self._save_ramp_info)

        # Start the control loop
        self.timer = self.create_timer(self.dt, self.send_control_signal)


    def gen_pressure_data(self):
        # Random target pressures: one extra so we have taus+1 cycles
        p_vals = self.pressure_max * np.random.rand(self.taus+2, self.n_regulators)
        p_vals[0] = 0.0

        Uf = np.zeros((self.N+1, self.n_regulators))
        for i in range(self.N+1):
            t = i * self.dt
            # find which cycle we're in
            idx = np.searchsorted(self.cycle_starts, t, side='right') - 1
            if idx >= self.taus+1:
                idx = self.taus  # clamp to last cycle
            t_into = t - self.cycle_starts[idx]
            ramp = self.ramp_times[idx]
            if t_into < ramp:
                α = t_into / ramp
                Uf[i] = p_vals[idx] * (1 - α) + p_vals[idx+1] * α
            else:
                Uf[i] = p_vals[idx+1]
        return Uf


    def send_control_signal(self):
        ts = time.time()
        if self.current_index < len(self.Uf):
            P1, P2, P3 = self.Uf[self.current_index]
            self.current_index += 1
        else:
            P1 = P2 = P3 = 0.0

        # Publish to ROS
        msg = Float64MultiArray()
        msg.data = [ts, float(P1), float(P2), float(P3)]
        self.pub.publish(msg)
        self.get_logger().debug(f"Published: {msg.data}")

        # Send to Arduino
        cmd = f"A,{P1:.2f},{P2:.2f},{P3:.2f},Z\n"
        self.ser.write(cmd.encode('utf-8'))
        self.get_logger().info(f"Sent to Arduino: {cmd.strip()}")


    def _save_ramp_info(self):
        # Save ramp_times, hold_time, and the sequence on shutdown
        try:
            # 1) ramp parameters
            with open('pressure_params.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['cycle_index','ramp_time'])
                for i, rt in enumerate(self.ramp_times):
                    w.writerow([i, rt])
                # also write hold_time as a final row
                w.writerow([])
                w.writerow(['hold_time', self.hold_time])

            # 2) full sequence
            np.savetxt(
                'pressure_sequence.csv',
                self.Uf,
                delimiter=',',
                header='P1,P2,P3',
                comments=''
            )

            print('💾 Saved pressure_params.csv and pressure_sequence.csv')
        except Exception as e:
            print(f"Failed to save pressure info: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RegulatorControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down regulator_control")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
