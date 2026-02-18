#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0
# RigidBodyFrame:
#   <id>,<x>,<y>,<z>,<qx>,<qy>,<qz>,<qw>;
#   …;(timestamp)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import socket

class OptiTrackReceiver(Node):
    """ROS2 node that receives OptiTrack UDP frames and republishes them."""

    def __init__(self):
        super().__init__('optitrack_udp_receiver')

        # Publisher for a flat list: [id, x, y, z, qx, qy, qz, qw, …, timestamp]
        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            '/optitrack/rigid_body_array',
            10
        )

        # Set up UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 5005))
        self.get_logger().info("🟢 Listening for OptiTrack UDP data on port 5005")

        # Timer fires at 100 Hz
        self.timer = self.create_timer(0.01, self.read_udp)

    def read_udp(self):
        """Read a UDP packet, parse rigid bodies, and publish."""
        try:
            self.sock.settimeout(0.001)
            data, _ = self.sock.recvfrom(2048)
            decoded = data.decode('utf-8').strip()

            if not decoded.startswith('RigidBodyFrame:'):
                return

            payload = decoded[len('RigidBodyFrame:'):]
            parts = payload.split(';')
            timestamp = float(parts[-1])
            bodies = parts[:-1]

            flat_array = []
            for body in bodies:
                fields = body.split(',')
                # Expect 8 fields per body: id, x, y, z, qx, qy, qz, qw
                if len(fields) == 8:
                    flat_array.extend(map(float, fields))

            # Append timestamp at end
            flat_array.append(timestamp)

            msg = Float32MultiArray(data=flat_array)
            self.publisher_.publish(msg)
            self.get_logger().info(
                f"📤 Published {len(bodies)} rigid bodies @ {timestamp:.4f}"
            )

        except socket.timeout:
            pass
        except Exception as e:
            self.get_logger().error(f"⚠️ UDP read error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OptiTrackReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔴 Shutting down OptiTrackReceiver")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
