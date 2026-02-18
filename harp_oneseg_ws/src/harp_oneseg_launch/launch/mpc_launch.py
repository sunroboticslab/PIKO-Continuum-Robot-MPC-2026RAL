#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # 1) Start OptiTrack UDP streaming immediately
    streaming_node = Node(
        package='optitrack_streaming',
        executable='udp_optitrack_publisher',
        name='optitrack_udp_publisher',
        output='screen',
    )

    # 2) Delay 2s, then start MPC regulator control
    mpc_node = Node(
        package='data_collection',
        executable='mpc_regulator_control',
        name='mpc_regulator_control',
        output='screen',

    )
    delayed_mpc = TimerAction(
        period=2.0,
        actions=[mpc_node],
    )

    ld = LaunchDescription()
    ld.add_action(streaming_node)
    ld.add_action(delayed_mpc)
    return ld
