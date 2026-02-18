# harp_oneseg_launch/launch/harp_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

def generate_launch_description():
    opti_track = Node(
        package='optitrack_streaming',
        executable='udp_optitrack_publisher',
        name='optitrack_udp_receiver'
    )

    regulator = Node(
        package='regulator_input',
        executable='regulator_control',
        name='regulator_control'
    )

    data_collector = Node(
        package='data_collection',
        executable='data_reciever',  # make sure this matches your install
        name='data_collector'
    )

    # when data_collector dies (exit code 0), shut down the entire launch
    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=data_collector,
            on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    return LaunchDescription([
        opti_track,
        regulator,
        data_collector,
        shutdown_on_exit,
    ])
