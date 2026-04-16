#!/usr/bin/env python3
#
# Launches elevator CV node and alignment controller (Phase 1 button-pressing).
# Run after Stretch + Nav2 are up; then run the elevator_task_master node.
# No ROS package required; scripts run via python3 from this directory.
#

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    script_dir = get_script_dir()
    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', os.path.join(script_dir, 'elevator_cv_node.py')],
            cwd=script_dir,
            output='screen',
            name='elevator_cv',
        ),
        ExecuteProcess(
            cmd=['python3', os.path.join(script_dir, 'elevator_alignment_controller_node.py')],
            cwd=script_dir,
            output='screen',
            name='elevator_alignment_controller',
        ),
    ])
