#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Elias Krantz"
__contact__ = "eliaskra@kth.se"

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='wrench_cw',
        description='Mode of the controller (rate, wrench, direct_allocation)'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all nodes'
    )

    setpoint_from_rviz_arg = DeclareLaunchArgument(
        'setpoint_from_rviz',
        default_value='False',
        description='Publish setpoint pose via rviz'
    )
    px4_uses_ned_arg = DeclareLaunchArgument(
        'px4_uses_ned',
        default_value='True',
        description='PX4 uses NED frame (default: true) or ENU frame (false)'
    )
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='False',
        description='Enable camera'
    )
    orbit_period_arg = DeclareLaunchArgument(
        'orbit_period',
        default_value='100.0',
        description='Period of the orbit in minutes'
    )
    skip_build_arg = DeclareLaunchArgument(
        'skip_build',
        default_value='False',
        description='Skip code generation and building of acados solver (faster)'
    )

    namespace = LaunchConfiguration('namespace')
    setpoint_from_rviz = LaunchConfiguration('setpoint_from_rviz')
    px4_uses_ned = LaunchConfiguration('px4_uses_ned')
    camera = LaunchConfiguration('camera')
    orbit_period = LaunchConfiguration('orbit_period')
    skip_build = LaunchConfiguration('skip_build')
    
    ld = LaunchDescription()
    ld.add_action(namespace_arg)
    ld.add_action(setpoint_from_rviz_arg)
    ld.add_action(px4_uses_ned_arg)
    ld.add_action(camera_arg)
    ld.add_action(orbit_period_arg)
    ld.add_action(skip_build_arg)
    
    ld.add_action(Node(
        package='px4_mpc',
        namespace=namespace,
        executable='mpc_spacecraft',
        name='mpc_spacecraft',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'mode': 'wrench_cw'},
            {'setpoint_from_rviz': setpoint_from_rviz},
            {'px4_uses_ned': px4_uses_ned},
            {'camera': camera},
            {'orbit_period': orbit_period},
            {'skip_build': skip_build}
        ]
    ))
    
    ld.add_action(Node(
        package='px4_mpc',
        namespace=namespace,
        executable='rviz_pos_marker',
        name='rviz_pos_marker',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(setpoint_from_rviz)
    ))
    
    ld.add_action(Node(
        package='px4_mpc',
        namespace=namespace,
        executable='cw_planner',
        name='cw_planner',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'orbit_period': orbit_period},  # period in minutes
        ]
    ))
    
    ld.add_action(Node(
        package='px4_mpc',
        namespace=namespace,
        executable='visualizer',
        name='visualizer',
        parameters=[
            {'px4_uses_ned': px4_uses_ned},
            {'camera': camera}
        ],
        condition=IfCondition(setpoint_from_rviz)
    ))
    
    ld.add_action(OpaqueFunction(function=launch_setup))

    # ros2 launch realsense2_camera rs_launch.py publish_tf:=true
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_inertial',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'inertial'],
        condition=IfCondition(camera)
    ))
    
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_camera',
        arguments=['2', '1.9', '2.3', '0.3010647', '0.3013046', '-0.6395013', '0.6400107', 'map', 'camera_link'], # camera 2
        condition=IfCondition(camera)
    ))

    return ld

def patch_rviz_config(original_config_path, namespace):
    """
    Patch the RViz configuration file to replace the namespace placeholder with the actual namespace.
    """
    with open(original_config_path, 'r') as f:
        content = f.read()

    # Replace placeholder with actual namespace
    content = content.replace('__NS__', f'/{namespace}' if namespace else '')

    # Write to temporary file
    tmp_rviz_config = tempfile.NamedTemporaryFile(delete=False, suffix='.rviz')
    tmp_rviz_config.write(content.encode('utf-8'))
    tmp_rviz_config.close()

    return tmp_rviz_config.name


def launch_setup(context, *args, **kwargs):
    """
    Function to set up the launch context and patch the RViz configuration.
    """
    namespace = LaunchConfiguration('namespace').perform(context)
    rviz_config_path = os.path.join(get_package_share_directory('px4_mpc'), 'config.rviz')
    patched_config = patch_rviz_config(rviz_config_path, namespace)

    return [
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', patched_config],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        )
    ]