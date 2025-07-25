#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2023 PX4 Development Team. All rights reserved.
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

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint

from mpc_msgs.srv import SetPose

class QuadrotorMPC(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # Get namespace
        self.namespace = self.declare_parameter('namespace', '').value
        self.namespace_prefix = f'/{self.namespace}' if self.namespace else ''

        # Get setpoint from rviz (true/false)
        self.setpoint_from_rviz = self.declare_parameter('setpoint_from_rviz', False).value

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile_sub)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub)

        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose,
                'set_pose',
                self.add_set_pos_callback
            )
        else:
            self.setpoint_pose_sub = self.create_subscription(
                PoseStamped,
                'px4_mpc/setpoint_pose',
                self.get_setpoint_pose_callback,
                0
            )

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', qos_profile_pub)
        self.predicted_path_pub = self.create_publisher(Path, 'px4_mpc/predicted_path', 10)
        self.reference_pub = self.create_publisher(Marker, 'px4_mpc/reference', 10
        )

        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Create Quadrotor and controller objects
        self.model = MultirotorRateModel()

        # Create MPC Solver
        MPC_HORIZON = 15

        # Spawn Controller
        self.mpc = MultirotorRateMPC(self.model)
        # self.ctl = SetpointMPC(model=self.quad,
        #                 dynamics=self.quad.model,
        #                 param='P1',
        #                 N=MPC_HORIZON,
        #                 ulb=ulb, uub=uub, xlb=xlb, xub=xub)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 3.0])

    def vehicle_attitude_callback(self, msg):
        # NED-> ENU transformation
        # Receives quaternion in NED frame as (qw, qx, qy, qz)
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

    def vehicle_local_position_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
    
    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=False
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)

        error_position = self.vehicle_local_position - self.setpoint_position

        x0 = np.array([error_position[0], error_position[1], error_position[2],
         self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2], 
         self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10, 1)

        u_pred, x_pred = self.mpc.solve(x0)

        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + 1
                # Publish time history of the vehicle path
            predicted_pose_msg = self.vector2PoseMsg('map', predicted_state[0:3] + self.setpoint_position, np.array([1.0, 0.0, 0.0, 0.0]))
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        thrust_rates = u_pred[0, :]
        # Hover thrust = 0.73
        thrust_command = -(thrust_rates[0] * 0.07 + 0.0)
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            setpoint_msg = VehicleRatesSetpoint()
            setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            setpoint_msg.roll = float(thrust_rates[1])
            setpoint_msg.pitch = float(-thrust_rates[2])
            setpoint_msg.yaw = float(-thrust_rates[3])
            setpoint_msg.thrust_body[0] = 0.0
            setpoint_msg.thrust_body[1] = 0.0
            setpoint_msg.thrust_body[2] = float(thrust_command)
            self.publisher_rates_setpoint.publish(setpoint_msg)

    def add_set_pos_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z

        return response
    
    def vector2PoseMsg(self, frame_id, position, attitude):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3]
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        return pose_msg

def main(args=None):
    rclpy.init(args=args)

    quadrotor_mpc = QuadrotorMPC()

    rclpy.spin(quadrotor_mpc)

    quadrotor_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
