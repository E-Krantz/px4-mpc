#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time

class SetpointPublisher(Node):
    def __init__(self):
        super().__init__('cw_planner')

        self.namespace = self.declare_parameter('namespace', '').value
        self.orbit_period = self.declare_parameter('orbit_period', 2.0).value  # period in minutes

        # Initial setpoint and CW offsets.
        self.delta_t0 = 20.0  # Time to hold initial position before starting CW motion (seconds)
        self.Ax = [1/2, 3/8] # Amplitude in x-direction
        self.y_center = [2.0, 5/4] # Center of the circle in y-direction
        self.pos0 = np.array([0.0, 2.0, 0.0], dtype=float)

        self.n = 2.0 * np.pi / (self.orbit_period * 60.0)  # angular velocity in rad/s

        self.periods = 1.0 # Number of orbits in each formation

        self.trajectory_publisher = self.create_publisher(JointTrajectory, 'px4_mpc/setpoint_trajectory', 10)
        self.odom_publisher = self.create_publisher(Odometry, 'px4_mpc/setpoint_odom', 10)
        self.timer_period = 0.01  # seconds
        time.sleep(1) # Give time for all inits...
        self.counter = 0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.t0 = time.monotonic()

        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]
        self.omega = np.array([0.0, 0.0, 0.0])  # Constant angular rate

        self.get_logger().info(
            f'CW planner started: hold={self.delta_t0:.1f}s, pos0={self.pos0.tolist()}, '
            f'Ax={self.Ax}, y_center={self.y_center}, orbit_period={self.orbit_period} min, n={self.n:.5f} rad/s'
        )

    def evaluate_reference(self, t_elapsed, t_query=None):
        """Evaluate reference at t_query, but clamp to current phase at t_elapsed."""
        T_hold = 30.0
        T_orbit = self.periods * self.orbit_period * 60.0

        static_poses = [
            np.array([0.5, 1.0, 0.0]),
            np.array([-0.5, 1.5, 0.0]),
            np.array([0.5, 2.0, 0.0]),
            # np.array([-0.5, 1.0, 0.0]),
        ]

        phases = [('hold', self.delta_t0, None)]
        for sp in static_poses:
            phases.append(('static', T_hold, sp))
        for i in range(len(self.Ax)):
            phases.append(('orbit', T_orbit, i))

        # Find current phase based on t_elapsed
        t = t_elapsed
        current_phase = None
        current_t_in_phase = 0.0
        for phase_type, duration, param in phases:
            if t < duration:
                current_phase = (phase_type, duration, param)
                current_t_in_phase = t
                break
            t -= duration

        if current_phase is None:
            self.t0 = time.monotonic()
            return self.pos0.copy(), np.zeros(3)

        # Use t_query if provided, but clamp to current phase
        if t_query is not None:
            dt_from_now = t_query - t_elapsed
            t_in_phase = current_t_in_phase + dt_from_now
            # Clamp to end of current phase
            t_in_phase = min(t_in_phase, current_phase[1] - 1e-6)
            t_in_phase = max(t_in_phase, 0.0)
        else:
            t_in_phase = current_t_in_phase

        phase_type, duration, param = current_phase
        if phase_type == 'hold':
            return self.pos0.copy(), np.zeros(3)
        elif phase_type == 'static':
            return param.copy(), np.zeros(3)
        elif phase_type == 'orbit':
            Ax, yc = self.Ax[param], self.y_center[param]
            pos = np.array([Ax * np.sin(self.n * t_in_phase),
                            yc + 2 * Ax * np.cos(self.n * t_in_phase), 0.0])
            vel = np.array([Ax * self.n * np.cos(self.n * t_in_phase),
                            -2 * Ax * self.n * np.sin(self.n * t_in_phase), 0.0])
            return pos, vel

    def timer_callback(self):
        t_elapsed = time.monotonic() - self.t0

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mocap'
        msg.joint_names = ['x', 'y', 'z']

        odom = Odometry()
        odom.header.frame_id = 'mocap'
        odom.child_frame_id = 'base_link'
        odom.header.stamp = msg.header.stamp
        odom.pose.pose.orientation.w = float(self.q[0])
        odom.pose.pose.orientation.x = float(self.q[1])
        odom.pose.pose.orientation.y = float(self.q[2])
        odom.pose.pose.orientation.z = float(self.q[3])

        odom.twist.twist.angular.x = float(self.omega[0])
        odom.twist.twist.angular.y = float(self.omega[1])
        odom.twist.twist.angular.z = float(self.omega[2])

        N = 30
        dt_mpc = 0.1
        for i in range(N+1):
            t_query = t_elapsed + i * dt_mpc
            pos, vel = self.evaluate_reference(t_elapsed, t_query)

            point = JointTrajectoryPoint()
            point.positions = pos.tolist()
            point.velocities = vel.tolist()
            point.time_from_start.sec = int(i * dt_mpc)
            point.time_from_start.nanosec = int((i * dt_mpc % 1) * 1e9)

            msg.points.append(point)

            if i == 0:
                odom.pose.pose.position.x = float(pos[0])
                odom.pose.pose.position.y = float(pos[1])
                odom.pose.pose.position.z = float(pos[2])

                odom.twist.twist.linear.x = float(vel[0])
                odom.twist.twist.linear.y = float(vel[1])
                odom.twist.twist.linear.z = float(vel[2])
        
        self.trajectory_publisher.publish(msg)
        self.odom_publisher.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SetpointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
