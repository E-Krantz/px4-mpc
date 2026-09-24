#!/usr/bin/env python3
"""
Setpoint publisher: periodic 60 s trajectory in an ENU world frame.

  t =  0-20 s : hold (0.5, 0)
  t = 20-25 s : straight line (0.5, 0) -> (1.5, 0)
  t = 25-35 s : hold (1.5, 0)
  t = 35-105 s : full circle, radius 1 m, about (2.5, 0), starting/ending at (1.5, 0)
  t = 105-115 s : hold (1.5, 0)
  t = 115-120 s : straight line (1.5, 0) -> (0.5, 0)
  then repeat

Body frame is FLU with x forward, so the attitude at every instant is a pure
yaw that makes body-x point at the circle centre (2.5, 0).
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# --- Trajectory definition (ENU, metres / seconds) ---------------------------
CENTRE = (2.5, 0.0)      # point the vehicle always faces, and circle centre
RADIUS = 1.0
P_START = (0.5, 0.0)     # dwell point
P_CIRCLE = (1.5, 0.0)    # circle entry/exit point (= CENTRE - RADIUS in x)
Z = 0.0

T_HOLD_START_END = 20.0
T_OUT_END = 25.0
T_WAIT_END = 35.0
T_CIRCLE_END = 105.0
T_WAIT2_END = 115.0
T_PERIOD = 120.0

# Circle starts at P_CIRCLE, which sits at angle pi as seen from CENTRE.
THETA_0 = math.pi
CIRCLE_DURATION = T_CIRCLE_END - T_WAIT_END
CCW = True  # set False for clockwise


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _smoothstep(s):
    """C1-continuous ease in/out; returns s in [0, 1] mapped to [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def position_at(t, smooth=True):
    """Position (x, y) on the trajectory at time t (seconds, any t >= 0)."""
    t = t % T_PERIOD
    ease = _smoothstep if smooth else (lambda s: s)

    if t < T_HOLD_START_END:
        return P_START, 'hold_start'

    if t < T_OUT_END:
        s = (t - T_HOLD_START_END) / (T_OUT_END - T_HOLD_START_END)
        return _lerp(P_START, P_CIRCLE, ease(s)), 'move_out'

    if t < T_WAIT_END:
        return P_CIRCLE, 'wait_before_circle'

    if t < T_CIRCLE_END:
        s = (t - T_WAIT_END) / CIRCLE_DURATION
        sweep = 2.0 * math.pi * (1.0 if CCW else -1.0)
        theta = THETA_0 + sweep * ease(s)
        return (CENTRE[0] + RADIUS * math.cos(theta),
                CENTRE[1] + RADIUS * math.sin(theta)), 'circle'

    if t < T_WAIT2_END:
        return P_CIRCLE, 'wait_after_circle'

    s = (t - T_WAIT2_END) / (T_PERIOD - T_WAIT2_END)
    return _lerp(P_CIRCLE, P_START, ease(s)), 'move_back'


def yaw_towards_centre(x, y):
    """Yaw about world z so that body x (forward, FLU) points at CENTRE."""
    dx = CENTRE[0] - x
    dy = CENTRE[1] - y
    if dx * dx + dy * dy < 1e-12:   # degenerate: sitting on the centre
        return 0.0
    return math.atan2(dy, dx)


def quat_from_yaw(yaw):
    """Return (qw, qx, qy, qz) for a rotation of `yaw` about world z (ENU)."""
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


class SetpointPublisher(Node):
    def __init__(self):
        super().__init__('setpoint_publisher')

        self.namespace = self.declare_parameter('namespace', '').value
        self.smooth = self.declare_parameter('smooth', True).value

        self.publisher_ = self.create_publisher(
            PoseStamped, f'px4_mpc/setpoint_pose', 10)

        self.timer_period = 0.01  # seconds -> 100 Hz
        time.sleep(5)             # give time for all inits...

        self.t0 = self.get_clock().now()
        self.last_segment = None
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        (px, py), segment = position_at(t, self.smooth)
        yaw = yaw_towards_centre(px, py)
        qw, qx, qy, qz = quat_from_yaw(yaw)

        pose = PoseStamped()
        pose.header.frame_id = 'mocap'
        pose.header.stamp = now.to_msg()
        pose.pose.position.x = float(px)
        pose.pose.position.y = float(py)
        pose.pose.position.z = float(Z)
        pose.pose.orientation.w = float(qw)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        self.publisher_.publish(pose)

        if segment != self.last_segment:
            self.get_logger().info(
                f'[t={t % T_PERIOD:5.1f}s] {segment}: '
                f'p=({px:.3f}, {py:.3f}, {Z:.3f}) yaw={math.degrees(yaw):7.2f} deg')
            self.last_segment = segment


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