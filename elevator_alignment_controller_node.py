#!/usr/bin/env python3
#
# Alignment controller: subscribes to button_detection (pose, confidence, has_detection)
# and optionally /elevator/alignment/request. When has_detection and confidence above
# threshold (and request True if used), transforms button pose to base_link, computes
# cmd_vel (yaw + lateral centering), publishes /cmd_vel and /elevator/alignment/status.
# No Nav2; direct Twist control.
#

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Float32, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

TOPIC_BUTTON_POSE = '/elevator/button_detection/pose'
TOPIC_BUTTON_CONFIDENCE = '/elevator/button_detection/confidence'
TOPIC_BUTTON_HAS_DETECTION = '/elevator/button_detection/has_detection'
TOPIC_ALIGNMENT_REQUEST = '/elevator/alignment/request'
TOPIC_ALIGNMENT_STATUS = '/elevator/alignment/status'
TOPIC_CMD_VEL = '/cmd_vel'

CONFIDENCE_THRESHOLD = 0.5
YAW_TOLERANCE_RAD = 0.05
LATERAL_TOLERANCE_M = 0.03
CONTROL_RATE_HZ = 20.0
K_YAW = 0.8
K_LATERAL = 0.3
MAX_ANGULAR_Z = 0.5
MAX_LINEAR_Y = 0.2
ALIGNMENT_TIMEOUT_SEC = 60.0


class ElevatorAlignmentControllerNode(Node):
    def __init__(self):
        super().__init__('elevator_alignment_controller')
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_pose = None
        self._latest_confidence = 0.0
        self._latest_has_detection = False
        self._alignment_requested = False

        self.create_subscription(PoseStamped, TOPIC_BUTTON_POSE, self._pose_cb, 10)
        self.create_subscription(Float32, TOPIC_BUTTON_CONFIDENCE, self._confidence_cb, 10)
        self.create_subscription(Bool, TOPIC_BUTTON_HAS_DETECTION, self._has_detection_cb, 10)
        self.create_subscription(Bool, TOPIC_ALIGNMENT_REQUEST, self._request_cb, 10)

        self._cmd_vel_pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)
        self._status_pub = self.create_publisher(String, TOPIC_ALIGNMENT_STATUS, 10)

        self._status = 'idle'
        self._align_start_time = None
        self._control_timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self._control_callback)

    def _pose_cb(self, msg: PoseStamped):
        self._latest_pose = msg

    def _confidence_cb(self, msg: Float32):
        self._latest_confidence = msg.data

    def _has_detection_cb(self, msg: Bool):
        self._latest_has_detection = msg.data

    def _request_cb(self, msg: Bool):
        self._alignment_requested = msg.data
        if msg.data and self._status == 'idle':
            self._status = 'aligning'
            self._align_start_time = self.get_clock().now()
            self._status_pub.publish(String(data='aligning'))

    def _control_callback(self):
        if self._status == 'idle':
            self._status_pub.publish(String(data='idle'))
            return
        if self._status == 'done' or self._status == 'failed':
            return

        if self._status == 'aligning':
            if self._align_start_time is not None:
                elapsed = (self.get_clock().now() - self._align_start_time).nanoseconds / 1e9
                if elapsed > ALIGNMENT_TIMEOUT_SEC:
                    self._status = 'failed'
                    self._status_pub.publish(String(data='failed'))
                    self._cmd_vel_pub.publish(Twist())
                    return

            if not (self._latest_has_detection and self._latest_confidence >= CONFIDENCE_THRESHOLD):
                self._cmd_vel_pub.publish(Twist())
                return
            if self._latest_pose is None:
                return

            try:
                pose_base = self._tf_buffer.transform(self._latest_pose, 'base_link')
            except TransformException as ex:
                self.get_logger().warn(f'Transform failed: {ex}', throttle_duration_sec=2.0)
                return

            x = pose_base.pose.position.x
            y = pose_base.pose.position.y
            yaw_to_button = math.atan2(y, x)
            lateral_error = y

            if abs(yaw_to_button) < YAW_TOLERANCE_RAD and abs(lateral_error) < LATERAL_TOLERANCE_M:
                self._status = 'done'
                self._status_pub.publish(String(data='done'))
                self._cmd_vel_pub.publish(Twist())
                return

            cmd = Twist()
            cmd.angular.z = max(-MAX_ANGULAR_Z, min(MAX_ANGULAR_Z, K_YAW * yaw_to_button))
            cmd.linear.y = max(-MAX_LINEAR_Y, min(MAX_LINEAR_Y, -K_LATERAL * lateral_error))
            self._cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ElevatorAlignmentControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
