#!/usr/bin/env python3
#
# Elevator CV node (placeholder). Subscribes to /elevator/vision/enable; when True,
# publishes button detection on three topics: pose (PoseStamped), confidence (Float32),
# has_detection (Bool). Placeholder: fixed pose in camera_optical_frame, confidence 1.0.
# No custom message package; std_msgs and geometry_msgs only.
#

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped

TOPIC_VISION_ENABLE = '/elevator/vision/enable'
TOPIC_BUTTON_POSE = '/elevator/button_detection/pose'
TOPIC_BUTTON_CONFIDENCE = '/elevator/button_detection/confidence'
TOPIC_BUTTON_HAS_DETECTION = '/elevator/button_detection/has_detection'

# Placeholder: pose frame and fixed position (1 m in front of camera, in camera_optical_frame)
PLACEHOLDER_FRAME_ID = 'camera_optical_frame'
PLACEHOLDER_X = 0.0
PLACEHOLDER_Y = 0.0
PLACEHOLDER_Z = 1.0  # 1 m in front
PUBLISH_RATE_HZ = 1.0


class ElevatorCVNode(Node):
    def __init__(self):
        super().__init__('elevator_cv')
        self._enabled = False
        self.create_subscription(Bool, TOPIC_VISION_ENABLE, self._enable_callback, 10)
        self._pose_pub = self.create_publisher(PoseStamped, TOPIC_BUTTON_POSE, 10)
        self._confidence_pub = self.create_publisher(Float32, TOPIC_BUTTON_CONFIDENCE, 10)
        self._has_detection_pub = self.create_publisher(Bool, TOPIC_BUTTON_HAS_DETECTION, 10)
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._timer_callback)

    def _enable_callback(self, msg: Bool):
        self._enabled = msg.data

    def _timer_callback(self):
        if not self._enabled:
            return
        stamp = self.get_clock().now().to_msg()
        # Pose (placeholder: fixed in camera optical frame)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = PLACEHOLDER_FRAME_ID
        pose.pose.position.x = PLACEHOLDER_X
        pose.pose.position.y = PLACEHOLDER_Y
        pose.pose.position.z = PLACEHOLDER_Z
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)
        self._confidence_pub.publish(Float32(data=1.0))
        self._has_detection_pub.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    node = ElevatorCVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
