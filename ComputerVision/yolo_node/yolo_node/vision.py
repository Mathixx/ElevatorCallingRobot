import os
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from yolo_msgs.msg import DetectedButtons

from yolo_node.detection_msg import build_detected_buttons
from yolo_node.frame_boxes import frame_boxes


def _subscription_qos(use_reliable: bool):
    """
    Return a QoS profile for the camera image subscription.

    Must match the camera publisher or the subscriber receives nothing.
    Stretch tutorials use create_subscription(..., 10) → RELIABLE, depth 10.
    Stock Intel realsense2_camera often uses sensor/BEST_EFFORT — set
    use_reliable_qos:=false if needed.
    """
    if use_reliable:
        return QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
    return qos_profile_sensor_data


class YOLO_Node(Node):

    def __init__(self, parent_path, model_path, video_path=None):
        super().__init__('yolo_node')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('use_reliable_qos', True)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.4)
        # Default avoids /detected_buttons if another node still publishes vision_msgs there.
        self.declare_parameter('detections_topic', '/yolo/detected_buttons')
        self.declare_parameter('show_display', True)

        self.publisher_ = self.create_publisher(Image, 'image', 10)
        det_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self._detections_pub = self.create_publisher(DetectedButtons, det_topic, 10)
        self.get_logger().info(
            f'Publishing yolo_msgs/DetectedButtons on "{det_topic}" '
            '(remap or set detections_topic to change; use one publisher per topic name).'
        )
        self.bridge = CvBridge()

        self.model = YOLO(os.path.join(parent_path, model_path))

        self.current_img = None
        self.frame_idx = 0

        self.window_name = 'YOLO'
        self._show_display = self.get_parameter('show_display').get_parameter_value().bool_value
        if self._show_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.use_file = video_path is not None
        self.latest_frame = None
        self._latest_image_header = Header()
        self._infer_busy = False
        self._got_camera_frame = False

        if self.use_file:
            self.cap = cv2.VideoCapture(os.path.join(parent_path, video_path))
            if self._show_display:
                self.gui_timer = self.create_timer(0.01, self.gui_callback)
            else:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                if fps <= 1.0 or fps > 120.0:
                    fps = 30.0
                self._playback_timer = self.create_timer(1.0 / fps, self._file_playback_tick)
            self.process_frame_from_file()
        else:
            self.cap = None
            topic = self.get_parameter('image_topic').get_parameter_value().string_value
            reliable = self.get_parameter('use_reliable_qos').get_parameter_value().bool_value
            qos = _subscription_qos(reliable)
            self.get_logger().info(
                f'Subscribing to {topic} (reliable_qos={reliable})'
            )
            self.get_logger().info(
                'Camera must publish this topic in ROS (e.g. '
                '"ros2 launch stretch_core d435i_low_resolution.launch.py"). '
                'stretch_realsense_visualizer.py alone does not publish ROS topics.'
            )
            self.subscription = self.create_subscription(
                Image,
                topic,
                self.image_callback,
                qos,
            )
            self.inference_timer = self.create_timer(1.0 / 30.0, self.inference_tick)
            if self._show_display:
                self.gui_timer = self.create_timer(0.01, self.gui_callback)
            self._diag_timer = self.create_timer(5.0, self._diagnostics_tick)

        self._placeholder = self._make_placeholder()

    def _make_placeholder(self):
        img = np.full((480, 640, 3), 48, dtype=np.uint8)
        msg = 'Waiting for camera...'
        cv2.putText(
            img, msg, (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img,
            'Launch: ros2 launch stretch_core d435i_low_resolution.launch.py',
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return img

    def _file_playback_tick(self):
        self.frame_idx += 1
        self.process_frame_from_file()

    def _diagnostics_tick(self):
        if self._got_camera_frame:
            self._diag_timer.cancel()
            return
        topic = self.get_parameter('image_topic').get_parameter_value().string_value
        reliable = self.get_parameter('use_reliable_qos').get_parameter_value().bool_value
        self.get_logger().warning(
            f'Still no frames on "{topic}" (reliable_qos={reliable}). '
            'Run: ros2 topic list | grep color ; ros2 topic hz <that_topic> . '
            'If hz shows nothing, start the camera launch (see log above). '
            'Try the other QoS: --ros-args -p use_reliable_qos:=false'
        )

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return
        self.latest_frame = frame
        self._latest_image_header = msg.header
        if not self._got_camera_frame:
            self._got_camera_frame = True
            self.get_logger().info('First camera frame received.')

    def inference_tick(self):
        if self.latest_frame is None or self._infer_busy:
            return
        self._infer_busy = True
        try:
            frame = self.latest_frame.copy()
            self.run_inference(frame, self._latest_image_header)
        finally:
            self._infer_busy = False

    def process_frame_from_file(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('End of video or invalid frame')
            return

        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        self.run_inference(frame, hdr)

    def run_inference(self, frame, image_header=None):
        frame = cv2.resize(frame, (640, 640))
        inference_w = float(frame.shape[1])
        self.current_img = frame.copy()

        conf = self.get_parameter('conf_threshold').get_parameter_value().double_value
        iou = self.get_parameter('iou_threshold').get_parameter_value().double_value
        results = self.model(frame, conf=conf, iou=iou)[0]

        bboxes, sobel_maps, centroids, classes = frame_boxes(results, self.current_img)
        boxes_list = list(results.boxes)

        if image_header is None:
            hdr = Header()
            hdr.stamp = self.get_clock().now().to_msg()
        else:
            hdr = deepcopy(image_header)

        # Bbox centers, offset_from_center_x, and sizes use the resized inference frame.
        self._detections_pub.publish(
            build_detected_buttons(hdr, bboxes, centroids, boxes_list, inference_w),
        )

        for box, sobel, cen, cla in zip(bboxes, sobel_maps, centroids, classes):
            x1, y1, x2, y2 = box.astype(int)

            self.current_img[y1:y2, x1:x2] = sobel
            bgr = (0, 255 if cla == 0 else 0, 255 if cla == 1 else 0)
            cv2.rectangle(self.current_img, (x1, y1), (x2, y2), bgr, 2)
            cv2.circle(self.current_img, (int(cen[0]), int(cen[1])), 5, (255, 0, 0), -1)

        msg = self.bridge.cv2_to_imgmsg(self.current_img, encoding='bgr8')
        self.publisher_.publish(msg)

    def gui_callback(self):
        if not self._show_display:
            return
        if self.use_file:
            show = self.current_img
        else:
            show = self.current_img if self.current_img is not None else self._placeholder

        if show is not None:
            cv2.imshow(self.window_name, show)

        key = cv2.waitKey(1) & 0xFF

        if self.use_file and key == ord('e'):
            self.frame_idx += 1
            self.get_logger().info(f'Frame {self.frame_idx}')
            self.process_frame_from_file()

        if self.use_file and key == ord('q'):
            self.frame_idx = max(self.frame_idx - 1, 0)
            self.get_logger().info(f'Frame {self.frame_idx}')
            self.process_frame_from_file()

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.shutdown()

    def shutdown(self):
        if self.cap is not None:
            self.cap.release()
        if self._show_display:
            cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    parent_path = '/home/stretch-re1/Desktop/ElevatorCallingRobot/yolo_node/yolo_node/'
    model_path = 'MODELS/two_cls_bcew.torchscript'
    video_path = None

    node = YOLO_Node(parent_path, model_path, video_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()


if __name__ == '__main__':
    main()
