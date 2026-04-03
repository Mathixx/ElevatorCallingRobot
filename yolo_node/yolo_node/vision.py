import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from yolo_node.frame_boxes import frame_boxes


def _subscription_qos(use_reliable: bool):
    """QoS must match the camera publisher or the subscriber receives nothing.

    Stretch tutorials use create_subscription(..., 10) → RELIABLE, depth 10.
    Stock Intel realsense2_camera often uses sensor/BEST_EFFORT — set use_reliable_qos:=false if needed.
    """
    if use_reliable:
        return QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
    return qos_profile_sensor_data


class YOLO_Node(Node):

<<<<<<< HEAD
    def __init__(self, parent_path, model_path, video_path=None):
        super().__init__('yolo_node')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        # Default True: matches Stretch capture_image / edge_detection tutorials (depth 10, RELIABLE).
        self.declare_parameter('use_reliable_qos', True)
=======
    def __init__(self, parent_path, model_path, video_path, params=[0.001, 0.7]):
        super().__init__('yolo_node')

        self.params = params
>>>>>>> e9247cc3a4e062decf92db96d06b3e3e3f1fa645

        self.publisher_ = self.create_publisher(Image, 'image', 10)
        self.bridge = CvBridge()

        self.model = YOLO(os.path.join(parent_path, model_path))

        self.current_img = None
        self.frame_idx = 0

        self.window_name = 'YOLO'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.use_file = video_path is not None
        self.latest_frame = None
        self._infer_busy = False
        self._got_camera_frame = False

        if self.use_file:
            self.cap = cv2.VideoCapture(os.path.join(parent_path, video_path))
            self.gui_timer = self.create_timer(0.01, self.gui_callback)
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
            # Fast path: only buffer frames; inference runs on a timer so the GUI keeps updating.
            self.inference_timer = self.create_timer(1.0 / 30.0, self.inference_tick)
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
        if not self._got_camera_frame:
            self._got_camera_frame = True
            self.get_logger().info('First camera frame received.')

    def inference_tick(self):
        if self.latest_frame is None or self._infer_busy:
            return
        self._infer_busy = True
        try:
            frame = self.latest_frame.copy()
            self.run_inference(frame)
        finally:
            self._infer_busy = False

    def process_frame_from_file(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('End of video or invalid frame')
            return

        self.run_inference(frame)

    def run_inference(self, frame):
        frame = cv2.resize(frame, (640, 640))
        self.current_img = frame.copy()

        results = self.model(frame, conf=self.params[0], iou=self.params[1])[0]

        bboxes, sobel_maps, centroids, classes = frame_boxes(results, self.current_img)
        print(sum(classes))

        for box, sobel, cen, cla in zip(bboxes, sobel_maps, centroids, classes):
            x1, y1, x2, y2 = box.astype(int)

            self.current_img[y1:y2, x1:x2] = sobel
            cv2.rectangle(self.current_img, (x1, y1), (x2, y2), (0, 255 if cla == 0 else 0, 255 if cla == 1 else 0), 2)
            cv2.circle(self.current_img, (int(cen[0]), int(cen[1])), 5, (255, 0, 0), -1)

        msg = self.bridge.cv2_to_imgmsg(self.current_img, encoding='bgr8')
        self.publisher_.publish(msg)

    def gui_callback(self):
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
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

<<<<<<< HEAD
    parent_path = '/home/stretch-re1/Desktop/ElevatorCallingRobot/yolo_node/yolo_node/'
    model_path = 'MODELS/two_cls_bcew.torchscript'
    video_path = None
=======
    parent_path = '/home/test/ros2_ws/src/yolo_node/yolo_node/'
    model_path = 'MODELS/SOTA_two_cls_bcew_penalty.torchscript'
    video_path = 'DATA_video_streams/video_3.mp4'
    params = [0.25, 0.4]
    print(params)
>>>>>>> e9247cc3a4e062decf92db96d06b3e3e3f1fa645

    node = YOLO_Node(parent_path, model_path, video_path, params)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()


if __name__ == '__main__':
    main()
