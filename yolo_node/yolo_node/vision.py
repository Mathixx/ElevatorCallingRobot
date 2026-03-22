import os
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from yolo_node.frame_boxes import frame_boxes


class YOLO_Node(Node):

    def __init__(self, parent_path, model_path, video_path):
        super().__init__('yolo_node')

        self.publisher_ = self.create_publisher(Image, 'image', 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(os.path.join(parent_path, video_path))

        self.model = YOLO(os.path.join(parent_path, model_path))

        self.current_img = None
        self.frame_idx = 0

        self.window_name = "YOLO"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # GUI timer
        self.gui_timer = self.create_timer(0.01, self.gui_callback)

        # Process first frame
        self.process_frame()

    def process_frame(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video or invalid frame")
            return

        frame = cv2.resize(frame, (640, 640))
        self.current_img = frame.copy()
        # print(self.current_img.shape, self.current_img.dtype)

        results = self.model(frame, conf=0.25, iou=0.6)[0]

        bboxes, sobel_maps, centroids, classes = frame_boxes(results, self.current_img)
        print(sum(classes))

        for box, sobel, cen, cla in zip(bboxes, sobel_maps, centroids, classes):
            x1, y1, x2, y2 = box.astype(int)

            self.current_img[y1:y2, x1:x2] = sobel
            cv2.rectangle(self.current_img, (x1, y1), (x2, y2), (0, 255 if cla == 0 else 0, 255 if cla == 1 else 0), 2)
            cv2.circle(self.current_img, (int(cen[0]), int(cen[1])), 5, (255, 0, 0), -1)


        # Publish ROS image
        msg = self.bridge.cv2_to_imgmsg(self.current_img, encoding="bgr8")
        self.publisher_.publish(msg)

    def gui_callback(self):

        if self.current_img is not None:
            cv2.imshow(self.window_name, self.current_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):   # next frame
            self.frame_idx += 1
            self.get_logger().info(f"Frame {self.frame_idx}")
            self.process_frame()

        if key == ord('q'):   # previous frame
            self.frame_idx = max(self.frame_idx - 1, 0)
            self.get_logger().info(f"Frame {self.frame_idx}")
            self.process_frame()

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.shutdown()

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    parent_path = 'home/test/ros2_ws/src/yolo_node/yolo_node/'
    model_path = 'MODELS/two_cls_bcew.torchscript'
    video_path = 'DATA_video_streams/video_3.mp4'

    node = YOLO_Node(parent_path, model_path, video_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()


if __name__ == "__main__":
    main()