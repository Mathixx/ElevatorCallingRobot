"""Build yolo_msgs/DetectedButtons from YOLO + frame_boxes outputs."""

from copy import deepcopy

from std_msgs.msg import Header
from yolo_msgs.msg import ButtonDetection, DetectedButtons


def build_detected_buttons(
    header: Header,
    bboxes,
    centroids,
    boxes,
    inference_width: float,
) -> DetectedButtons:
    """Centroids, sizes, and offset_from_center_x use the inference pixel frame."""
    msg = DetectedButtons()
    msg.header = deepcopy(header)
    half_w = float(inference_width) / 2.0
    for bbox, cen, box in zip(bboxes, centroids, boxes):
        x1, y1, x2, y2 = bbox.astype(float)
        cx = float(cen[0])
        d = ButtonDetection()
        d.class_id = int(box.cls.item())
        d.score = float(box.conf.item()) if box.conf is not None else 0.0
        d.center_x = cx
        d.center_y = float(cen[1])
        d.theta = 0.0
        d.size_x = float(x2 - x1)
        d.size_y = float(y2 - y1)
        d.offset_from_center_x = cx - half_w
        msg.detections.append(d)
    return msg
