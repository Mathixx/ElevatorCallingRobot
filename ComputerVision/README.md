# ElevatorCallingRobot

ROS 2 workspace: **yolo_msgs** (custom messages) + **yolo_node** (YOLO inference on camera or video).

## Build

From this directory (the colcon workspace root):

```bash
colcon build --packages-select yolo_msgs yolo_node
```

Rebuild **yolo_msgs** whenever you change `.msg` files, then rebuild **yolo_node**.

## Source the overlay (required)

Every new terminal that runs `ros2` or subscribes to **yolo_msgs** must load the install space (so `ros2 topic echo` and tools know your message types):

```bash
cd /path/to/ElevatorCallingRobot
source install/setup.bash
```

Without this, commands like `ros2 topic echo /yolo/detected_buttons` may report an invalid/unknown type.

## Run the node

Start your camera (or whatever publishes `sensor_msgs/Image` on the color topic), then:

```bash
source install/setup.bash
ros2 run yolo_node run_node
```

Useful parameters:

| Parameter | Default | Notes |
|-----------|---------|--------|
| `show_display` | `true` | `false` for headless (no OpenCV window); still publishes |
| `image_topic` | `/camera/color/image_raw` | Input camera topic |
| `use_reliable_qos` | `true` | Set `false` if the camera uses sensor QoS |
| `detections_topic` | `/yolo/detected_buttons` | Output topic for structured detections |
| `conf_threshold` | `0.25` | YOLO confidence |
| `iou_threshold` | `0.4` | YOLO IoU |

Example (no GUI):

```bash
ros2 run yolo_node run_node --ros-args -p show_display:=false
```

## Published topics (two)

1. **`/image`** — `sensor_msgs/Image` (`bgr8`)  
   Annotated view: boxes, Sobel overlay in each box, centroid dots. Use for debugging / Foxglove Image panel.

2. **`/yolo/detected_buttons`** (override with `detections_topic`) — `yolo_msgs/DetectedButtons`  
   One message per processed frame: `header` (time + `frame_id` from the camera when using live input), plus `detections[]`. Each entry has `class_id`, `score`, `center_x` / `center_y`, `theta`, `size_x` / `size_y`, and **`offset_from_center_x`** (pixels from the image vertical midline in the **640×640 inference** frame; use for alignment).  
   Use for navigation / logic; echo only works after `source install/setup.bash`.

```bash
ros2 topic echo /yolo/detected_buttons
```

Model path and weights are set in `yolo_node/yolo_node/vision.py` (`main`); adjust there if your layout differs.
