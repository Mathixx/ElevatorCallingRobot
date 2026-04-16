# Elevator Task Master (simple)

One script + one config. Navigate Stretch RE1 to the elevator on a precomputed map, then run the button-pressing sequence (placeholders). No ROS package, no `colcon`, no `install/` or `log/`.

**This IS a ROS2 node** (inherits from `Node`) and can interact with other nodes via topics, services, and actions. It's just not launched via `ros2 run`/`ros2 launch`—you run it directly with `python3`.

## What you need

- ROS 2 Humble, Stretch RE1, Nav2: `sudo apt install ros-humble-nav2-simple-commander`
- A saved map and Nav2 running (e.g. `ros2 launch stretch_nav2 navigation.launch.py map:=/path/to/map.yaml`)
- Optional: PyYAML to load config (`pip install pyyaml`)

## Usage

1. **Terminal 1** — Start Stretch + Nav2 with your map:
   ```bash
   ros2 launch stretch_nav2 navigation.launch.py map:=/home/stretch-re1/Desktop/ElevatorOpener/map/my_lab_map.yaml
   ```
   If you don't use `use_initial_pose` in the config, set the robot's initial pose (e.g. via Foxglove Studio via SSH, or programmatically).

2. **Optional — Terminal 2** — Launch RViz to check that the robot has loaded its initial pose correctly:
   ```bash
   source /opt/ros/humble/setup.bash
   rviz2
   ```
   In RViz: set **Fixed Frame** to `map`, then add **Map** (topic `/map`) and **TF**. You should see the map and the robot pose. If you use a saved config next time: `rviz2 -d path/to/your.rviz`.

3. **Terminal 2 or 3** — Run the node script from this directory:
   ```bash
   source /opt/ros/humble/setup.bash
   # source your Stretch workspace if needed
   cd /path/to/ElevatorOpener
   python3 elevator_task_master_node.py
   ```

No build step. No `install/` or `log/` folders.

## Config

Edit **`elevator_params.yaml`** in this directory. **No defaults:** the script errors if the file is missing or any required key is absent.

- **Required:** `elevator_x`, `elevator_y`, `elevator_yaw` — goal in front of the elevator (map frame, m and rad)
- `use_initial_pose` — if `true`, the robot's initial pose is set from the file (headless). **When true, these are required:** `initial_pose_x`, `initial_pose_y`, `initial_pose_yaw` (map frame, m and rad)

Get poses from Foxglove Studio (2D Nav Goal / 2D Pose Estimate) or from your map. The elevator pose and (when enabled) initial pose come only from this params file.

## Launching RViz to check initial pose

To confirm the robot has correctly loaded its initial pose (from params or from AMCL), launch RViz in a **second terminal** while Nav2 is running:

```bash
source /opt/ros/humble/setup.bash
# source your Stretch workspace if needed
rviz2
```

Then in RViz:

1. Set **Fixed Frame** (left panel, under "Global Options") to `map`.
2. Click **Add** → **By topic** → choose **Map** (topic `/map`) → OK.
3. Click **Add** → **By display type** → **TF** → OK.

You should see the map and the robot’s pose (from TF). Save the config via **File → Save Config As** (e.g. `rviz/check_initial_pose.rviz`) so next time you can run:

```bash
rviz2 -d /path/to/ElevatorOpener/rviz/check_initial_pose.rviz
```

## Interacting with other nodes

This is a ROS2 node, so you can:
- **Create service clients** (e.g. for Stretch lift/arm): `self.create_client(ServiceType, '/service_name')`
- **Create subscribers** (e.g. camera topics): `self.create_subscription(MsgType, '/topic', callback, qos)`
- **Create publishers** (e.g. status): `self.create_publisher(MsgType, '/topic', qos)`
- **Call services**: `future = client.call_async(request); rclpy.spin_until_future_complete(self, future)`

See comments in `elevator_task_master_node.py` for examples.

## Phase 1: Button-pressing sequence (CV + alignment)

Phase 1 adds head‑90° right, CV (placeholder), and alignment so the robot faces and centers on the button. **Alignment uses `cmd_vel` only (no Nav2).**

### Launch order

1. **Terminal 1** — Stretch + Nav2 (as in Usage above):
   ```bash
   ros2 launch stretch_nav2 navigation.launch.py map:=/home/stretch-re1/Desktop/ElevatorOpener/map/my_lab_map.yaml
   ```

2. **Terminal 2** — CV + alignment nodes (manipulation launch):
   ```bash
   source /opt/ros/humble/setup.bash
   cd /home/stretch-re1/Desktop/ElevatorOpener
   ros2 launch elevator_manipulation.launch.py
   ```
   Or from anywhere: `ros2 launch /path/to/ElevatorOpener/elevator_manipulation.launch.py`

3. **Terminal 3** — Master node (after Nav2 is up and manipulation nodes are running):
   ```bash
   source /opt/ros/humble/setup.bash
   cd /home/stretch-re1/Desktop/ElevatorOpener
   python3 elevator_task_master_node.py
   ```

The master will: (1) navigate to the elevator, (2) turn head 90° right, (3) enable CV and wait for button detection, (4) request alignment and wait until status is `done`, then finish Phase 1.

### Phase 1 topics (std_msgs / geometry_msgs only)

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/elevator/vision/enable` | std_msgs/Bool | master | CV node |
| `/elevator/button_detection/pose` | geometry_msgs/PoseStamped | CV node | master, alignment |
| `/elevator/button_detection/confidence` | std_msgs/Float32 | CV node | master, alignment |
| `/elevator/button_detection/has_detection` | std_msgs/Bool | CV node | master, alignment |
| `/elevator/alignment/request` | std_msgs/Bool | master | alignment |
| `/elevator/alignment/status` | std_msgs/String | alignment | master |
| `/cmd_vel` | geometry_msgs/Twist | alignment | Stretch base |

The CV node is a **placeholder**: when enabled it publishes a fixed pose (in `camera_optical_frame`), confidence 1.0, and `has_detection` true at ~1 Hz. Replace with a real detector later.

## Manipulation (beyond Phase 1)

Lift height, real button detection, and arm press are not implemented yet. Implement those in `execute_button_pressing_sequence()` and/or new nodes as needed.
