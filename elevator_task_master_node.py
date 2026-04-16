#!/usr/bin/env python3
#
# Elevator Task Master — ROS2 Node (standalone script, no ROS package).
# This IS a ROS2 node (inherits from Node) and can interact with other nodes via topics/services/actions.
# Run: source /opt/ros/humble/setup.bash (and Stretch workspace); then python3 elevator_task_master_node.py
# Config: elevator_params.yaml in this directory.
#

import math
import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Bool, Float32, String
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

try:
    import yaml
except ImportError:
    yaml = None

# Topic names for button-pressing sequence (Phase 1)
TOPIC_VISION_ENABLE = '/elevator/vision/enable'
TOPIC_BUTTON_POSE = '/elevator/button_detection/pose'
TOPIC_BUTTON_CONFIDENCE = '/elevator/button_detection/confidence'
TOPIC_BUTTON_HAS_DETECTION = '/elevator/button_detection/has_detection'
TOPIC_ALIGNMENT_REQUEST = '/elevator/alignment/request'
TOPIC_ALIGNMENT_STATUS = '/elevator/alignment/status'
ACTION_FOLLOW_JOINT_TRAJECTORY = '/stretch_controller/follow_joint_trajectory'
HEAD_PAN_RIGHT_RAD = 1.5708  # 90 deg right (rad)
CONFIDENCE_THRESHOLD = 0.5
BUTTON_WAIT_TIMEOUT_SEC = 30.0
ALIGNMENT_WAIT_TIMEOUT_SEC = 60.0

# Stow: prefer stretch_core driver service (no extra deps, same process as controller)
STOW_SERVICE_NAME = '/stow_the_robot'  # std_srvs/Trigger; if driver is namespaced, set e.g. '/stretch_driver/stow_the_robot'
# Fallback stow positions (official Stretch ROS2 stow uses FollowJointTrajectory with these)
STOW_LIFT_M = 0.2
STOW_WRIST_EXTENSION_M = 0.0
STOW_WRIST_YAW_RAD = 3.14
STOW_HEAD_PAN_RAD = 0.0  # Forward
STOW_HEAD_TILT_RAD = 0.0  # Forward (adjust if needed)


def yaw_to_quaternion(yaw_rad: float) -> tuple:
    """Convert yaw (radians, rotation about Z) to quaternion (x, y, z, w)."""
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return (0.0, 0.0, qz, qw)


# Required keys: always need elevator goal; if use_initial_pose then need initial pose too.
REQUIRED_ELEVATOR_KEYS = ('elevator_x', 'elevator_y', 'elevator_yaw')
REQUIRED_INITIAL_POSE_KEYS = ('initial_pose_x', 'initial_pose_y', 'initial_pose_yaw')


def load_config():
    """Load elevator_params.yaml from same directory as this script. Errors if file or required params missing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'elevator_params.yaml')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'elevator_params.yaml not found at {path}. Create it with elevator_x, elevator_y, elevator_yaw (and optionally use_initial_pose, initial_pose_*).')
    if yaml is None:
        raise ImportError('PyYAML required to load elevator_params.yaml. Install: pip install pyyaml')
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    # Support both flat keys and nested elevator_task_master.ros__parameters
    if 'elevator_task_master' in data and 'ros__parameters' in data['elevator_task_master']:
        config = data['elevator_task_master']['ros__parameters']
    else:
        config = data

    missing = [k for k in REQUIRED_ELEVATOR_KEYS if k not in config]
    if missing:
        raise ValueError(
            f'elevator_params.yaml missing required keys: {missing}. '
            f'Required: {list(REQUIRED_ELEVATOR_KEYS)}.'
        )
    use_initial = config.get('use_initial_pose', False)
    if use_initial:
        missing_init = [k for k in REQUIRED_INITIAL_POSE_KEYS if k not in config]
        if missing_init:
            raise ValueError(
                f'use_initial_pose is true but elevator_params.yaml missing: {missing_init}. '
                f'Required when use_initial_pose is true: {list(REQUIRED_INITIAL_POSE_KEYS)}.'
            )
    return config


class ElevatorTaskMaster(Node):
    """
    ROS2 Node: Sequential master: Init -> Nav to elevator -> (placeholder) Manipulation -> Done.
    Blocking; run once and leave it.
    
    This IS a ROS2 node and can interact with other nodes:
    - Create service clients: self.create_client(ServiceType, '/service_name')
    - Create subscribers: self.create_subscription(MsgType, '/topic', callback, qos)
    - Create publishers: self.create_publisher(MsgType, '/topic', qos)
    - Call services: client.call_async(request) or client.call(request)
    """
    def __init__(self, config: dict):
        super().__init__('elevator_task_master')
        self.navigator = BasicNavigator()
        self.config = config

        # Head control (FollowJointTrajectory) for Step 1
        self._trajectory_client = ActionClient(self, FollowJointTrajectory, ACTION_FOLLOW_JOINT_TRAJECTORY)
        self._joint_state = None
        self._joint_state_sub = self.create_subscription(
            JointState, '/stretch/joint_states', self._joint_state_callback, 10
        )

        # Button-pressing sequence: publishers
        self._vision_enable_pub = self.create_publisher(Bool, TOPIC_VISION_ENABLE, 10)
        self._alignment_request_pub = self.create_publisher(Bool, TOPIC_ALIGNMENT_REQUEST, 10)

        # Button-pressing sequence: latest from topics (updated by callbacks)
        self._latest_pose = None
        self._latest_confidence = 0.0
        self._latest_has_detection = False
        self._latest_alignment_status = ''

        # Subscribers for button detection and alignment status
        self.create_subscription(
            PoseStamped, TOPIC_BUTTON_POSE, self._button_pose_callback, 10
        )
        self.create_subscription(
            Float32, TOPIC_BUTTON_CONFIDENCE, self._button_confidence_callback, 10
        )
        self.create_subscription(
            Bool, TOPIC_BUTTON_HAS_DETECTION, self._button_has_detection_callback, 10
        )
        self.create_subscription(
            String, TOPIC_ALIGNMENT_STATUS, self._alignment_status_callback, 10
        )

        # Stow: use driver service when available (stretch_core), else trajectory fallback
        self._stow_client = self.create_client(Trigger, STOW_SERVICE_NAME)

    def _joint_state_callback(self, msg: JointState):
        self._joint_state = msg

    def _button_pose_callback(self, msg: PoseStamped):
        self._latest_pose = msg

    def _button_confidence_callback(self, msg: Float32):
        self._latest_confidence = msg.data

    def _button_has_detection_callback(self, msg: Bool):
        self._latest_has_detection = msg.data

    def _alignment_status_callback(self, msg: String):
        self._latest_alignment_status = msg.data

    def _stow_robot(self) -> bool:
        """Stow the robot (camera forward). Uses stretch_core /stow_the_robot service when
        available; otherwise falls back to FollowJointTrajectory (same as official Stretch
        stow_command tutorial). We do not call stretch_body.robot.stow() to avoid
        conflicting with the running ROS2 driver."""
        # Prefer driver service (stretch_core exposes /stow_the_robot)
        if self._stow_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Calling /stow_the_robot service...')
            req = Trigger.Request()
            future = self._stow_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.result() is not None and future.result().success:
                self.get_logger().info('Robot stowed (driver service).')
                return True
            if future.result() is not None:
                self.get_logger().warn(f'Stow service returned success=False: {future.result().message}')
        else:
            self.get_logger().info('Stow service not available; using trajectory fallback.')

        # Fallback: same approach as official Stretch ROS2 stow (FollowJointTrajectory)
        if not self._trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowJointTrajectory action server not available for stow.')
            return False

        joint_dict = {}
        if self._joint_state:
            joint_dict = {name: float(self._joint_state.position[i])
                          for i, name in enumerate(self._joint_state.name)}
        else:
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.5)
                if self._joint_state:
                    joint_dict = {name: float(self._joint_state.position[i])
                                  for i, name in enumerate(self._joint_state.name)}
                    break

        stow_joints = {
            'joint_lift': STOW_LIFT_M,
            'wrist_extension': STOW_WRIST_EXTENSION_M,
            'joint_wrist_yaw': STOW_WRIST_YAW_RAD,
            'head_pan': STOW_HEAD_PAN_RAD,
            'head_tilt': STOW_HEAD_TILT_RAD,
        }
        joint_names = []
        start_positions = []
        target_positions = []
        for joint_name, target_pos in stow_joints.items():
            start_pos = joint_dict.get(joint_name, target_pos)
            joint_names.append(joint_name)
            start_positions.append(start_pos)
            target_positions.append(target_pos)

        if not joint_names:
            self.get_logger().warn('No joints for stow fallback; skipping.')
            return True

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.header.frame_id = 'base_link'
        p0 = JointTrajectoryPoint()
        p0.positions = start_positions
        p0.time_from_start = Duration(seconds=0.0).to_msg()
        p1 = JointTrajectoryPoint()
        p1.positions = target_positions
        p1.time_from_start = Duration(seconds=4.0).to_msg()
        goal.trajectory.points = [p0, p1]

        self.get_logger().info(f'Stowing robot via trajectory (joints: {joint_names})...')
        future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.result():
            self.get_logger().error('Failed to send stow trajectory goal.')
            return False
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
        if not result_future.result():
            self.get_logger().error('Stow trajectory did not complete.')
            return False
        self.get_logger().info('Robot stowed (camera forward).')
        return True

    def _move_head_90_right(self) -> bool:
        """Send FollowJointTrajectory to move head pan to 90 deg right. Returns True on success."""
        if not self._trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowJointTrajectory action server not available.')
            return False
        # Use current head_pan if we have joint_states, else assume 0 (forward)
        start_pan = 0.0
        if self._joint_state and 'head_pan' in self._joint_state.name:
            idx = self._joint_state.name.index('head_pan')
            start_pan = float(self._joint_state.position[idx])
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['head_pan']
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        goal.trajectory.header.frame_id = 'base_link'
        p0 = JointTrajectoryPoint()
        p0.positions = [start_pan]
        p0.time_from_start = Duration(seconds=0.0).to_msg()
        p1 = JointTrajectoryPoint()
        p1.positions = [HEAD_PAN_RIGHT_RAD]
        p1.time_from_start = Duration(seconds=2.0).to_msg()
        goal.trajectory.points = [p0, p1]
        self.get_logger().info('Sending head pan to 90 deg right...')
        future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.result():
            self.get_logger().error('Failed to send head trajectory goal.')
            return False
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
        if not result_future.result():
            self.get_logger().error('Head trajectory did not complete.')
            return False
        self.get_logger().info('Head at 90 deg right.')
        return True

    def _get_required(self, key: str):
        """Get required param; config was already validated so key must exist."""
        if key not in self.config:
            raise ValueError(f'elevator_params.yaml missing required key: {key}')
        return self.config[key]

    def _make_pose_stamped(self, x: float, y: float, yaw_rad: float, frame_id: str = 'map') -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def execute_mission(self):
        self.get_logger().info('Starting Elevator Mission...')

        # Stow robot and ensure camera is forward (before navigation)
        self.get_logger().info('Stowing robot (camera forward)...')
        if not self._stow_robot():
            self.get_logger().error('Stow failed; continuing anyway.')

        if self.config.get('use_initial_pose', False):
            initial_pose = self._make_pose_stamped(
                self._get_required('initial_pose_x'),
                self._get_required('initial_pose_y'),
                self._get_required('initial_pose_yaw'),
            )
            self.navigator.setInitialPose(initial_pose)
            self.get_logger().info('Initial pose set from elevator_params.yaml')
        else:
            self.get_logger().info(
                'use_initial_pose is false: not setting initial pose from config. '
                'Set robot pose in RViz/Foxglove first, or set use_initial_pose: true in elevator_params.yaml.'
            )

        self.get_logger().info('Waiting for Nav2...')
        self.navigator.waitUntilNav2Active()

        elevator_pose = self._make_pose_stamped(
            self._get_required('elevator_x'),
            self._get_required('elevator_y'),
            self._get_required('elevator_yaw'),
        )
        self.get_logger().info('Navigating to elevator...')
        self.navigator.goToPose(elevator_pose)

        feedback_interval = 2.0
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback is not None:
                self.get_logger().info(
                    f'Distance remaining: {feedback.distance_remaining:.2f} m',
                    throttle_duration_sec=feedback_interval,
                )

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Arrived at elevator.')
            self.execute_button_pressing_sequence()
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('Navigation canceled.')
        elif result == TaskResult.FAILED:
            self.get_logger().error('Navigation failed.')

    def execute_button_pressing_sequence(self):
        """
        Execute Phase 1 button-pressing sequence:
        Step 1: Head 90° right (FollowJointTrajectory).
        Step 2: Enable CV and wait for button (has_detection + confidence >= threshold).
        Step 3: Request alignment and wait until status "done" or "failed".
        """
        self.get_logger().info('Starting button-pressing sequence (Phase 1).')

        # Step 1 – Head 90° right
        if not self._move_head_90_right():
            self.get_logger().error('Step 1 failed: head 90° right.')
            return
        self.get_logger().info('Step 1 done: head at 90° right.')

        # Comment out the line below to block/stop the sequence here (for testing)
        return

        # Step 2 – Enable CV and wait for button (has_detection and confidence >= threshold)
        self._latest_has_detection = False
        self._latest_confidence = 0.0
        self._vision_enable_pub.publish(Bool(data=True))
        self.get_logger().info('CV enabled; waiting for button detection...')
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=BUTTON_WAIT_TIMEOUT_SEC)
        while self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._latest_has_detection and self._latest_confidence >= CONFIDENCE_THRESHOLD:
                self.get_logger().info('Button detected (confidence >= threshold).')
                break
        if not (self._latest_has_detection and self._latest_confidence >= CONFIDENCE_THRESHOLD):
            self.get_logger().error('Step 2 failed: button not detected within timeout.')
            return
        self.get_logger().info('Step 2 done: button detected.')

        # Step 3 – Request alignment and wait for status "done" or "failed"
        self._latest_alignment_status = ''
        self._alignment_request_pub.publish(Bool(data=True))
        self.get_logger().info('Alignment requested; waiting for status...')
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=ALIGNMENT_WAIT_TIMEOUT_SEC)
        while self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._latest_alignment_status == 'done':
                self.get_logger().info('Alignment done.')
                break
            if self._latest_alignment_status == 'failed':
                self.get_logger().error('Alignment failed.')
                return
        if self._latest_alignment_status != 'done':
            self.get_logger().error('Step 3 failed: alignment did not complete within timeout.')
            return
        self.get_logger().info('Step 3 done: aligned to button.')

        self.get_logger().info('Button-pressing sequence (Phase 1) complete.')


def main(args=None):
    config = load_config()
    rclpy.init(args=args)
    node = ElevatorTaskMaster(config)
    try:
        node.execute_mission()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.navigator.lifecycleShutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
