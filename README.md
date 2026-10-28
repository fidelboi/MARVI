# MARVI --- Mobile Autonomous Robot with Vision Intelligence

MARVI is a low-cost, differential-drive mobile robot developed at
**Swami Keshwanand Institute of Technology (SKIT), Jaipur**. It uses
camera-based localization and path tracking to navigate without relying
on an onboard LiDAR sensor. The robot follows waypoints sent from a
mobile device and includes **obstacle avoidance** to help it navigate
around obstacles along its route.

## Features

-   **Vision-based localization:** Uses an overhead USB 2D camera and
    AprilTags to estimate the robot's position and orientation.
-   **Camera calibration and coordinate mapping:** Maps image pixels to
    real-world floor coordinates using camera calibration and a
    homography transformation.
-   **Waypoint-based path planning:** A trajectory can be drawn on a
    mobile device and sent to the robot as a sequence of waypoints.
-   **Path tracking:** Uses the Pure Pursuit algorithm to follow the
    commanded trajectory.
-   **Obstacle avoidance:** Includes obstacle-avoidance behaviour so the
    robot can respond to obstacles instead of blindly following the
    planned path.
-   **ROS 2 integration:** Separates perception, navigation, and motor
    control into modular software components.
-   **Low-cost design:** Uses a Raspberry Pi 4B and camera-based
    perception rather than an expensive onboard LiDAR.
-   **In-house fabrication:** The chassis was designed and fabricated
    using facilities at SKIT.

## How It Works

1.  **Image capture** --- An overhead camera captures the robot and its
    operating area.
2.  **Robot detection** --- AprilTag detection identifies the robot in
    the camera image.
3.  **Localization** --- Camera calibration and homography map image
    coordinates to floor coordinates. The robot's position and heading
    are estimated for navigation.
4.  **Path input** --- A user draws a path on a mobile device. The path
    is represented as waypoints and sent to MARVI.
5.  **Path tracking** --- The navigation system uses Pure Pursuit to
    calculate steering behaviour toward a look-ahead point on the path.
6.  **Obstacle avoidance** --- The obstacle-avoidance logic responds to
    detected obstacles and helps the robot avoid collisions while
    navigating.
7.  **Motor control** --- ROS 2 sends velocity or motion commands to the
    motor-control component, which drives the differential-drive base.

## Hardware

  Component                    Purpose
  ---------------------------- ------------------------------------------
  Raspberry Pi 4B              Main onboard computer
  USB 2D camera                Visual sensing and robot detection
  AprilTag marker              Robot identification and pose estimation
  NEMA 17 stepper motors × 2   Differential-drive motion
  TB6600 motor drivers × 2     Stepper-motor control
  12 V batteries × 2           Power supply
  Custom-fabricated chassis    Mechanical platform

## Software and Algorithms

-   **ROS 2** --- Communication between perception, navigation, and
    motor-control modules.
-   **OpenCV** --- Camera-image processing and computer-vision
    operations.
-   **AprilTag detection** --- Identifies the robot marker in camera
    frames.
-   **Homography / coordinate transformation** --- Converts image
    positions into floor-plane coordinates.
-   **Pose estimation** --- Estimates the robot's position and heading.
-   **Pure Pursuit** --- Tracks a waypoint-based trajectory.
-   **Obstacle avoidance** --- Adds obstacle-responsive behaviour to the
    navigation system.

> Update this section with the exact ROS 2 distribution, camera/AprilTag
> packages, obstacle-detection method, and navigation interfaces used in
> your final implementation.

## System Architecture

``` text
Overhead USB Camera
        |
        v
Image Processing + AprilTag Detection
        |
        v
Camera Calibration / Homography
        |
        v
Robot Pose in Floor Coordinates
        |
        +------------------------------+
                                       |
Mobile Device --> Waypoints --> Path Tracking (Pure Pursuit)
                                       |
                               A* Path Planning
                                       |
                                       v
                              Pure Pursuit Tracking
                                       |
                                       v
                               Motion Commands
                                       |
                                       v
                              ROS 2 Motor Driver
                                       |
                                       v
                          TB6600 Drivers + NEMA 17
                                       |
                                       v
                                Robot Motion
```

## Running the ROS 2 Nodes

Run these commands from the ROS 2 workspace environment, with the workspace built and sourced. The commands below follow the project's current node setup.

### 1. Start the camera server

```bash
ros2 run robot_path_follow camera_server --ros-args -p camera_index:=2
```

Start the second camera/drawing server in a separate terminal if your workflow requires it:

```bash
ros2 run robot_path_follow camera_server2 --ros-args -p camera_index:=2
```

### 2. Start the path mapper

```bash
ros2 run robot_path_follow path_mapper --ros-args \\
  -p cal_file:=ros_ws/src/robot_path_follow/config/cal.yaml \\
  -p spacing:=0.15
```

### 3. Start AprilTag pose estimation

```bash
ros2 run robot_path_follow apriltag_pose --ros-args \\
  --params-file ros_ws/src/robot_path_follow/config/params.yaml \\
  -p cal_file:=ros_ws/src/robot_path_follow/config/cal.yaml
```

### 4. Start path generation

```bash
ros2 run robot_path_follow path_generator4 --ros-args \\
  -p cal_file:=ros_ws/src/robot_path_follow/config/cal.yaml \\
  -r camera/image_raw:=/camera/image_raw \\
  -r robot_pose:=/robot_pose
```

Run each long-running node in its own terminal unless your setup manages the processes another way. Make sure the package is built, the correct workspace is sourced in every terminal, the camera index is correct, and the calibration/parameter file paths exist.

### Launch the system

Instead of starting the nodes individually, you can use the launch file:

```bash
ros2 launch robot_path_follow robot_path_follow.launch.py
```

The launch file should be run from an environment where the `robot_path_follow` package is discoverable by ROS 2.

## ROS 2 Integration

The software is organized into modular components for perception,
navigation, and motor control. A typical command flow is:

1.  The perception component publishes the robot's estimated pose.
2.  The navigation component receives the pose and requested waypoints.
3.  Path tracking and obstacle avoidance determine the required motion.
4.  The motor-control component converts motion commands into motor
    actions.

**Note:** Replace these descriptions with the actual node names, topic
names, message types, and launch commands from your ROS 2 workspace
before publishing the repository.

## Getting Started

The exact setup depends on the ROS 2 distribution, package names, camera
configuration, and motor-control implementation used in your workspace.

### Prerequisites

-   Raspberry Pi 4B with a compatible Linux installation
-   ROS 2 installation compatible with the project
-   USB camera
-   OpenCV and the required AprilTag detection package
-   Motor drivers, stepper motors, power supply, and emergency stop or
    safe power-disconnect arrangement

### Build and run

After cloning the repository and installing its dependencies, use the
commands appropriate for your workspace. For a standard ROS 2 `colcon`
workspace, the build pattern is:

``` bash
cd ~/marvi_ws
source /opt/ros/<ros-distro>/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

Replace `<ros-distro>` with the installed ROS 2 distribution. Launch
commands are intentionally omitted until the repository's actual launch
files and package names are known.

## Calibration and Operation

1.  Secure the overhead camera so it cannot move during operation.
2.  Calibrate the camera and determine the floor-plane homography.
3.  Confirm that the AprilTag is detected reliably and that the
    estimated pose is expressed in the correct coordinate frame.
4.  Verify the waypoint coordinate conversion before enabling motor
    power.
5.  Test path tracking at low speed in a clear area.
6.  Test obstacle avoidance with safe, non-fragile obstacles and a clear
    emergency-stop procedure.
7.  Increase operating speed only after localization, stopping
    behaviour, and obstacle avoidance have been validated.

## Safety Notes

-   Test the robot at low speed and keep an emergency stop or accessible
    power disconnect available.
-   Keep hands, cables, and loose objects away from wheels and moving
    mechanisms.
-   Validate obstacle detection and stopping distance in the actual
    operating environment.
-   Camera-based localization depends on camera placement, lighting,
    marker visibility, and calibration quality.
-   Obstacle avoidance does not guarantee collision-free motion in every
    situation; supervise testing.

## Limitations

-   The overhead camera requires a suitable view of the operating area.
-   Occlusion, poor lighting, camera movement, and calibration errors
    can reduce localization accuracy.
-   Navigation performance depends on waypoint quality, controller
    tuning, motor response, and obstacle-detection reliability.
-   The system is designed around its configured test environment;
    operation in a new environment may require recalibration and
    retesting.

## Future Improvements

-   Add quantitative evaluation of localization error and path-tracking
    accuracy.
-   Measure obstacle-avoidance success rate and minimum clearance across
    repeatable tests.
-   Add recovery behaviour for blocked paths or loss of visual tracking.
-   Document ROS 2 nodes, topics, parameters, launch files, and
    reproducible setup steps.
-   Add demonstrations, architecture diagrams, and recorded test
    results.

## Team

**Team members** - Syed Burhan - Jayesh Dubey - Vishal Kumar

**Mentors** - Dr. Kamal Sharma - Dr. Namrata Saxena

**Institution:** Swami Keshwanand Institute of Technology (SKIT), Jaipur

------------------------------------------------------------------------

*MARVI --- camera-based localization, waypoint navigation, path
tracking, and obstacle avoidance on a low-cost mobile robot.*
