from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    camera_server = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_path_follow', 'camera_server',
             '--ros-args', '-p', 'camera_index:=0'],
        output='screen')

    path_mapper = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_path_follow', 'path_mapper',
             '--ros-args',
             '-p', 'cal_file:=ros_ws/src/robot_path_follow/config/cal.yaml',
             '-p', 'spacing:=0.15'],
        output='screen')

    apriltag_pose = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_path_follow', 'apriltag_pose',
             '--ros-args',
             '--params-file', 'ros_ws/src/robot_path_follow/config/params.yaml',
             '-p', 'cal_file:=ros_ws/src/robot_path_follow/config/cal.yaml'],
        output='screen')

    path_follower_pure = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_path_follow', 'path_follower_pure',
             '--ros-args',
             '--params-file', 'ros_ws/src/robot_path_follow/config/params.yaml',
             '-r', 'left_wheel/rpm:=motor2/cmd_rpm',
             '-r', 'right_wheel/rpm:=motor1/cmd_rpm'],
        output='screen')

    return LaunchDescription([
        camera_server,
        path_mapper,
        apriltag_pose,
        path_follower_pure,
    ])
