import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Direct path to src config files so changes are read live 
    src_dir = os.path.expanduser('~/ros_ws/src/robot_path_follow')
    cal_file_path = os.path.join(src_dir, 'config', 'cal.yaml')
    params_file_path = os.path.join(src_dir, 'config', 'params.yaml')

    # 1. Camera Server Node 
    camera_server_node = Node(
        package='robot_path_follow',
        executable='camera_server2',
        name='camera_server2',
        output='screen',
        parameters=[{'camera_index': 2}],
    )

    # 2. Path Mapper Node 
    path_mapper_node = Node(
        package='robot_path_follow',
        executable='path_mapper',
        name='path_mapper',
        output='screen',
        parameters=[{'cal_file': cal_file_path}, {'spacing': 0.15}],
    )

    # 3. AprilTag Detector Node 
    apriltag_pose_node = Node(
        package='robot_path_follow',
        executable='apriltag_pose',
        name='apriltag_pose',
        output='screen',
        parameters=[params_file_path, {'cal_file': cal_file_path}],
    )

    # 4. Path Generator Node 
    path_generator_node = Node(
        package='robot_path_follow',
        executable='path_generator4',
        name='path_generator4',
        output='screen',
        parameters=[{'cal_file': cal_file_path}],
        remappings=[
            ('camera/image_raw', '/camera/image_raw'),
            ('robot_pose', '/robot_pose'),
        ],
    )

    # 5. Path Follower Node 
    path_follower_node = Node(
        package='robot_path_follow',
        executable='path_follower_pure4',
        name='path_follower',
        output='screen',
        parameters=[params_file_path, {'cal_file': cal_file_path}],
        remappings=[
            ('left_wheel/rpm', 'motor2/cmd_rpm'),
            ('right_wheel/rpm', 'motor1/cmd_rpm'),
        ],
    )

    return LaunchDescription([
        camera_server_node,
        path_mapper_node,
        apriltag_pose_node,
        path_generator_node,
        path_follower_node,
    ])

