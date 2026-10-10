import os
from glob import glob
from setuptools import setup

package_name = "robot_path_follow"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="Phone-drawn path to world path for a differential-drive robot",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_server = robot_path_follow.camera_server_node:main",
            "path_mapper = robot_path_follow.path_mapper_node:main",
            "apriltag_pose = robot_path_follow.apriltag_pose_node:main",
            "path_follower_pure2 = robot_path_follow.path_follower_pure2:main",
            "path_follower_carrot = robot_path_follow.path_follower_carrot:main",
            "path_follower = robot_path_follow.path_follower_node:main",
            "path_follower_pure = robot_path_follow.path_follower_pure:main",
            "path_follower_pure3 = robot_path_follow.path_follower_pure3:main",
            "path_follower_pure4 = robot_path_follow.path_follower_pure4:main",
            "obstacle_planner = robot_path_follow.obstacle_planner:main",
        ],
    },
)
