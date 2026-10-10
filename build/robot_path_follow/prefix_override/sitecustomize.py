import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pie/ros_ws/src/robot_path_follow/install/robot_path_follow'
