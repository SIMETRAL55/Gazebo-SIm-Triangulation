from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('thermal_3d_localization')
    cfgs = [os.path.join(pkg,'config',f'cam{i}.yaml') for i in range(1,9)]
    return LaunchDescription([
      Node(package='thermal_3d_localization',
           executable='thermal_detector_node',
           name=f'det{i}',
           parameters=[{'config':cfgs[i-1]}],
           output='screen', 
           arguments=['--ros-args','--log-level','det{i}:=DEBUG']) for i in range(1,9)
    ] + [
      Node(package='thermal_3d_localization',
           executable='triangulation_node',
           name='triangulator',
           output='screen', 
           arguments=['--ros-args','--log-level','triangulator:=DEBUG'])
    ])
