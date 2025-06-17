import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    world_path = os.path.join(
        get_package_share_directory('mocap_ir_ros2'),
        'worlds',
        'mocap_ir.sdf'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ]),
        launch_arguments={'gz_args': world_path}.items()
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='image_bridge_ros_subscriber',
        arguments=['/camera/image_raw', 'thermal_camera_16_bit', 
                   '/thermal_camera_corner_1/image', '/thermal_camera_corner_2/image',
                   '/thermal_camera_corner_3/image', '/thermal_camera_corner_4/image',
                   '/thermal_camera_mid_front/image', '/thermal_camera_mid_back/image',
                   '/thermal_camera_mid_left/image', '/thermal_camera_mid_right/image'],
        output='screen'
    )

    info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge_ros_subscriber',
        arguments=['/thermal_camera_corner_1/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo', '/thermal_camera_corner_2/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
                   '/thermal_camera_corner_3/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo', '/thermal_camera_corner_4/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
                   '/thermal_camera_mid_front/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo', '/thermal_camera_mid_back/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
                   '/thermal_camera_mid_left/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo', '/thermal_camera_mid_right/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo'],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        image_bridge,
        info_bridge
    ])
