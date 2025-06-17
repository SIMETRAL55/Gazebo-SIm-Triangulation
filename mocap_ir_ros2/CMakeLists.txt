cmake_minimum_required(VERSION 3.10)
project(mocap_ir_ros2)

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ros_gz_sim REQUIRED)            # ← add
find_package(ros_gz_image REQUIRED)          # ← add
# find_package(launch REQUIRED)                # ← add
# find_package(launch_ros REQUIRED)            # ← add

ament_auto_find_build_dependencies()         # ← add this

install(DIRECTORY launch worlds
  DESTINATION share/${PROJECT_NAME}/
)

ament_auto_package()
