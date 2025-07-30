# Gazebo Thermal Camera Triangulation

A ROS2-based motion capture system using thermal cameras for 3D target localization through triangulation in Gazebo simulation environment.

## Overview

This project implements a thermal camera-based motion capture system that uses multiple thermal cameras positioned around a room to detect and triangulate the 3D position of thermal targets. The system consists of two main ROS2 packages:

- **mocap_ir_ros2**: Gazebo simulation environment with thermal cameras
- **thermal_3d_localization**: 3D triangulation and localization algorithms

## Features

- 🎯 **Multi-camera triangulation**: Uses 8 thermal cameras for robust 3D localization
- 🌡️ **Thermal imaging**: Specialized thermal camera sensors in Gazebo simulation
- 📐 **Advanced triangulation**: Implements both ground-plane intersection and linear least-squares methods
- 🎮 **ROS2 integration**: Full ROS2 Humble compatibility with message synchronization
- 📊 **Real-time visualization**: RViz markers for camera FOV, detection rays, and target position
- 🔧 **Configurable parameters**: JSON-based camera calibration and system configuration

## System Architecture

### Camera Setup
The system uses 8 thermal cameras positioned strategically:
- **4 Corner cameras**: Positioned at room corners at 5m height looking down
- **4 Mid cameras**: Positioned at walls at 2.5m height for side coverage

### Coordinate System
- **World frame**: `map` (configurable)
- **Camera frames**: Individual frames for each thermal camera
- **Target frame**: `thermal_target` for the triangulated position

## Installation

### Prerequisites
- ROS2 Humble or Iron
- Gazebo Garden/Harmonic
- Python 3.8+
- OpenCV
- NumPy

### Dependencies
```sudo apt install ros-humble-gazebo-ros-pkgs```\
```sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge ros-humble-ros-gz-image```\
```sudo apt install python3-opencv python3-numpy```\
```pip install message-filters```


### Build Instructions
**Clone the repository**

```git clone https://github.com/SIMETRAL55/Gazebo-SIm-Triangulation.git```\
```cd Gazebo-SIm-Triangulation```

**Build the packages**\
```colcon build```

**Source the workspace**\
```source install/setup.bash```


## Usage

### 1. Launch the Gazebo Simulation
```ros2 launch mocap_ir_ros2 mocap_ir.launch.py```

This launches:
- Gazebo world with thermal cameras
- Image bridge for camera feeds
- Camera info bridge for calibration data

### 2. Start the Triangulation Node
```ros2 run thermal_3d_localization triangulation_node```

### 3. Visualize in RViz
```ros2 run rviz2 rviz2```

Add the following topics:
- `/triangulation_markers` (MarkerArray) - Camera FOV and detection visualization
- `/thermal_target/position` (PointStamped) - 3D target position
- Camera image topics for thermal feeds


## Algorithm Details

### Triangulation Methods

1. **Ground-Plane Intersection**: Projects rays to ground plane (z=0) for initial estimate
2. **Linear Least-Squares**: Minimizes ray-to-point distances for optimal 3D position
3. **Outlier Removal**: Filters detections with high reprojection errors

### Synchronization
Uses time-based message synchronization with configurable time window to handle multiple camera feeds.

### Visualization
Provides comprehensive RViz visualization including:
- Camera positions and orientations
- Field-of-view polygons
- Detection rays from 2D to 3D
- Ground-plane intersections
- Final triangulated target position

