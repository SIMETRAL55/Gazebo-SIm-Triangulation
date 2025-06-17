#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory
     
from thermal_3d_localization.thermal_detector import ThermalDetector
     
class DetectorTester(Node):
    def __init__(self, config_path, raw_dir, detected_dir):
        super().__init__('detector_tester')
        self.bridge = CvBridge()
        self.raw_dir = raw_dir
        self.detected_dir = detected_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.detected_dir, exist_ok=True)

        # publisher for synthetic image
        self.pub_image = self.create_publisher(
            Image, '/thermal_camera_corner_1/image', 1)

        # subscription to detector output
        self.received_uv = None
        self.create_subscription(
            PointStamped,
            '/thermal_camera_corner_1/point_uv',
            self.uv_callback, 1)

        # subscription to debug image
        self.debug_img = None
        self.create_subscription(
            Image,
            '/thermal_camera_corner_1/debug_image',
            self.debug_callback, 1)

        # detector node
        self.detector = ThermalDetector(config_path=config_path)

    def uv_callback(self, msg):
        # when detector publishes, save received_uv
        self.received_uv = (int(msg.point.x), int(msg.point.y))

    def debug_callback(self, msg):
        # save debug overlay image
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        idx = len(os.listdir(self.detected_dir))
        cv2.imwrite(f"{self.detected_dir}/{idx:03d}.png", img)
        self.debug_img = img

    def publish_and_save(self, u, v, idx):
        # create synthetic raw image
        img = np.zeros((240, 320), dtype=np.uint16)
        cv2.circle(img, (int(u), int(v)), 3, 65535, -1)
        # save raw
        raw8 = (img >> 8).astype('uint8')
        cv2.imwrite(f"{self.raw_dir}/{idx:03d}.png", raw8)
        # publish
        msg = self.bridge.cv2_to_imgmsg(img, 'mono16')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'thermal_camera_corner_1/link/thermal_camera'
        self.pub_image.publish(msg)

def calculate_accuracy(results):
    """
    Calculate the pixel accuracy of detection by comparing the detected pixel to the ground truth.

    For each frame, if detection is missed, we assign a maximum error based on the image diagonal.
    Then, an accuracy is computed as: 
        accuracy = 100 * (1 - (average_error / max_error)).
    
    Args:
        results (list of tuple): Each element is a tuple ( (true_u, true_v), detected_pixel )
                                  where detected_pixel is either (u, v) if detection occurred or None.

    Returns:
        tuple: (accuracy_percentage, average_error)
    """
    # Maximum error is the image diagonal for a 320x240 image.
    max_error = (320**2 + 240**2) ** 0.5
    errors = []
    for gt, det in results:
        if det is None:
            error = max_error
        else:
            error = ((gt[0]-det[0])**2 + (gt[1]-det[1])**2) ** 0.5
        errors.append(error)
    avg_error = sum(errors) / len(errors)
    accuracy_percent = max(0, 100 * (1 - avg_error / max_error))
    return accuracy_percent, avg_error

def main():
    rclpy.init()
    pkg_share = get_package_share_directory('thermal_3d_localization')
    config_path = os.path.join(pkg_share, 'config', 'cam1.yaml')

    raw_folder = os.path.expanduser('~/thermal_test/raw')
    det_folder = os.path.expanduser('~/thermal_test/detected')

    tester = DetectorTester(config_path, raw_folder, det_folder)
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    executor.add_node(tester.detector)

    detection_results = []  # list of tuples: (ground truth, detection)

    try:
        for i in range(50):
            # random blob
            u = np.random.uniform(10, 309)
            v = np.random.uniform(10, 229)
            tester.received_uv = None
            tester.debug_img = None

            tester.publish_and_save(u, v, i)
            # Wait up to 0.3 seconds for detection, spinning in short increments.
            start_time = tester.get_clock().now()
            while (tester.received_uv is None and 
                   (tester.get_clock().now() - start_time).nanoseconds < 300000000):
                executor.spin_once(timeout_sec=0.05)

            if tester.received_uv:
                print(f"Frame {i:03d}: true=({u:.1f},{v:.1f}), detected={tester.received_uv}")
            else:
                print(f"Frame {i:03d}: true=({u:.1f},{v:.1f}), detection missed")
            detection_results.append(((u, v), tester.received_uv))
        # Calculate and print the pixel accuracy after all frames.
        accuracy, avg_error = calculate_accuracy(detection_results)
        print(f"Detection Accuracy: {accuracy:.2f}% (Average Pixel Error: {avg_error:.2f})")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
