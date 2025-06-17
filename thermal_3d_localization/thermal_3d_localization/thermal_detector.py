#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2, yaml, numpy as np

class ThermalDetector(Node):
    def __init__(self, config_path=None):
        super().__init__('thermal_detector')
        # Load config from argument or ROS parameter
        if config_path:
            cfg = yaml.safe_load(open(config_path, 'r'))
        else:
            self.declare_parameter('config', '')
            cfg_file = self.get_parameter('config').value
            if not cfg_file:
                self.get_logger().error('Missing "config" parameter')
                raise RuntimeError('Missing config file')
            cfg = yaml.safe_load(open(cfg_file, 'r'))

        self.bridge = CvBridge()
        K = np.array(cfg['camera_info']['K']).reshape(3,3)
        D = np.array(cfg['camera_info']['D'])
        w, h = cfg['camera_info']['width'], cfg['camera_info']['height']
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)

        topic = cfg['image_topic']
        self.pub = self.create_publisher(PointStamped, topic.replace('/image','/point_uv'), 10)
        self.pub_dbg = self.create_publisher(Image, topic.replace('/image','/debug_image'), 10)
        self.create_subscription(Image, topic, self.cb_image, 10)

    def cb_image(self, msg):
        img_raw = self.bridge.imgmsg_to_cv2(msg, 'mono16')
        img = cv2.remap(img_raw, self.map1, self.map2, cv2.INTER_LINEAR)
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return
        u, v = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        pt = PointStamped()
        pt.header = msg.header
        pt.point.x = float(u)
        pt.point.y = float(v)
        pt.point.z = 0.0
        self.pub.publish(pt)

        dbg = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dbg, [c], -1, (0,255,0), 2)
        cv2.circle(dbg, (u,v), 5, (0,0,255), -1)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = msg.header
        self.pub_dbg.publish(out)

def main():
    rclpy.init()
    # allow config_path injection when instantiated
    node = ThermalDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
