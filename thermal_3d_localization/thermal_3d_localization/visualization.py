#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

class Viz(Node):
    def __init__(self):
        super().__init__('thermal_viz')
        self.sub=self.create_subscription(PointStamped,'/thermal_point_3d',self.cb,10)
        self.pub=self.create_publisher(Marker,'/thermal_point_marker',10)

    def cb(self,msg:PointStamped):
        m=Marker()
        m.header=msg.header; m.ns='thermal'; m.id=0
        m.type=Marker.SPHERE; m.action=Marker.ADD
        m.pose.position=msg.point; m.scale.x=m.scale.y=m.scale.z=0.05
        m.color.r=1.0; m.color.a=1.0
        self.pub.publish(m)

def main():
    rclpy.init()
    node=Viz()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
