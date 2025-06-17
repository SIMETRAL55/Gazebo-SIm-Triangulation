#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import ModelStates
import numpy as np

class Validator(Node):
    def __init__(self):
        super().__init__('validator')
        self.est=None; self.gt=None
        self.sub_e=self.create_subscription(PointStamped,'/thermal_point_3d',self.cb_est,10)
        self.sub_g=self.create_subscription(ModelStates,'/gazebo/model_states',self.cb_gt,10)
    def cb_est(self,msg):
        self.est=np.array([msg.point.x,msg.point.y,msg.point.z])
        self.eval()
    def cb_gt(self,msg):
        if 'box' in msg.name:
            i=msg.name.index('box')
            p=msg.pose[i].position
            self.gt=np.array([p.x,p.y,p.z])
            self.eval()
    def eval(self):
        if self.est is not None and self.gt is not None:
            err=np.linalg.norm(self.est-self.gt)
            self.get_logger().info(f'Error: {err:.4f} m')
            self.est=None; self.gt=None

def main():
    rclpy.init()
    node=Validator(); rclpy.spin(node); rclpy.shutdown()

if __name__=='__main__':
    main()
