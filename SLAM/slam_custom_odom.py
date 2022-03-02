#!/usr/bin/env python3


import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry


class odom_calculator(object):
    def __init__(self, receive_topic_name, pub_topic_name, publish_rate):
        self.receive_topic_name = receive_topic_name
        self.pub_topic_name = pub_topic_name
        self.publish_rate = publish_rate
        self.msg = Odometry()

    def get_euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.

        Input
          :param roll: The roll (rotation around x-axis) angle in radians.
          :param pitch: The pitch (rotation around y-axis) angle in radians.
          :param yaw: The yaw (rotation around z-axis) angle in radians.

        Output
          :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [qx, qy, qz, qw]

    def listen(self, msg):
        self.msg = msg
        pose_x = self.msg.pose.pose.position.x
        pose_y = self.msg.pose.pose.position.y
        pose_z = self.msg.pose.pose.position.z
        pose_qx = self.msg.pose.pose.orientation.x
        pose_qy = self.msg.pose.pose.orientation.y
        pose_qz = self.msg.pose.pose.orientation.z
        pose_qw = self.msg.pose.pose.orientation.w
        rx, ry, rz = self.get_euler_from_quaternion(pose_qx, pose_qy, pose_qz, pose_qw)
        rx_new = -ry
        ry_new = rx
        rz_new = rz
        new_x, new_y, new_z, new_w = self.get_quaternion_from_euler(rx_new, ry_new, rz_new)

        self.msg.pose.pose.position.x = -pose_y
        self.msg.pose.pose.position.y = pose_x
        self.msg.pose.pose.position.z = pose_z
        self.msg.pose.pose.orientation.x = new_x
        self.msg.pose.pose.orientation.y = new_y
        self.msg.pose.pose.orientation.z = new_z
        self.msg.pose.pose.orientation.w = new_w


    def loop_sub(self):
        rospy.logwarn("Starting to read and publish...")
        rospy.Subscriber(self.receive_topic_name, Odometry, self.listen)
        rospy.init_node('Custom_ODOM')
        rate = rospy.Rate(self.publish_rate)
        pub_odom = rospy.Publisher(self.pub_topic_name, Odometry, queue_size=10)

        while not rospy.is_shutdown():
            try:
                pub_odom.publish(self.msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")
            except rospy.ROSTimeMovedBackwardsException:
                rospy.logerr("ROS Time Backwards! Just ignore the exception!")

def main():
    rospy.init_node('Custom_ODOM')
    receive_topic_name = "/tractor1/odom"
    pub_topic_name = "/tractor1/custom_odom"
    publish_rate = 15 # hz
    odom_1 = odom_calculator(receive_topic_name, pub_topic_name, publish_rate)
    odom_1.loop_sub()
    rospy.logwarn("Finished reading!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
