#!/usr/bin/env python3


import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class scan_calculator(object):
    def __init__(self, receive_topic_name, pub_topic_name, publish_rate):
        self.receive_topic_name = receive_topic_name
        self.pub_topic_name = pub_topic_name
        self.publish_rate = publish_rate
        self.msg = LaserScan()
        self.correction_angle = np.pi

    def listen(self, msg):
        self.msg = msg
        axis_roll = int(self.correction_angle/self.msg.angle_increment)
        self.msg.ranges = np.roll(self.msg.ranges, axis_roll)
        self.msg.intensities = np.roll(self.msg.intensities, axis_roll)

    def loop_sub(self):
        rospy.logwarn("Starting to read and publish...")
        rospy.Subscriber(self.receive_topic_name, LaserScan, self.listen)
        rospy.init_node('Custom_ODOM')
        rate = rospy.Rate(self.publish_rate)
        pub_scan = rospy.Publisher(self.pub_topic_name, LaserScan, queue_size=5)

        while not rospy.is_shutdown():
            try:
                pub_scan.publish(self.msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")
            except rospy.ROSTimeMovedBackwardsException:
                rospy.logerr("ROS Time Backwards! Just ignore the exception!")

def main():
    rospy.init_node('Custom_ODOM')
    receive_topic_name = "/tractor1/scan"
    pub_topic_name = "/tractor1/custom_scan"
    publish_rate = 5 # hz
    scan_1 = scan_calculator(receive_topic_name, pub_topic_name, publish_rate)
    scan_1.loop_sub()
    rospy.logwarn("Finished reading!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
