#!/usr/bin/env python3

from __future__ import print_function
import rospy
import numpy as np
from lidar_reader_pkg.srv import closest_distance_finder_service, closest_distance_finder_serviceResponse
from sensor_msgs.msg import LaserScan


class Lidar_Service(object):
    def __init__(self, topic_name):
        self.topic = topic_name

        # Region of interest
        self.angle_min_view = 120 * np.pi / 180
        self.angle_max_view = 240 * np.pi / 180

        self.lidar_data = []
        self.lidar_data_cleaned = []
        self.theta = []
        self.min_dist = None
        self.wait = True

    def listen_once(self, msg):
        self.msg = msg
        self.angle_inc = msg.angle_increment
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.time_inc = msg.time_increment
        self.range_min = msg.range_min
        self.range_max = msg.range_max
        self.ind_min = int((self.angle_min_view - self.angle_min) // self.angle_inc)
        self.ind_max = int((self.angle_max_view - self.angle_min) // self.angle_inc)
        self.theta = np.arange(self.angle_min_view, self.angle_max_view - self.angle_inc, self.angle_inc) - np.pi / 2
        self.lidar_data = [float(j) if not np.isinf(float(j)) else self.range_max for j in msg.ranges]
        self.lidar_data = self.lidar_data[self.ind_min:self.ind_max]
        for _, value in enumerate(self.lidar_data):
            if value != 0.0:
                self.lidar_data_cleaned.append(value)
        self.min_dist = min(self.lidar_data_cleaned)
        self.sub.unregister()
        self.wait = False
        self.lidar_data_cleaned = []


    def callback(self, request):
        self.sub = rospy.Subscriber(self.topic, LaserScan, self.listen_once)
        while True:
            if not self.wait:
                value = self.min_dist
                self.min_dist = self.range_max
                print("Closest obstacle distance is: ", value)
                return closest_distance_finder_serviceResponse(value)


    def loop(self):
        rospy.init_node('closest_distance_finder_server')
        s = rospy.Service('find_closest_distance', closest_distance_finder_service, self.callback)
        print("Ready to find closest obstacle distance.......")
        rospy.spin()


if __name__ == "__main__":
    lidar_1_topic = "/tractor1/scan"
    lidar_1_service = Lidar_Service(topic_name=lidar_1_topic)
    lidar_1_service.loop()