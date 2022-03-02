#!/usr/bin/env python3

'''
Lidar Subscriber python code
Author: Arjun Pradeep
'''


import rospy
import rosbag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from sensor_msgs.msg import LaserScan
np.seterr(divide='ignore', invalid='ignore')


class lidar_subscriber(object):
    def __init__(self, topic_name):
        # Region of interest
        self.angle_min_view = 120 * np.pi / 180
        self.angle_max_view = 240 * np.pi / 180

        # User choice
        self.animate_rate = 1  # hz
        self.check_distance = 0.35  # m

        self.topic_name = topic_name
        self.lidar_data = []
        self.lidar_data_queue = []
        self.theta = []
        self.collision_alert = False
        self.run_once = True
        self.fig, self.ax = plt.subplots()



    def check_for_collision(self):
        dst_values_corrected = [value for value in self.lidar_data if value > 0.0]
        if len(dst_values_corrected) > 0:
            min_dst = np.min(dst_values_corrected)
            if min_dst < self.check_distance:
                self.collision_alert = True
            else:
                self.collision_alert = False


    def listen(self, msg):
        if self.run_once == True:
            self.angle_inc = msg.angle_increment
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.time_inc = msg.time_increment
            self.range_min = msg.range_min
            self.range_max = msg.range_max
            self.ind_min = int((self.angle_min_view - self.angle_min) // self.angle_inc)
            self.ind_max = int((self.angle_max_view - self.angle_min) // self.angle_inc)
            self.theta = np.arange(self.angle_min_view, self.angle_max_view-self.angle_inc, self.angle_inc) - np.pi/2
            self.lidar_data = [float(j) if not np.isinf(float(j)) else self.range_max for j in msg.ranges]
            self.lidar_data = self.lidar_data[self.ind_min:self.ind_max]
            self.lidar_data_queue = self.lidar_data
            self.run_once = False
        else:
            self.lidar_data = [float(j) if not np.isinf(float(j)) else self.range_max for j in msg.ranges]
            self.lidar_data = self.lidar_data[self.ind_min:self.ind_max]
            self.check_for_collision()


    def animate(self, i):
        if self.run_once:
            time.sleep(2)

        x = self.lidar_data * np.cos(self.theta)
        y = self.lidar_data * np.sin(self.theta)

        # # If cabin present --> To take cabin area out-of-plot
        # x_mask = np.where((x >= 0.15) | (x <= -0.15))
        # y_mask = np.where((y >= 0.1) | (y <= -0.15))
        # mask = np.union1d(x_mask, y_mask)
        # x = x[mask]
        # y = y[mask]

        self.ax.clear()

        self.ax.scatter(x, y, c='r', s=1, label='LiDAR detections')
        self.ax.plot(0.0, 0.0, 'bo', markersize=5, label='LiDAR location')
        self.ax.plot([self.range_max*np.cos(self.angle_max_view - np.pi/2), 0, self.range_max*np.cos(self.angle_min_view - np.pi/2)],
                     [self.range_max*np.sin(self.angle_max_view - np.pi/2), 0, self.range_max*np.sin(self.angle_min_view - np.pi/2)],
                     linewidth=1, linestyle='dashed')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_title('LIVE LiDAR PLOT - TruckLab', fontsize=20)
        self.ax.set_xlabel('X - axis (m)', fontsize=18)
        self.ax.set_ylabel('Y - axis (m)', fontsize=16)
        self.ax.legend()
        self.ax.set_aspect(1.0)
        # self.fig.set_size_inches(8, 8)

        if self.collision_alert:
            self.ax.text(-2, -4, 'Collision ALERT !!!', style='italic', fontsize=20, color="red")

    def loop(self):
        rospy.logwarn("Starting to read and plot...")
        rospy.Subscriber(self.topic_name, LaserScan, self.listen)

        # Plotting real-time
        ani = FuncAnimation(self.fig, self.animate, interval=self.animate_rate*1000)
        plt.show()

        rospy.spin()


def main():
    rospy.init_node('LIDAR_monitor')
    lidar_1_topic = "/tractor2/scan"
    lidar_1_sub = lidar_subscriber(topic_name=lidar_1_topic)
    lidar_1_sub.loop()
    rospy.logwarn("Finished reading!")

if __name__ == '__main__':
    main()

