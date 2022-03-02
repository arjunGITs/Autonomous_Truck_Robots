#!/usr/bin/env python3

'''
Lidar Subscriber python code
Author: Arjun Pradeep
'''


import rospy
import numpy as np
import pandas as pd
import cv2
from sensor_msgs.msg import LaserScan
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *



class lidar_subscriber(object):
    def __init__(self, topic_name, rate):
        self.topic_name = topic_name
        self.rate = rate

        # Region of interest
        self.angle_min_view = 120 * np.pi / 180
        self.angle_max_view = 240 * np.pi / 180

        self.topic_name = topic_name
        self.lidar_data = []
        self.theta = []
        self.run_once = True

        self.x_map_lim_max = 4.0
        self.y_map_lim_max = 4.0
        self.x_map_lim_min = -4.0
        self.y_map_lim_min = -4.0

        # Image properties
        self.im_size_w = 416
        self.aspect_ratio = (self.y_map_lim_max - self.y_map_lim_min) / (self.x_map_lim_max - self.x_map_lim_min)
        self.im_size_h = int(self.im_size_w * self.aspect_ratio)
        self.size = self.im_size_h, self.im_size_w, 3
        self.im = []

        # Map to image coordinates
        self.x_resolution = (self.x_map_lim_max - self.x_map_lim_min) / self.im_size_w
        self.y_resolution = (self.y_map_lim_max - self.y_map_lim_min) / self.im_size_h

        # Technique to subscribe at a given rate
        self.processing = False
        self.new_msg = False

        self.yolo = Load_Yolo_model()

        self.count = 1



    def listen(self, msg):
        if not self.processing:
            self.new_msg = True
            if self.run_once == True:
                self.angle_inc = msg.angle_increment
                self.angle_min = msg.angle_min
                self.angle_max = msg.angle_max
                self.time_inc = msg.time_increment
                self.range_min = msg.range_min
                self.range_max = msg.range_max
                self.ind_min = int((self.angle_min_view - self.angle_min) // self.angle_inc)
                self.ind_max = int((self.angle_max_view - self.angle_min) // self.angle_inc)
                self.theta = np.arange(self.angle_min_view, self.angle_max_view - self.angle_inc,
                                       self.angle_inc) - np.pi / 2
                self.lidar_data = [float(j) if not np.isinf(float(j)) else self.range_max for j in msg.ranges]
                self.lidar_data = self.lidar_data[self.ind_min:self.ind_max]
                self.run_once = False
            else:
                self.lidar_data = [float(j) if not np.isinf(float(j)) else self.range_max for j in msg.ranges]
                self.lidar_data = self.lidar_data[self.ind_min:self.ind_max]


    def view_shape(self):
        x = self.lidar_data * np.cos(self.theta)
        y = self.lidar_data * np.sin(self.theta)

        # Image
        self.im = np.zeros(self.size, dtype=np.uint8)

        X = (x - self.x_map_lim_min) / self.x_resolution
        Y = (self.y_map_lim_max - y) / self.y_resolution


        for i in range(len(X)):
            if int(Y[i]) <= 416 and int(X[i]) <= 416:
                self.im[int(Y[i]), int(X[i])] = (0, 0, 255)

        return self.im


    def loop(self):
        rospy.logwarn("Starting to read...")
        rospy.Subscriber(self.topic_name, LaserScan, self.listen)
        while not rospy.is_shutdown():
            try:
                if self.new_msg:
                    self.processing = True
                    self.new_msg = False
                    rospy.logwarn("Reading...")
                    self.im = self.view_shape()
                    # cv2.imwrite(f"/home/emeinder/catkin_ws/src/lidar_reader_pkg/src/scripts_arjun/captured_images/{self.count}.jpg", self.im)
                    im_detected = detect_image(self.yolo, self.im,
                                 input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES,
                                 rectangle_colors=(255, 0, 0))
                    # cv2.imwrite(
                    #     f"/home/emeinder/catkin_ws/src/lidar_reader_pkg/src/scripts_arjun/captured_images/{self.count}_detected.jpg",
                    #     im_detected)
                    cv2.imshow("Raw image", self.im)
                    cv2.imshow("Detections", im_detected)
                    cv2.waitKey(10)

                    self.count += 1
                    rospy.Rate(self.rate).sleep()
                    self.processing = False
            except rospy.ROSInterruptException:
                rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")


def main():
    rospy.init_node('LIDAR_monitor')
    lidar_1_topic = "/tractor2/scan"
    lidar_1_rate = 0.5   # hz
    lidar_1_sub = lidar_subscriber(topic_name=lidar_1_topic, rate=lidar_1_rate)
    lidar_1_sub.loop()
    rospy.logwarn("Finished reading!")

if __name__ == '__main__':
    main()

