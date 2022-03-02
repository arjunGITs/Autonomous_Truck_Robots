#!/usr/bin/env python3

'''
Lidar Subscriber python code
Author: Arjun Pradeep
'''


import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class camera_subscriber(object):
    def __init__(self, topic_name):
        # Params
        self.image = None
        self.img_count = 1
        self.topic_name = topic_name

    def read_and_write(self, msg):
        # rospy.loginfo('Image received...')
        img_arr = np.fromstring(msg.data, np.uint8)
        self.image = cv2.imdecode(img_arr, 1)
        # cv2.imwrite(f"/home/emeinder/catkin_ws/src/camera_reader_pkg/src/scripts_arjun/saved_images/{self.img_count:06}.jpg", self.image)
        cv2.imshow("Video Captured", self.image)
        cv2.waitKey(10)
        self.img_count += 1
        self.image = []
        # rospy.loginfo('Image saved as ...')

    def loop(self):
        rospy.logwarn("Starting to read...")
        rospy.Subscriber(self.topic_name, CompressedImage, self.read_and_write)
        rospy.spin()


def main():
    rospy.init_node("CAMERA_monitor")
    # camera_1_topic = "/camera/rgb/image_raw"
    camera_1_topic = "/raspicam_node/image/compressed"
    camera_1_sub = camera_subscriber(topic_name=camera_1_topic)
    camera_1_sub.loop()

if __name__ == '__main__':
    main()

