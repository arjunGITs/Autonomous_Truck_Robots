#!/usr/bin/env python3

from __future__ import print_function
import rospy
import numpy as np
import cv2
from camera_reader_pkg.srv import image_capture_service, image_capture_serviceResponse
from sensor_msgs.msg import CompressedImage


class Camera_Service(object):
    def __init__(self, topic_name):
        self.image = None
        self.img_count = 0
        self.topic_name = topic_name
        self.wait = True

    def listen_once(self, msg):
        img_arr = np.fromstring(msg.data, np.uint8)
        self.image = cv2.imdecode(img_arr, 1)
        # cv2.imwrite(f"/home/emeinder/catkin_ws/src/camera_reader_pkg/src/scripts_arjun/saved_images/{self.img_count:06}.jpg", self.image)
        cv2.imshow("Video Captured", self.image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        self.image = []
        self.sub.unregister()
        self.wait = False


    def callback(self, request):
        self.sub = rospy.Subscriber(self.topic_name, CompressedImage, self.listen_once)
        while True:
            if not self.wait:
                self.img_count += 1
                print("Image captured: ", self.img_count)
                return image_capture_serviceResponse(f"Image: {self.img_count} captured")


    def loop(self):
        rospy.init_node('image_capture_server')
        s = rospy.Service('capture_image', image_capture_service, self.callback)
        print("Ready to capture image...")
        rospy.spin()


if __name__ == "__main__":
    camera_1_topic = "/raspicam_node/image/compressed"
    camera_1_service = Camera_Service(topic_name=camera_1_topic)
    camera_1_service.loop()