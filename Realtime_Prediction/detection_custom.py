
# This code has been forked from [https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3]

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, Load_Yolo_model
from yolov3.configs import *


image_path   = r"/home/emeinder/TRUCKLAB_ARJUN/Codes/Realtime_prediction/Realtime_data/1.jpg"
yolo = Load_Yolo_model()
count = 1

while True:
    t1 = time.perf_counter()
    detect_image(yolo, image_path, "/home/emeinder/TRUCKLAB_ARJUN/Codes/Realtime_prediction/Realtime_data/1_detected.jpg",
                 input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    t2 = time.perf_counter()
    print("taken time: ", t2-t1)

    print(count); count+= 1
    time.sleep(4)



