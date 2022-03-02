import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import random, randint
import shapely
from shapely.geometry import Polygon, Point, mapping

def generate_random_lidar_point(boundary_dict, polygon_list):
    bool_within_boundary = False
    bool_object_intersect = True
    polygon_boundary = Polygon(boundary_dict["polygon"])
    while not bool_within_boundary or bool_object_intersect:
        xc = randint(0, boundary_dict["image_size"][0])
        yc = randint(0, boundary_dict["image_size"][1])
        theta = 0.0 + random() * 360.0
        pt = Point(xc, yc)
        bool_within_boundary = polygon_boundary.contains(pt)
        if bool_within_boundary:
            bool_object_intersect = False
            for i in range(len(polygon_list)):
                bool_object_intersect = polygon_list[i].contains(pt)
                if bool_object_intersect:
                    break
    return xc, yc, theta


def lidar_sense_scene(img_scene, pt_lidar, lidar_max_range, pose_angle, angle_lim_minus, angle_lim_plus, angle_inc, map_img_scale, pred_img_scale, max_error, clr=(0, 0, 255)):
    lidar_points = []
    scene_lidar_points = []
    img_scene[img_scene >= 200] = 255
    img_scene[img_scene < 200] = 0
    img_scene_lidar = img_scene.copy()
    img_scene_lidar = cv2.circle(img_scene_lidar, (pt_lidar[0], pt_lidar[1]), radius=8, color=(255, 0, 0), thickness=-1)
    x_lim = img_scene.shape[0]
    y_lim = img_scene.shape[1]
    max_range = min(int((x_lim + y_lim) / 2), lidar_max_range)
    size_pred_img = (416, 416, 3)
    img_ego_lidar = np.zeros(size_pred_img, dtype='uint8')
    for angle in np.linspace((pose_angle - angle_lim_minus), (pose_angle + angle_lim_plus), int((angle_lim_plus+angle_lim_minus)/angle_inc), False):
        x_max, y_max = (pt_lidar[0] + max_range * np.cos(angle), pt_lidar[1] + max_range * np.sin(angle))
        for i in range(1000):
            breaker = False
            u = i / 1000
            x_check = int(x_max * u + pt_lidar[0] * (1 - u))
            y_check = int(y_max * u + pt_lidar[1] * (1 - u))
            if 0 < x_check < x_lim and 0 < y_check < y_lim:
                color = img_scene[y_check, x_check, :]
                if (color[0], color[1], color[2]) == (0, 0, 0):
                    error = np.random.uniform(-max_error, max_error) * map_img_scale[0]
                    scene_lidar_points.append([x_check, y_check])
                    lidar_point = [int(x_check - pt_lidar[0] + error*np.cos(angle)),
                                         int(y_check - pt_lidar[1] + error*np.sin(angle))]
                    lidar_points.append(lidar_point)
                    img_scene_lidar = cv2.circle(img_scene_lidar, (int(x_check + error*np.cos(angle)), int(y_check + error*np.sin(angle))), radius=3, color=clr, thickness=-1)
                    # img_scene_lidar[y_check, x_check] = clr
                    breaker = True
                    break
            if breaker:
                break

    img_scene_lidar = cv2.circle(img_scene_lidar, (pt_lidar[0], pt_lidar[1]), radius=int(max_range), color=(0, 255, 0), thickness=2)
    img_scene_lidar = cv2.line(img_scene_lidar, (pt_lidar[0], pt_lidar[1]),
                               (int(pt_lidar[0]+(np.cos(pose_angle-angle_lim_minus))*max_range),
                                int(pt_lidar[1]+(np.sin(pose_angle-angle_lim_minus))*max_range)),
                               (255, 0, 0), 3)
    img_scene_lidar = cv2.line(img_scene_lidar, (pt_lidar[0], pt_lidar[1]),
                               (int(pt_lidar[0] + (np.cos(pose_angle + angle_lim_plus))*max_range),
                                int(pt_lidar[1] + (np.sin(pose_angle + angle_lim_plus))*max_range)),
                               (255, 0, 0), 3)

    theta = -pose_angle - np.pi/2
    for pt in lidar_points:
        x = pt[0] * (pred_img_scale[0] / map_img_scale[0])
        y = pt[1] * (pred_img_scale[1] / map_img_scale[1])

        pt[0] = int(x * np.cos(theta) - y * np.sin(theta) + size_pred_img[0] / 2)
        pt[1] = int(x * np.sin(theta) + y * np.cos(theta) + size_pred_img[1] / 2)

        img_ego_lidar[pt[1], pt[0], :] = (0, 0, 255)

    return img_scene_lidar, img_ego_lidar, scene_lidar_points


def generate_random_position(boundary_dict, object_dict, polygon_list):
    bool_within_boundary = False
    bool_object_intersect = True
    while not bool_within_boundary or bool_object_intersect:
        polygon_boundary = Polygon(boundary_dict["polygon"])
        polygon_current_object = Polygon(object_dict["polygon"])

        xc = randint(0, boundary_dict["image_size"][0])
        yc = randint(0, boundary_dict["image_size"][1])
        theta = 0.0 + random() * 360.0
        pose = [xc, yc, theta]

        rotated_polygon = shapely.affinity.rotate(polygon_current_object, theta, origin='center', use_radians=False)
        transformed_polygon = shapely.affinity.translate(rotated_polygon, xoff=xc - object_dict["center_coords"][0], yoff=yc - object_dict["center_coords"][1])


        bool_within_boundary = polygon_boundary.contains(transformed_polygon)

        if bool_within_boundary:
            bool_object_intersect = False
            for i in range(len(polygon_list)):
                bool_object_intersect = polygon_list[i].intersects(transformed_polygon)
                if bool_object_intersect:
                    break


    polygon_detail = transformed_polygon

    x1, y1 = polygon_boundary.exterior.xy
    x2, y2 = transformed_polygon.exterior.xy
    plt.plot(x1, y1, c="blue")
    plt.plot(x2, y2, c="red")
    plt.gca().set_aspect('equal', adjustable='box')

    return pose, polygon_detail



if __name__ == "__main__":

    # User choices --> Training data generation
    total_num_images = 2
    total_objects = 4

    # File path details
    data_directory = r"D:\Local_PDEng_ASD\TruckLab_LOCAL\codes_LOCAL\TruckLab_Simulation\GENERATED_DATA"
    # save_as_directory = r"D:\Local_PDEng_ASD\TruckLab_LOCAL\codes_LOCAL\CODES_TruckLab\TensorFlow-2.x-YOLOv3\trucklab_synthetic_data"
    save_as_directory = r"D:\Local_PDEng_ASD\TruckLab_LOCAL\codes_LOCAL\TruckLab_Simulation\GENERATED_DATA"

    data_type = r"\Dataset_train"
    # data_type = r"\Dataset_test"


    # Map and Prediction image properties
    map_img_size = [800, 800, 3]
    map_width = 7  # m
    map_depth = 7  # m
    map_img_scale_x = map_img_size[0] / map_width
    map_img_scale_y = map_img_size[1] / map_depth
    map_img_scale = [map_img_scale_x, map_img_scale_y]


    pred_img_size = [416, 416, 3]
    pred_map_width = 3.5 *2  # m
    pred_map_depth = 3.5 *2  # m
    pred_img_scale_x = pred_img_size[0] / pred_map_width
    pred_img_scale_y = pred_img_size[1] / pred_map_depth
    pred_img_scale = [pred_img_scale_x, pred_img_scale_y]

    # LiDAR properties
    lidar_max_range = 3.5 * map_img_scale[0]  # m
    angle_lim_minus = 180 * np.pi / 180
    angle_lim_plus = 180 * np.pi / 180
    angle_inc = 0.017501922
    max_error = 0.02  # m

    # Boundary --> Initialization
    img_boundary_path = r"map_for_scenes/map_boundary.jpg"
    details_boundary_path = r"map_for_scenes/map_details_boundary.csv"
    img_boundary = cv2.imread(img_boundary_path)
    img_boundary = cv2.cvtColor(img_boundary, cv2.COLOR_BGR2RGB)
    with open(details_boundary_path, 'rb') as f:
        data = f.read()
    details_boundary = pickle.loads(data)

    details_objects = []
    img_objects = []

    # Objects --> Initialization

    # Tractor
    img_object_path = r"map_for_scenes/map_tractor.jpg"
    details_object_path = r"map_for_scenes/map_details_tractor.csv"
    img_object = cv2.imread(img_object_path)
    img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
    img_objects.append(img_object)
    with open(details_object_path, 'rb') as f:
        data = f.read()
    details_objects.append(pickle.loads(data))

    # Trailer
    img_object_path = r"map_for_scenes/map_trailer.jpg"
    details_object_path = r"map_for_scenes/map_details_trailer.csv"
    img_object = cv2.imread(img_object_path)
    img_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
    img_objects.append(img_object)
    with open(details_object_path, 'rb') as f:
        data = f.read()
    details_objects.append(pickle.loads(data))

    total_labels = []
    labels = ""

    for img_num in range(1, total_num_images + 1):
        print(img_num)
        # Fixing image paths
        img_path = data_directory + data_type + f"\{img_num:06}" + ".jpg"
        img_path_saveas = save_as_directory + data_type + f"\{img_num:06}" + ".jpg"

        polygon_list = []
        object_num_list = []
        img_scene = img_boundary.copy()
        num_objects_in_scene = randint(0, total_objects)


        for i in range(num_objects_in_scene):
            object_num = randint(0, 1)
            pose, polygon_detail = generate_random_position(details_boundary, details_objects[object_num], polygon_list)
            polygon_list.append(polygon_detail)
            object_num_list.append(object_num)

            M = np.float32([[1, 0, pose[0]-details_objects[object_num]["center_coords"][0]], [0, 1, pose[1]-details_objects[object_num]["center_coords"][1]]])
            img_shifted = cv2.warpAffine(img_objects[object_num].copy(), M, (img_boundary.shape[1], img_boundary.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))
            center_of_rot = (pose[0], pose[1])
            M = cv2.getRotationMatrix2D(center_of_rot, -pose[2], 1.0)
            img_transformed = cv2.warpAffine(img_shifted, M, (img_boundary.shape[1], img_boundary.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))

            img_scene[img_transformed <= 200] = 0


        pose_lidar = generate_random_lidar_point(details_boundary, polygon_list)
        pt_lidar = pose_lidar[0], pose_lidar[1]
        pose_angle = pose_lidar[2] * np.pi/180
        img_scene_lidar, img_ego_lidar, scene_lidar_points = lidar_sense_scene(img_scene, pt_lidar, lidar_max_range, pose_angle,
                                                           angle_lim_minus, angle_lim_plus, angle_inc,
                                                           map_img_scale, pred_img_scale, max_error, clr=(0, 0, 255))

        polygon_activation_indices = []
        for pnt in scene_lidar_points:
            pt = Point(pnt[0], pnt[1])
            for i in range(len(polygon_list)):
                if polygon_list[i].contains(pt):
                    polygon_activation_indices.append(i)
        polygon_activation_indices = list(set(polygon_activation_indices))
        print(polygon_activation_indices)


        labels_str = str(img_path_saveas) + str(" ")
        for i in polygon_activation_indices:
            poly_mapped = mapping(polygon_list[i])
            poly_coordinates = poly_mapped["coordinates"][0]

            x_vals = [row[0] for row in poly_coordinates]
            y_vals = [row[1] for row in poly_coordinates]

            x_scaled = [(val - pose_lidar[0]) * (pred_img_scale[0] / map_img_scale[0]) for val in x_vals]
            y_scaled = [(val - pose_lidar[1]) * (pred_img_scale[1] / map_img_scale[1]) for val in y_vals]

            x_vals_lidar = []
            y_vals_lidar = []
            theta = -pose_angle - np.pi / 2
            for j in range(len(x_scaled)):
                x_vals_lidar.append(x_scaled[j] * np.cos(theta) - y_scaled[j] * np.sin(theta) + pred_img_size[0] / 2)
                y_vals_lidar.append(x_scaled[j] * np.sin(theta) + y_scaled[j] * np.cos(theta) + pred_img_size[1] / 2)

            min_x = int(min(x_vals_lidar))
            min_y = int(min(y_vals_lidar))
            max_x = int(max(x_vals_lidar))
            max_y = int(max(y_vals_lidar))

            label = [min_x, min_y, max_x, max_y, object_num_list[i]]
            labels = labels + str(label).strip("[]").replace(" ", "") + str(" ")

        labels_str = labels_str + labels
        total_labels.append(labels_str)
        labels = ""

        cv2.imwrite(img_path, img_ego_lidar)



        cv2.imshow("image", img_scene)
        cv2.imwrite("scenes_generated/image.jpg", img_scene)
        # plt.show()
        # cv2.waitKey(0)

        cv2.imshow("lidar points", img_scene_lidar)
        cv2.imshow("Lidar ego", img_ego_lidar)
        cv2.waitKey(0)


    with open(data_directory + data_type + ".txt", 'w') as f:
        for line in total_labels:
            f.write(line+'\n')