import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt

poly = Polygon([[1, 0], [0, 3], [5, 6], [6, 5]])

poly_mapped = mapping(poly)
poly_coordinates = poly_mapped["coordinates"][0]
x_vals = [row[0] for row in poly_coordinates]
y_vals = [row[1] for row in poly_coordinates]
min_x = min(x_vals)
min_y = min(y_vals)
max_x = max(x_vals)
max_y = max(y_vals)
print([min_x, min_y])
print([max_x, max_y])

x1, y1 = poly.exterior.xy
plt.plot(x1, y1, c="blue")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# img = cv2.imread('map_for_scenes/map_trailer.jpg')
# rows, cols, _ = img.shape
#
# theta = 45 * np.pi/180
#
# M = np.float32([[np.cos(theta), -np.sin(theta), 100], [np.sin(theta), np.cos(theta), 50]])
# dst = cv2.warpAffine(img, M, (cols+300, rows+300))
#
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print(np.linspace(0, 2 * np.pi, 10, False))

# theta = -20 * np.pi/180
# print(np.cos(theta))