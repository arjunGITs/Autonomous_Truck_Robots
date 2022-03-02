import cv2
import numpy as np

def draw_grid(img, pxstep=50, grid_size=1, line_color=(230, 230, 230), thickness=1, type_=cv2.LINE_AA):
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    position = (img.shape[0]-150, 25)
    cv2.putText(
        img,  # numpy array on which text is written
        f"Grid size = {grid_size} m",  # text
        position,  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        0.5,  # font size
        line_color)  # font color

    return img

def create_map_img(width, depth, grid_pix, grid_size):
    im_size = (width, depth, 3)
    img = np.zeros(im_size, np.uint8) + 255
    img = draw_grid(img, grid_pix, grid_size)
    return img

if __name__ == "__main__":

    """
    User choices
    """
    choice = 1  # [0, 1] --> [boundary, object]

    file_name = [r"\map_boundary", r"\map_object"]
    file_path = r"map_formats" + file_name[choice] + ".jpg"

    """
    Map --> Boundary
    """
    map_widths = [7, 3]  # m
    map_depths = [7, 3]  # m
    grid_size = 1  # m

    map_width = map_widths[choice]
    map_depth = map_depths[choice]

    """
    Image Properties
    """
    img_width = 800    # pixels
    img_height = int(map_depth/map_width * img_width)

    map_to_img_ratio = [img_width/map_width, img_height/map_depth]
    grid_px = int(grid_size * map_to_img_ratio[0])
    print(grid_px)
    img = create_map_img(img_height, img_width, grid_px, grid_size)
    cv2.imwrite(file_path, img)
    cv2.imshow("Created map image", img)
    cv2.waitKey(0)