
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_img(path):
    img = cv2.imread(path)
    return img

def generate_mask_by_points(img_size, points):
    """

    :param img_size: H x W
    :param points: [[30, 30], [500, 500], [30, 500]]
    :return: mask
    """
    H, W, C = img_size
    img = np.zeros((H, W, C))

    points = np.array(points, np.int32)

    # fill
    cv2.fillPoly(img, [points], 1)

    # reconstruct
    tmp = np.array(img, np.uint8)[:, :, 0]
    for i in range(C):
        img[:, :, i] = tmp

    return np.array(img, np.uint8)

if __name__=="__main__":

    # read img
    path = "./imgs/1.tif"
    img = read_img(path)

    # generate mask
    points = [[30, 30], [300, 500], [5000, 5000], [ 2500, 3500], [30, 2500] ]
    img_size = np.shape(img)
    mask = generate_mask_by_points(img_size, points=points)


    # fill
    img = cv2.multiply(img, mask)

    # show
    cv2.imshow("img", img)
    cv2.waitKey()