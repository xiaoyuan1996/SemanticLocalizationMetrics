import numpy as np
import skimage
from skimage.feature import peak_local_max
import cv2
from scipy import ndimage
from scipy.ndimage import maximum_filter


a = np.array([])
print(np.mean(a))
exit()
#
# tmp = np.array([[3, 3, 0, 0, 3],
#        [0, 0, 2, 1, 3],
#        [0, 1, 1, 1, 2],
#        [3, 2, 1, 2, 0],
#        [2, 2, 1, 2, 1]])
#
# mx = maximum_filter(tmp, size=3)
#
# tmp = np.where(mx == tmp, tmp, 0)
#
# print(tmp)
# coordinates = peak_local_max(tmp)
# print(coordinates)
# exit(0)

prob_path = "../predict/cache/probmap_1.jpg"
src_img = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)

gray_img = src_img
for i in range(10):
    gray_img = cv2.medianBlur(gray_img, 251)


mx = maximum_filter(gray_img, size=1000)
gray_img = np.where(mx == gray_img, gray_img, 0)
# gray_img = np.where(mx == gray_img)
gray_img = np.asarray(gray_img > 0, np.uint8) * 255

# print(gray_img)
#
labels = skimage.measure.label(gray_img,connectivity=2)
tmp = skimage.measure.regionprops(labels)
# print(tmp.area)
for prop in tmp:
    print(prop.centroid)
    print(prop.area)

    center = tuple([int(i) for i in prop.centroid][::-1])
    print(center)
    cv2.circle(src_img, center, 3, 1, 4)


# for i in coordinates:
#     cv2.circle(src_img, tuple(i), 3, 1, 4)


cv2.imshow("img", src_img)
cv2.waitKey()




mx = maximum_filter(gray_img, size=150)
tmp = np.where(mx == gray_img, gray_img, 0)

cv2.imshow("img", tmp)
cv2.waitKey()
exit(0)



pos_x, pos_y = np.where(gray_img == 255)
print(len(pos_x))
print(gray_img[1024, 2174])
exit(0)

print(np.max(gray_img))
print(np.shape(gray_img))

unique,count=np.unique(gray_img,return_counts=True)
data_count=dict(zip(unique,count))
print(data_count)

exit(0)

for i in range(10):
    gray_img = cv2.medianBlur(gray_img, 251)


# thresh = cv2.threshold(gray_img, 0, 255, 150)[1]
print(cv2.THRESH_BINARY_INV)
print(cv2.THRESH_OTSU)
thresh = cv2.threshold(gray_img, 0, 255,cv2.THRESH_OTSU )[1]

print(thresh)

cv2.imshow("img", thresh)
cv2.waitKey()


D = ndimage.distance_transform_edt(thresh)
coordinates = peak_local_max(D, min_distance=200,labels=thresh)

print(len(coordinates))
print(coordinates)

for i in coordinates:
    cv2.circle(gray_img, tuple(i), 3, 1, 4)

cv2.imshow("img", gray_img)
cv2.waitKey()
