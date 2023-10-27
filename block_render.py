import cv2
import numpy as np
import pickle
import random
from utils.ply import read_ply

img = cv2.imread('/home/pz/Pictures/isprs_test_color.jpg')

KD2Tree_file = '/home/pz/Desktop/KPConv-PyTorch/Data/isprs/input_0.400/Vaihingen3D_test_KDTree.pkl'
with open(KD2Tree_file, 'rb') as f:
    search_2dtree = pickle.load(f)
points = search_2dtree.data


x0 = 14
y0 = 15


stride = 30.0 * 0.5

xmin = np.min(points[:, 0])
xmax = np.max(points[:, 0])
ymin = np.min(points[:, 1])
ymax = np.max(points[:, 1])

xmov = (30.0 * 2 - (xmax - xmin) % int(30.0 * 2)) * 0.5
ymov = (30.0 * 2 - (ymax - ymin) % int(30.0 * 2)) * 0.5

xstart = np.floor(xmin - xmov) + stride
xend = xmax
ystart = np.floor(ymin - ymov) + stride
yend = ymax

cind = []
center_pts = []

while xstart < xend:
    img_j = x0 + int(1.65 * (xstart - xmin))
    while ystart < yend:

        cent_p = [[xstart, ystart]]

        if len(search_2dtree.query_radius(cent_p,
                                          r=stride)[0]) > 0:

            img_i = y0 + int(1.65*(ymax-ystart))
            cv2.circle(img, (img_j, img_i), 50, (0, 0, 0), 2)

        ystart += 30.0
    ystart = np.floor(ymin - ymov) + stride
    xstart += 30.0




# x1s = range(100, 300)
# x1 = random.sample(x1s, k=4)
# y1s = range(80,200)
# y1 = random.sample(y1s, k=4)
#
# for x, y in zip(x1, y1):
#     cv2.circle(img, (x, y), 50, (0,0,0), 2)
#
# x1s = range(100, 600)
# x1 = random.sample(x1s, k=12)
# y1s = range(250,450)
# y1 = random.sample(y1s, k=12)
#
# for x, y in zip(x1, y1):
#     cv2.circle(img, (x, y), 50, (0,0,0), 2)
#
# x1s = range(400, 600)
# x1 = random.sample(x1s, k=4)
# y1s = range(480,650)
# y1 = random.sample(y1s, k=4)
#
# for x, y in zip(x1, y1):
#     cv2.circle(img, (x, y), 50, (0,0,0), 2)

cv2.imwrite('/home/pz/Pictures/isprs_test_block.jpg', img)

cv2.imshow('test.jpg',img)
cv2.waitKey(0)
cv2.destroyWindow()

