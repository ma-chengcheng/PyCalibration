# ** coding: utf-8**
import sys
import cv2
import copy
import numpy as np
from openni import openni2, nite2, utils
import logging
import glob

"""读取标定图片"""


def image_cal():
    objp = np.zeros((6*7, 3), np.float32)

    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    images = glob.glob("/home/tany/python study/PyCalibration/data/*.jpg")

    # 景物点坐标
    obj_points = []

    # 图像点坐标
    img_points = []

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    for image in images:
        img = cv2.imread(image)
        width, height, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape

        size = img.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (6, 7))

        if ret is True:
            obj_points.append(objp)
            gray = cv2.drawChessboardCorners(gray, (6, 7), corners, patternWasFound=False)

        cv2.imshow('img', gray)
        cv2.waitKey(1000)

        return obj_points, img_points, size

"""从相机读取"""

# cv2.namedWindow('cap', cv2.WINDOW_NORMAL)
#
# objp = np.zeros((6*7, 3), np.float32)
#
# objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
#
# # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # 景物点坐标
# obj_points = []
#
# # 图像点坐标
# img_points = []
#
#
# cap = cv2.VideoCapture(0)
#
# times = 1
#
# while True:
#
#     # if times == 5:
#     #     break
#
#     # 获取视频流
#     ret, frame = cap.read()
#     # 展示视频流
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame2 = copy.copy(frame)
#     size = frame.shape[::-1]
#
#     # 提取角点
#     ret, corners = cv2.findChessboardCorners(frame, (7, 6), None)
#
#     if ret is True:
#         obj_points.append(objp)
#
#         # 进一步提取亚像素角点
#         corners2 = cv2.cornerSubPix(frame, corners, (5, 5), (-1, -1), criteria)
#
#         if corners2 is not None:
#             img_points.append(corners2)
#         else:
#             img_points.append(corners)
#
#         # cv2.imwrite('{0}.jpg'.format(times), frame)
#         # print "拍{0}张".format(times)
#         # times += 1
#
#         # 绘制内角点
#         frame = cv2.drawChessboardCorners(frame, (7, 6), corners, patternWasFound=False)
#
#     cv2.imshow("cap", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     if cv2.waitKey(1) & 0xFF == ord(' '):
#         cv2.imwrite('{0}.jpg'.format(times), frame2)
#         print "拍{0}张".format(times)
#         times += 1
#
# cap.release()
# cv2.destroyAllWindows()
#
# 标定
ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(image_cal(), None, None)

print("img_points: \n {0}".format(img_points))


# 尺寸
print("尺寸: \n {0}".format(size))

# 内参数矩阵
print("内参数矩阵：\n {0}".format(mtx))

# 畸变系数
print("畸变系数：\n {0}".format(dist_coeffs))

# 旋转向量
print("旋转向量：\n {0}".format(rvecs))

# 平移向量
print("平移向量：\n {0}".format(tvecs))

"""openni2与opencv"""
# openni2.initialize()
#
# dev = openni2.Device.open_any()
#
# try:
#     userTracker = nite2.UserTracker(dev)
# except utils.NiteError as ne:
#     logger.error("Unable to start the NiTE human tracker. Check "
#                  "the error messages in the console. Model data "
#                  "(s.dat, h.dat...) might be inaccessible.")
#     sys.exit(-1)
#
# # stream_color = openni2.VideoStream(dev, openni2.SENSOR_COLOR)
# while True:
#     frame = userTracker.read_frame()
#     cv2.imread(frame)
#
# nite2.unload()
# openni2.unload()
