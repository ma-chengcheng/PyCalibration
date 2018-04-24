# ** coding: utf-8**
import os
import cv2
import numpy as np
import glob
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)

original_img_dir = os.path.join(os.getcwd(), 'data')
processed_img_dir = os.path.join(os.getcwd(), 'data2')

object_points = np.zeros((6 * 7, 3), np.float32)  # 世界坐标系点
object_points[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y


def image_calibration():
    """读取标定图片"""
    img_count = 0
    images = glob.glob("{0}/*.jpg".format(original_img_dir))

    objp = np.zeros((6*7, 3), np.float32)

    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 景物点坐标
    obj_points = []

    # 图像点坐标
    img_points = []

    cv2.namedWindow('original_img', cv2.WINDOW_NORMAL)

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (6, 7))

        if ret is True:
            obj_points.append(objp)

            # 进一步提取亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            if corners2 is not None:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            gray = cv2.drawChessboardCorners(gray, (6, 7), corners, patternWasFound=False)

        cv2.imshow('original_img', gray)

        img_count += 1
        img_name = 'processed_img_{}.jpg'.format(img_count)
        cv2.imwrite(os.path.join(processed_img_dir, img_name), gray)

        cv2.waitKey(100)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    return mtx, dist, rvecs, tvecs


def calibrate(img):
    """
    图像进行标定
    参数：图像
    返回：内参矩阵、畸变系数、旋转向量与平移向量
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)      # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001

    obj_points = []                                                                 # 景物点坐标
    img_points = []                                                                 # 图像点坐标

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                    # 转为灰度图

    size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)                    # 提取角点

    if ret is True:
        obj_points.append(object_points)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)      # 进一步提取亚像素角点

        if corners2 is not None:
            corners = corners2

        img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

        return corners, ret, mtx, dist, rvecs, tvecs
    else:
        return False


def operate(frame):

    img_count = 0                                                                   # 照片张数
    key = cv2.waitKey(1)

    if key == ord('q'):
        return False
    elif key == ord('g'):
        img_count += 1
        img_name = 'original_img_{}.jpg'.format(img_count)
        cv2.imwrite(os.path.join(original_img_dir, img_name), frame)
        logging.debug("拍{0}张".format(img_count))


def draw_point(img, corners):
    """ 绘制内角 """

    for point in corners:
        point = tuple(point[0])
        cv2.circle(img=img, center=point, radius=3, color=(0, 0, 255), thickness=-1)

    return img


def draw_axis(img, rvecs, tvecs, mtx, dist, corners):

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def get_image():

    cv2.namedWindow('cap', cv2.WINDOW_NORMAL)                                       # 创建窗体

    cap = cv2.VideoCapture(1)                                                       # 打开相机

    while True:

        ret, frame = cap.read()                                                     # 获取视频流q

        frame = cv2.resize(frame, (400, 300))                                         # 图像缩放至400*300

        if calibrate(frame) is not False:
            corners, ret, mtx, dist, rvecs, tvecs = calibrate(frame)

            frame = draw_point(frame, corners)

            frame = draw_axis(frame, rvecs, tvecs, mtx, dist, corners)

        if operate(frame) is False:
            break

        cv2.imshow("cap", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # get_image()
    mtx, dist, rvecs, tvecs = image_calibration()

    # 内参数矩阵
    print("内参数矩阵：\n {0}".format(mtx.reshape(3, 3)))

    # 畸变系数
    print("畸变系数：\n {0}".format(dist))

    # 旋转向量
    print("旋转向量：\n {0}".format(np.array(rvecs).reshape(-1, 3)))

    # 平移向量
    print("平移向量：\n {0:10}".format(np.array(tvecs).reshape(-1, 3)))

