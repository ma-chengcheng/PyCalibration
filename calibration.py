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

        cv2.waitKey(500)

    ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    return mtx, dist_coeffs, rvecs, tvecs


def get_image():
    # 创建窗体
    cv2.namedWindow('cap', cv2.WINDOW_NORMAL)

    # 打开相机
    cap = cv2.VideoCapture(0)

    # 照片张数
    img_count = 0

    while True:

        # 获取视频流
        ret, frame = cap.read()

        # 展示视频流
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 提取角点
        ret, corners = cv2.findChessboardCorners(frame, (7, 6), None)

        if ret is True:
            # 绘制内角点
            frame = cv2.drawChessboardCorners(frame, (7, 6), corners, patternWasFound=False)

        cv2.imshow("cap", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('g'):
            img_count += 1
            img_name = 'original_img_{}.jpg'.format(img_count)
            cv2.imwrite(os.path.join(original_img_dir, img_name), frame)
            logging.debug("拍{0}张".format(img_count))
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mtx, dist_coeffs, rvecs, tvecs = image_calibration()

    # 内参数矩阵
    print("内参数矩阵：\n {0}".format(mtx))

    # 畸变系数
    print("畸变系数：\n {0}".format(dist_coeffs))

    # 旋转向量
    print("旋转向量：\n {0}".format(rvecs))

    # 平移向量
    print("平移向量：\n {0}".format(tvecs))

