# ** coding: utf-8**
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

IMAGE_DIR = os.path.join(os.getcwd(), 'image')


class CalibrateException(Exception):
    pass


# 标定板信息
class ChessBoardInfo(object):

    def __init__(self, n_cols=0, n_rows=0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.size = (n_rows, n_cols)


def _get_corners(img, broad):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ok, corners) = cv2.findChessboardCorners(img, (broad.n_cols, broad.n_rows))
    if not ok:
        return ok, corners
    else:
        radius = 5
        cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        return ok, corners


class Calibrator(object):
    """
    使用findChessboardCorners找到角点

    :return (ok, corners)
    """

    def __init__(self, board):
        self._board = board
        self.size = None
        self.count = 0

    def get_corners(self, img):
        ok, corners = _get_corners(img, self._board)
        if ok:
            return ok, corners
        else:
            return False, None

    def collect_corners(self, images):

        corners = [self.get_corners(img) for img in images]

        good_corners = []

        for (ok, corner) in corners:
            if ok:
                self.count += 1
                good_corners.append(corner)

        if not good_corners:
            raise CalibrateException("No corners found in images!")
        else:

            self.size = (cv2.imread(images[0]).shape[0], cv2.imread(images[0]).shape[1])
            return good_corners

    def get_object_points(self):

        object_points = []
        if self.count is 0:
            raise CalibrateException("No images!")

        for i in range(self.count):
            object_point = np.zeros((self._board.n_cols * self._board.n_rows, 3), np.float32)
            object_point[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            object_points.append(object_point)
        return object_points

    def cal_param(self, img_points):
        """
        计算相机内外参
        :return: ok, mtx, dist, rvecs, tvecs
        """
        obj_points = self.get_object_points()

        ok, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            self.size,
            None,
            None
        )

        if not ok:
            return False, None, None, None, None
        else:
            return ok, mtx, dist, rvecs, tvecs

    def pose_estimation(self, mtx, dist, rvecs, tvecs):
        """
        cvProjectPoints2的函数计算反投影，返回经反投影后像素点的标
        :return: img_points
        """
        img_points = []
        for i in range(self.count):
            object_point = np.zeros((self._board.n_cols * self._board.n_rows, 3), np.float32)
            object_point[:, :2] = np.mgrid[0:self._board.n_cols, 0:self._board.n_rows].T.reshape(-1, 2)
            rvec = np.array(rvecs[i])
            tvec = np.array(tvecs[i])

            img_point, jac = cv2.projectPoints(object_point, rvec, tvec, mtx, dist)
            img_points.append(img_point)
        return img_points

    def draw_chart(self, project_points, img_points):
        """
        画出偏差图
        """
        x = np.array(range(self.count*self._board.n_rows*self._board.n_cols)) + 1
        y = project_points - img_points
        y_x = np.array(y.T[-1:]).flatten()
        y_y = np.array(y.T[:-1]).flatten()

        plt.ylim(-1.0, 1.0)                                                         # 设置纵轴上下限

        plt.scatter(x, y_x)
        plt.scatter(x, y_y)
        plt.title('x&y offset')

        plt.show()

    @ staticmethod
    def get_intrinsic(mtx):
        """
        获取内参
        """
        mtx = np.column_stack((mtx, np.zeros((3, 1))))
        return mtx

    @ staticmethod
    def get_extrinsics(rvecs, tvecs):
        """
        返回外参
        :param rvecs: 旋转向量组
        :param tvecs: 位移向量组
        :return: 外参数模型组
        """
        exteriores = []
        for rvec, tvec in zip(rvecs, tvecs):
            P = tvec
            R = cv2.Rodrigues(np.array(rvec))
            exterior = np.column_stack((np.array(R[0]), P))
            exterior = np.row_stack((exterior, np.zeros((1, 4))))
            exteriores.append(exterior)

        return exteriores


if __name__ == '__main__':
    board = ChessBoardInfo(n_cols=7, n_rows=6)
    calibrator = Calibrator(board)
    corners = calibrator.collect_corners(glob.glob('{0}/*.jpg'.format(IMAGE_DIR)))
    ok, mtx, dist, rvecs, tvecs = calibrator.cal_param(corners)

    # 内参矩阵
    intrinsic = calibrator.get_intrinsic(mtx)

    # 外参数矩阵
    extrinsics = calibrator.get_extrinsics(rvecs, tvecs)

    # 内参矩阵与外参矩阵的乘积
    M = np.matrix(intrinsic).dot(extrinsics[0])

    # 实际图像坐标
    camera_pose = np.matrix(np.array([1, 1, 0, 1])).T
    u, v, z = np.matrix(M).dot(camera_pose)
    print(u/z, v/z)

    print(corners[0].reshape((board.n_rows, board.n_cols, 2)))
