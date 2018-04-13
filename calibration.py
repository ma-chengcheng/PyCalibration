# ** coding: utf-8**
import cv2
import numpy as np

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
print(objp)

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
# while True:
#     # 获取视频流
#     ret, frame = cap.read()
#     # 展示视频流
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 提取角点
#     ret, corners = cv2.findChessboardCorners(frame, (7, 6), None)
#
#     if ret is True:
#         # 进一步提取亚像素角点
#         corners2 = cv2.cornerSubPix(frame, corners, (5, 5), (-1, -1), criteria)
#
#         # 绘制内角点
#         frame = cv2.drawChessboardCorners(frame, (7, 6), corners, patternWasFound=False)
#
#         print(corners)
#
#     cv2.imshow("cap", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # 标定
# ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera()
#
# # 内参数矩阵
# print("内参数矩阵：", mtx)
#
# # 畸变系数
# print("畸变系数：", dist_coeffs)
#
# # 旋转向量
# print("旋转向量：", rvecs)
#
# # 平移向量
# print("平移向量：", tvecs)

