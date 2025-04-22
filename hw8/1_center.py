import numpy as np

def rectilinear_1_center(points):
    """
    给定二维点集合，返回 rectilinear 1-center 的坐标（最小曼哈顿最大距离的点）
    :param points: (n, 2) 的 numpy 数组，表示 n 个点的 x, y 坐标
    :return: (q_x, q_y)
    """
    points = np.array(points)

    # 四个方向的线性投影
    f1 = points[:, 0] + points[:, 1]  # x + y
    f2 = points[:, 0] - points[:, 1]  # x - y
    f3 = -points[:, 0] + points[:, 1]  # -x + y
    f4 = -points[:, 0] - points[:, 1]  # -x - y

    # 每个方向的最小值和最大值
    m1, M1 = f1.min(), f1.max()
    m2, M2 = f2.min(), f2.max()
    m3, M3 = f3.min(), f3.max()
    m4, M4 = f4.min(), f4.max()

    # 中心的线性投影值 = 中点
    t1 = (M1 + m1) / 2
    t2 = (M2 + m2) / 2

    # 解方程组：
    # x + y = t1
    # x - y = t2
    # 解出：
    q_x = (t1 + t2) / 2
    q_y = (t1 - t2) / 2

    return q_x, q_y

if __name__ =='__main__':
    points = [
        [1, 1],
        [-1, -1],
        [-1, 1],
        [1, -1]
    ]
    print(rectilinear_1_center(points))