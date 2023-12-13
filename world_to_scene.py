# File      :world_to_scene.py
# Auther    :WooChi
# Time      :2023/03/15
# Version   :1.0
# Function  :世界坐标系到场景坐标系

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D


def fit_a_plane(x2, y2, z2):
    # 创建系数矩阵A
    A = np.zeros((3, 3))
    for i in range(len(x2)):
        A[0, 0] = A[0, 0] + x2[i] ** 2
        A[0, 1] = A[0, 1] + x2[i] * y2[i]
        A[0, 2] = A[0, 2] + x2[i]
        A[1, 0] = A[0, 1]
        A[1, 1] = A[1, 1] + y2[i] ** 2
        A[1, 2] = A[1, 2] + y2[i]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = len(x2)
    # print(A)

    # 创建b
    b = np.zeros((3, 1))
    for i in range(len(x2)):
        b[0, 0] = b[0, 0] + x2[i] * z2[i]
        b[1, 0] = b[1, 0] + y2[i] * z2[i]
        b[2, 0] = b[2, 0] + z2[i]
    # print(b)

    # 求解X
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, b)
    print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))

    # 计算方差
    R = 0
    for i in range(len(x2)):
        R = R + (X[0, 0] * x2[i] + X[1, 0] * y2[i] + X[2, 0] - z2[i]) ** 2
    print('方差为：%.*f' % (3, R))

    # 展示图像
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    # ax1.set_zlabel("z")
    # ax1.scatter(x2, y2, z2, c='r', marker='o')
    # x_p = np.linspace(np.min(x2), np.max(x2), 100)
    # y_p = np.linspace(np.min(y2), np.max(y2), 100)
    # x_p, y_p = np.meshgrid(x_p, y_p)
    # z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]
    # ax1.plot_wireframe(x_p, y_p, z_p, rstride=10, cstride=10)
    plt.pause(2)

    return [X[0, 0], X[1, 0], X[2, 0]]


def get_normal_vector(point1, point2, point3):
    """三个点计算平面法向量"""
    vect1 = np.array(point2) - np.array(point1)
    vect2 = np.array(point3) - np.array(point1)
    norm_vect = np.cross(vect1, vect2)
    return norm_vect


def get_R_matrix( vector_src, vector_tgt):
    """计算两平面(法向量)间的旋转矩阵"""
    vector_src  =  vector_src / np.linalg.norm( vector_src)
    vector_tgt =  vector_tgt / np.linalg.norm( vector_tgt)

    c = np.dot( vector_src,  vector_tgt)
    n_vector = np.cross( vector_src,  vector_tgt)

    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]))

    I = np.eye(3)
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
    return R_w2c
    

def plot_linear_cube(poses=[], poses_new=[], x=-1, y=-1, z=-1, dx=2, dy=2, dz=2, color="black",pts=np.array([[0,0,0]])):
    fig = plt.figure()
    ax = Axes3D(fig)

    # 画边框
    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]

    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z] * 5, **kwargs)
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)

    # 画位置
    if (len(poses) > 0):
        ax.scatter(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], marker="o")

    if (len(poses_new) > 0):
        ax.scatter(poses_new[:, 0, 3], poses_new[:, 1, 3], poses_new[:, 2, 3], marker="o", color="red")

    # 画姿态
    kwargs_X = {'alpha': 1, 'color': "red"}
    kwargs_Y = {'alpha': 1, 'color': "green"}
    kwargs_Z = {'alpha': 1, 'color': "blue"}

    for i in range(len(poses)):
        x_ = poses[i, :3, 3] + poses[i, :3, 0]
        y_ = poses[i, :3, 3] + poses[i, :3, 1]
        z_ = poses[i, :3, 3] + poses[i, :3, 2]

        ax.plot3D([poses[i, 0, 3], x_[0]], [poses[i, 1, 3], x_[1]], [poses[i, 2, 3], x_[2]], **kwargs_X)
        ax.plot3D([poses[i, 0, 3], y_[0]], [poses[i, 1, 3], y_[1]], [poses[i, 2, 3], y_[2]], **kwargs_Y)
        ax.plot3D([poses[i, 0, 3], z_[0]], [poses[i, 1, 3], z_[1]], [poses[i, 2, 3], z_[2]], **kwargs_Z)

    for i in range(len(poses_new)):
        x_ = poses_new[i, :3, 3] + poses_new[i, :3, 0]
        y_ = poses_new[i, :3, 3] + poses_new[i, :3, 1]
        z_ = poses_new[i, :3, 3] + poses_new[i, :3, 2]

        ax.plot3D([poses_new[i, 0, 3], x_[0]], [poses_new[i, 1, 3], x_[1]], [poses_new[i, 2, 3], x_[2]], **kwargs_X)
        ax.plot3D([poses_new[i, 0, 3], y_[0]], [poses_new[i, 1, 3], y_[1]], [poses_new[i, 2, 3], y_[2]], **kwargs_Y)
        ax.plot3D([poses_new[i, 0, 3], z_[0]], [poses_new[i, 1, 3], z_[1]], [poses_new[i, 2, 3], z_[2]], **kwargs_Z)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    if len(pts)>1:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker=".")
    else:
        center=np.mean(pts[:,:3],axis=0)
        ax.scatter(center[0],center[1],center[2],marker="o",c="purple")

    # plt.pause(2)
    plt.show()


def scene_lookdown(poses, center_tgt=np.array([0, 0, 1]),scale=2):
    # 0. 伴随矩阵，记录变换
    scale_mat = np.eye(4).astype(np.float32)
    # 1. 旋转变换
    # 1.1 拟合平面方程
    f_p = fit_a_plane(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3])

    # 在平面上拿出三个点
    points = np.array([[-1., -1., 0.],
                       [-1., 1., 0.],
                       [1., 1., 0.]])

    points[0, 2] = points[0, 0] * f_p[0] + points[0, 1] * f_p[1] + f_p[2]
    points[1, 2] = points[1, 0] * f_p[0] + points[1, 1] * f_p[1] + f_p[2]
    points[2, 2] = points[2, 0] * f_p[0] + points[2, 1] * f_p[1] + f_p[2]

    # 1.2 计算法向量
    normal_p = get_normal_vector(points[0, :], points[1, :], points[2, :])

    # 目标平面的法向量
    normal_cube = np.array([0., 0., -1.])

    # 1.3 求旋转矩阵
    R_w2c = get_R_matrix(normal_p, normal_cube)

    # 1.4 对位置进行旋转变换
    # 先从位姿矩阵中取出位置坐标
    p_src = np.zeros(shape=(len(poses[:, 0, 3]), 3))
    p_src[:, 0] = poses[:, 0, 3]
    p_src[:, 1] = poses[:, 1, 3]
    p_src[:, 2] = poses[:, 2, 3]

    np.savetxt("poses_location.txt",p_src,delimiter=",")

    # 再对位置坐标进行旋转变换
    p_new = np.dot(R_w2c, np.transpose(p_src))

    # 把变换后的相机位置坐标放到位姿矩阵中
    poses_new = poses.copy()
    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]


    # 1.5 对姿态进行旋转变换，其实就是对三个列向量（坐标轴）进行旋转变换
    poses_new[:, :3, 0] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 0])).transpose()
    poses_new[:, :3, 1] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 1])).transpose()
    poses_new[:, :3, 2] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 2])).transpose()

    scale_mat[:3,:3]=R_w2c
    if(scale>0):
        # 2. 对位置进行缩放变换
        max_vertices = np.max(p_new, axis=1)
        min_vertices = np.min(p_new, axis=1)
        # 这里求的是缩放因子，因为立方体的边长为2
        scene_scale = scale/ (np.max(max_vertices - min_vertices))


        # 2.1 应用缩放参数
        poses_new[:, :3, 3] *= scene_scale
        p_new[0, :] = poses_new[:, 0, 3]
        p_new[1, :] = poses_new[:, 1, 3]
        p_new[2, :] = poses_new[:, 2, 3]

        # 伴随矩阵同步记录变换
        scale_mat[3,3] *= scene_scale

    # 3. 对位置进行平移变换
    T_move = np.array([center_tgt - np.mean(p_new, axis=1)])
    p_new = p_new + T_move.transpose()

    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]

    np.savetxt("poses_new.txt", p_new.T, delimiter=",")
    #
    # 伴随矩阵，记录变换
    scale_mat[:3, 3] += T_move[0]

    plot_linear_cube(poses, poses_new)

    plot_linear_cube( poses_new)


    return poses_new,scale_mat


def scene_true(poses, center_tgt=np.array([0, 0, 0]),scale=2):
    # 0. 伴随矩阵，记录变换
    scale_mat = np.eye(4).astype(np.float32)
    p_new = np.zeros(shape=(len(poses[:, 0, 3]), 3))
    p_new[:, 0] = poses[:, 0, 3]
    p_new[:, 1] = poses[:, 1, 3]
    p_new[:, 2] = poses[:, 2, 3]

    p_new = np.transpose(p_new)

    # 把变换后的相机位置坐标放到位姿矩阵中
    poses_new = poses.copy()
    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]

    # 2. 不缩放
    if (scale > 0):
        # 2. 对位置进行缩放变换
        max_vertices = np.max(p_new, axis=1)
        min_vertices = np.min(p_new, axis=1)
        # 这里求的是缩放因子，因为立方体的边长为2
        scene_scale = scale / (np.max(max_vertices - min_vertices))

        # 2.1 应用缩放参数
        poses_new[:, :3, 3] *= scene_scale
        p_new[0, :] = poses_new[:, 0, 3]
        p_new[1, :] = poses_new[:, 1, 3]
        p_new[2, :] = poses_new[:, 2, 3]

        # 伴随矩阵同步记录变换
        scale_mat[3, 3] *= scene_scale


    # 平移变换
    T_move = np.array([center_tgt - np.mean(p_new, axis=1)])
    p_new = p_new + T_move.transpose()

    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]

    scale_mat[:3, 3] += T_move[0]


    plot_linear_cube(poses, poses_new)

    plot_linear_cube( poses_new)
    # 伴随矩阵，记录变换
    return poses_new,scale_mat



def poses_transform(poses, center_tgt=np.array([0, 0, 1])):
    # 1. 旋转变换
    # 1.1 拟合平面方程
    f_p = fit_a_plane(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3])

    # 在平面上拿出三个点
    points = np.array([[-1., -1., 0.],
                       [-1., 1., 0.],
                       [1., 1., 0.]])

    points[0, 2] = points[0, 0] * f_p[0] + points[0, 1] * f_p[1] + f_p[2]
    points[1, 2] = points[1, 0] * f_p[0] + points[1, 1] * f_p[1] + f_p[2]
    points[2, 2] = points[2, 0] * f_p[0] + points[2, 1] * f_p[1] + f_p[2]

    # 1.2 计算法向量
    normal_p = get_normal_vector(points[0, :], points[1, :], points[2, :])

    # 目标平面的法向量
    normal_cube = np.array([0., 0., -1.])

    # 1.3 求旋转矩阵
    R_w2c = get_R_matrix(normal_p, normal_cube)

    # 1.4 对位置进行旋转变换
    # 先从位姿矩阵中取出位置坐标
    p_src = np.zeros(shape=(len(poses[:, 0, 3]), 3))
    p_src[:, 0] = poses[:, 0, 3]
    p_src[:, 1] = poses[:, 1, 3]
    p_src[:, 2] = poses[:, 2, 3]
    # 再对位置坐标进行旋转变换
    p_new = np.dot(R_w2c, np.transpose(p_src))

    # 把变换后的相机位置坐标放到位姿矩阵中
    poses_new = poses.copy()
    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]

    # 1.5 对姿态进行旋转变换，其实就是对三个列向量（坐标轴）进行旋转变换
    poses_new[:, :3, 0] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 0])).transpose()
    poses_new[:, :3, 1] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 1])).transpose()
    poses_new[:, :3, 2] = np.dot(R_w2c, np.transpose(poses_new[:, :3, 2])).transpose()

    # 2. 缩放变换
    # 2.1 求相机位置坐标所占空间大小
    max_vertices = np.max(p_new, axis=1)
    min_vertices = np.min(p_new, axis=1)

    # 2.2 求缩放因子，目标尺寸为2
    scene_scale = 2 / (np.max(max_vertices - min_vertices))

    # 2.2. 对位置进行缩放变换
    poses_new[:, :3, 3] *= scene_scale
    p_new[0, :] = poses_new[:, 0, 3]
    p_new[1, :] = poses_new[:, 1, 3]
    p_new[2, :] = poses_new[:, 2, 3]

    # 3. 对位置进行平移变换
    T_move = np.array([center_tgt - np.mean(p_new, axis=1)])
    p_new = p_new - T_move.transpose()

    poses_new[:, 0, 3] = p_new[0, :]
    poses_new[:, 1, 3] = p_new[1, :]
    poses_new[:, 2, 3] = p_new[2, :]

    return poses_new

if __name__ == '__main__':
    pass
