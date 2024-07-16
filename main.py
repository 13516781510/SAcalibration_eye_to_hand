import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_plane(points):
    def plane_func(params, points):
        a, b, c, d = params
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return a * x + b * y + c * z + d

    points = np.array(points)
    params_initial = [1, 1, 1, -points.mean(axis=0).dot([1, 1, 1])]
    params, _ = leastsq(plane_func, params_initial, args=(points,))
    return params


def plot_points_and_planes_z(ax, points, plane_params, color, label):
    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label=f'{label} points')

    # 绘制平面
    a, b, c, d = plane_params
    xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 10),
                         np.linspace(min(points[:, 1]), max(points[:, 1]), 10))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)
def plot_points_and_planes_y(ax, points, plane_params, color, label):
    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label=f'{label} points')

    # 绘制平面
    a, b, c, d = plane_params
    xx, zz = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 10),
                         np.linspace(min(points[:, 2]), max(points[:, 2]), 10))
    yy = (-a * xx - c * zz - d) / b
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)

def plot_points_and_planes_x(ax, points, plane_params, color, label):
    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, label=f'{label} points')

    # 绘制平面
    a, b, c, d = plane_params
    yy,zz = np.meshgrid(np.linspace(min(points[:, 1]), max(points[:, 1]), 10),
                         np.linspace(min(points[:, 2]), max(points[:, 2]), 10))
    xx = (-c * zz - b * yy - d) / a
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)
def calculate_coordinate_system(plane_params):
    normal = plane_params[:3]
    normal = normal / np.linalg.norm(normal)
    return normal


def calculate_origin(A, d):
    if np.linalg.matrix_rank(A) == 3:
        origin = np.linalg.solve(A, d)
    else:
        origin = np.array([0, 0, 0])  # 假设原点在(0,0,0)
    return origin


def calculate_transformation_matrix(origin1, origin2, axes1, axes2):
    R = np.linalg.inv(axes2) @ axes1
    T = origin2 - (R @ origin1)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T
    return transformation_matrix


# 读取点数据
pointsa1 = np.loadtxt('pointsa1.txt')
pointsb1 = np.loadtxt('pointsb1.txt')
pointsc1 = np.loadtxt('pointsc1.txt')

pointsa2 = np.loadtxt('pointsa2.txt')
pointsb2 = np.loadtxt('pointsb2.txt')
pointsc2 = np.loadtxt('pointsc2.txt')

# 拟合两个数据集的三个面的平面
plane_a1 = fit_plane(pointsa1)
plane_b1 = fit_plane(pointsb1)
plane_c1 = fit_plane(pointsc1)

plane_a2 = fit_plane(pointsa2)
plane_b2 = fit_plane(pointsb2)
plane_c2 = fit_plane(pointsc2)

# 计算第一个数据集的三个平面的法向量
normal_a1 = plane_a1[:3]
normal_b1 = plane_b1[:3]
normal_c1 = plane_c1[:3]

# 计算第一个数据集的坐标轴（使用法向量计算坐标系）
x_axis1 = normal_a1 / np.linalg.norm(normal_a1)
y_axis1 = normal_b1 / np.linalg.norm(normal_b1)
z_axis1 = normal_c1 / np.linalg.norm(normal_c1)

axes1 = np.column_stack((x_axis1, y_axis1, z_axis1))

# 解方程组计算第一个数据集的原点坐标
A1 = np.array([normal_a1, normal_b1, normal_c1])
d1 = -np.array([plane_a1[3], plane_b1[3], plane_c1[3]])
origin1 = calculate_origin(A1, d1)

# 计算第二个数据集的三个平面的法向量
normal_a2 = plane_a2[:3]
normal_b2 = plane_b2[:3]
normal_c2 = plane_c2[:3]

# 计算第二个数据集的坐标轴（使用法向量计算坐标系）
x_axis2 = normal_a2 / np.linalg.norm(normal_a2)
y_axis2 = normal_b2 / np.linalg.norm(normal_b2)
z_axis2 = normal_c2 / np.linalg.norm(normal_c2)

axes2 = np.column_stack((x_axis2, y_axis2, z_axis2))

# 解方程组计算第二个数据集的原点坐标
A2 = np.array([normal_a2, normal_b2, normal_c2])
d2 = -np.array([plane_a2[3], plane_b2[3], plane_c2[3]])
origin2 = calculate_origin(A2, d2)

# 计算两个坐标系的转移矩阵
transformation_matrix = calculate_transformation_matrix(origin1, origin2, axes1, axes2)
print("Transformation matrix from coordinate system 1 to 2:")
print(transformation_matrix)

# 绘制结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个数据集的点和拟合平面
plot_points_and_planes_z(ax, pointsa1, plane_a1, 'r', 'Dataset 1 Plane A')
plot_points_and_planes_y(ax, pointsb1, plane_b1, 'g', 'Dataset 1 Plane B')
plot_points_and_planes_x(ax, pointsc1, plane_c1, 'b', 'Dataset 1 Plane C')


# 绘制第一个数据集的坐标轴
ax.quiver(*origin1, *x_axis1, length=10, color='r', label='Dataset 1 X axis')
ax.quiver(*origin1, *y_axis1, length=10, color='g', label='Dataset 1 Y axis')
ax.quiver(*origin1, *z_axis1, length=10, color='b', label='Dataset 1 Z axis')

# 绘制第二个数据集的点和拟合平面
plot_points_and_planes_z(ax, pointsa2, plane_a2, 'm', 'Dataset 2 Plane A')
plot_points_and_planes_y(ax, pointsb2, plane_b2, 'c', 'Dataset 2 Plane B')
plot_points_and_planes_x(ax, pointsc2, plane_c2, 'y', 'Dataset 2 Plane C')

# 绘制第二个数据集的坐标轴
ax.quiver(*origin2, *x_axis2, length=10, color='m', label='Dataset 2 X axis')
ax.quiver(*origin2, *y_axis2, length=10, color='c', label='Dataset 2 Y axis')
ax.quiver(*origin2, *z_axis2, length=10, color='y', label='Dataset 2 Z axis')

# 设置图形标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fitted Coordinate Systems and Points')
ax.legend()

plt.show()
