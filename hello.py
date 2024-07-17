import numpy as np

np.set_printoptions(suppress=True, precision=8)
import numpy as np
point1 = np.array([18.823423, -50.267420, -998.146684, 1])  # 齐次坐标
point2 = np.array([65.218913 , -10.216712 , -1001.484967,1])  # 齐次坐标
point3 = np.array([105.493478 , -56.160742 , -1004.845655, 1])  # 齐次坐标
point4 = np.array([58.902523,  -96.161571,  -998.146684, 1])  # 齐次坐标
T_B_to_A = np.array([[0.750757, 0.660578, 0.009498, 341.94],
                     [-0.660534, 0.750688, 0.012706, 1513.68],
                     [0.007682, -0.010165, 0.999919, -305.18],
                     [0, 0, 0, 1]])
# 相机下坐标到标定块上坐标的转移矩阵
T_C_to_B = np.array([[0.754252571, 0.64756621, - 0.108448434, - 90.40384004],
                     [-0.651483068, 0.75866263, -0.000908224, 110.4216461],
                     [0.081687639, 0.071337349, 0.994101661, 994.3075799],
                     [0, 0, 0, 1]])
def calculate(point):
    # T_C_to_B是target到camera的

    point_B = np.dot(T_C_to_B, point)
    print("相机坐标系下坐标点",point, "在标定快坐标系下的坐标:", point_B)

    point_D = np.dot(T_B_to_A, point_B)

    print("相机坐标系下坐标点",point, "在基座标系下的坐标:", point_D)

calculate(point1)
calculate(point2)
calculate(point3)
calculate(point4)
res = np.dot(T_B_to_A, T_C_to_B)
print("两个转移矩阵左乘得到，直接左乘的转移矩阵：",res)
#[[   0.13668088    0.98799827   -0.0725764   356.45472779]
 # [  -0.98623207    0.14268585    0.08358314 1668.92068685]
 # [   0.09409752    0.06859437    0.99319727  687.23012265]
 # [   0.            0.            0.            1.        ]]
point_D = np.dot(res, point4)
print("相机坐标系下坐标点", point4, "在基座标系下的坐标:", point_D)