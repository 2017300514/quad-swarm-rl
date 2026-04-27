# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/speed_test/quadrotor.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是多机碰撞响应函数的微型性能对比脚本。
# 它把一个“朴素 python 复写版”与正式 `perform_collision_between_drones` 放到相同输入上，
# 只比较每次碰撞响应的耗时，不负责验证环境级行为。

import sys
import timeit
import numpy as np

from gym_art.quadrotor_multi.collisions.quadrotors import compute_col_norm_and_new_velocities
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


# 这里手写了一份与正式实现逻辑等价、但不追求优化的参考版本，
# 用来给下面的 `timeit` 提供 baseline。
def normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # 先按弹性碰撞几何求出法向速度交换，再叠加论文实现里用于打散对称碰撞的随机扰动。
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
    vel_change = (v2new - v1new) * collision_norm
    dyn1_vel_shift = vel_change
    dyn2_vel_shift = -vel_change

    # 这里反复采样扰动，直到碰后速度沿碰撞法向的方向确实分离，而不是继续互相挤压。
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel1_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        vel2_noise = -cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)

        dyn1_vel_shift = vel_change + vel1_noise
        dyn2_vel_shift = -vel_change + vel2_noise

        dyn1_new_vel_dir = np.dot(vel1 + dyn1_vel_shift, collision_norm)
        dyn2_new_vel_dir = np.dot(vel2 + dyn2_vel_shift, collision_norm)

        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            break

    # 线速度和角速度的最终裁剪/扰动仍然复用正式工具函数，避免把 benchmark 写成另一套物理规则。
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    new_omega = compute_new_omega()
    omega1 += new_omega
    omega2 -= new_omega

    return vel1, omega1, vel2, omega2


# 先 benchmark 参考实现，再 benchmark 正式实现。
# 两边输入完全相同，这样输出的均值能直接作为“优化前后碰撞响应开销”的粗略对照。
def test_perform_collision_between_drones():
    SETUP_CODE = '''from __main__ import normal_perform_collision_between_drones; import numpy as np'''

    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e3))

    print('Speed: perform_collision_between_drones')
    print('Normal:   ', times)
    print('Mean:   ', np.mean(times))


    SETUP_CODE = '''from gym_art.quadrotor_multi.collisions.quadrotors import perform_collision_between_drones; import numpy as np'''

    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=6,
                          number=int(1e3))

    print('Optimized:   ', times)
    print('Mean:   ', np.mean(times[1:]))


# 手工运行入口。
def speed_test():
    test_perform_collision_between_drones()
    return


if __name__ == "__main__":
    sys.exit(speed_test())
