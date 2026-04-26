#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件提供碰撞修正时复用的最底层数值工具。
# 上游输入来自障碍碰撞和机间碰撞模块给出的速度修正量或角速度扰动尺度；
# 下游输出会直接写回 `QuadrotorDynamics` 的速度和角速度。
# 这层不负责判断谁撞了谁，只负责把“碰撞后应该怎么衰减/扰动”变成具体数值。

import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS


@njit
def compute_new_vel(max_vel_magn, vel, vel_shift, low=0.2, high=0.8):
    # 碰撞模块先给出一个理想的速度修正方向 `vel_shift`，
    # 这里再叠加一个随机衰减比例，把碰撞后的速度幅值压回更合理的范围。
    vel_decay_ratio = np.random.uniform(low, high)
    vel_new = vel + vel_shift
    vel_new_mag = np.linalg.norm(vel_new)
    vel_new_dir = vel_new / (vel_new_mag + EPS if vel_new_mag == 0.0 else vel_new_mag)
    vel_new_mag = min(vel_new_mag * vel_decay_ratio, max_vel_magn)
    vel_new = vel_new_dir * vel_new_mag

    # 返回值仍是“在原速度基础上就地改写后的新速度”，
    # 供障碍碰撞和机间碰撞函数直接写回动力学层。
    vel_shift = vel_new - vel
    vel += vel_shift
    return vel


@njit
def compute_new_omega(magn_scale=20.0):
    # 这里生成碰撞后的随机角速度扰动方向和幅值。
    # 目的不是精确模拟真实碰撞力矩，而是让碰撞后姿态确实被打乱，训练中能感知到失稳代价。
    # This will amount to max 3.5 revolutions per second
    omega_max = magn_scale * np.pi
    omega = np.random.uniform(-1, 1, size=(3,))
    omega_mag = np.linalg.norm(omega)

    omega_dir = omega / (omega_mag + EPS if omega_mag == 0.0 else omega_mag)
    omega_mag = np.random.uniform(omega_max / 2, omega_max)
    omega = omega_dir * omega_mag

    return omega


if __name__ == "__main__":
    # 这一段只是本地速度测试入口，不参与训练主链路。
    def main():
        import timeit
        SETUP_CODE = '''from __main__ import calculate_collision_matrix; import numpy as np'''

        TEST_CODE = '''calculate_collision_matrix(positions=np.ones((8, 3)), arm=0.05, hitbox_radius=2)'''

        # timeit.repeat statement
        times = timeit.repeat(setup=SETUP_CODE,
                              stmt=TEST_CODE,
                              repeat=5,
                              number=int(1e4))

        # printing minimum exec. time
        print('times:   ', times)
        print('mean times:   ', np.mean(times[1:]))


    if __name__ == '__main__':
        import sys

        sys.exit(main())
