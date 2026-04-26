#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/room.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件负责房间边界碰撞后的动力学修正。
# 上游输入来自 `QuadrotorDynamics` 已经做过位置裁剪后的墙面/天花板碰撞标记；
# 下游输出直接写回无人机的速度和角速度，并间接影响后续观测、奖励和 episode 统计。
# 与地面接触不同，墙面和天花板碰撞不在动力学内部持续处理，而是由多机环境额外调用这里的修正函数。

import numba as nb
import numpy as np
from numba import njit


def perform_collision_with_wall(drone_dyn, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # 撞墙后不会简单把速度置零，而是重采一个朝房间内部偏转的新速度方向，
    # 同时按一定比例衰减速度幅值，避免无人机贴墙连续震荡。
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    # 当前位置决定“应该朝哪个半空间弹回去”。
    # 例如贴在左墙时，x 方向必须重新指向房间内部的正向。
    drone_pos = drone_dyn.pos
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    if x_list[0]:
        direction[0] = np.random.uniform(low=0.1, high=1.0)
    elif x_list[1]:
        direction[0] = np.random.uniform(low=-1.0, high=-0.1)

    if y_list[0]:
        direction[1] = np.random.uniform(low=0.1, high=1.0)
    elif y_list[1]:
        direction[1] = np.random.uniform(low=-1.0, high=-0.1)

    direction[2] = np.random.uniform(low=-1.0, high=-0.5)

    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # 碰撞同时给一个随机角速度冲击，表示撞墙后的姿态扰动。
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    drone_dyn.omega += new_omega


@njit
def perform_collision_with_wall_numba(vel, pos, omega, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                      lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # 这是墙碰撞修正的 numba 版本，语义与 Python 路径一致，
    # 只是为了让多机高频环境里批量碰撞修正更便宜。
    drone_speed = np.linalg.norm(vel)
    real_speed = nb.random.uniform(damp_low_speed_ratio * drone_speed, damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, 0.1, 6.0)

    # 和 Python 版本一样，这里根据当前贴到的是哪一面墙，约束回弹方向朝室内。
    drone_pos = pos
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    direction = np.random.uniform(-1.0, 1.0, size=(3,))
    if x_list[0]:
        direction[0] = np.random.uniform(0.1, 1.0)
    elif x_list[1]:
        direction[0] = np.random.uniform(-1.0, -0.1)

    if y_list[0]:
        direction[1] = np.random.uniform(0.1, 1.0)
    elif y_list[1]:
        direction[1] = np.random.uniform(-1.0, -0.1)

    direction[2] = np.random.uniform(-1.0, -0.5)

    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    vel = real_speed * direction_norm

    # 角速度也同步被打乱，避免碰撞后姿态仍保持“过于理想”的稳定状态。
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(-1, 1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(omega_max / 2, omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    omega += new_omega

    return vel, omega


def perform_collision_with_ceiling(drone_dyn, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                   lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # 撞天花板时处理比撞墙更简单：
    # 只要确保 z 方向重新朝下，并附带一定速度衰减和角速度扰动即可。
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    drone_dyn.omega += new_omega
