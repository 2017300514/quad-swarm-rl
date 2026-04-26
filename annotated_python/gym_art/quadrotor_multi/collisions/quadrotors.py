#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/quadrotors.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件处理多机环境里的机间碰撞与近距离接近惩罚。
# 上游输入来自 `QuadrotorEnvMulti` 每一步同步出的所有无人机位置、速度和角速度；
# 下游输出包括碰撞对列表、pairwise 距离表、逐 agent 接近惩罚，以及碰撞后写回动力学层的新速度/角速度。
# 结合 `quadrotor_multi.py` 一起看时，这里就是多机安全约束的底层数值实现。

import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


@njit
def compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2):
    # 先用两架无人机当前位置差构造碰撞法向。
    # 后面无论是弹性近似速度交换，还是碰撞后相反方向分离，都围绕这条法向展开。
    collision_norm = pos1 - pos2
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # 这里只取双方速度在碰撞法向上的分量，
    # 垂直于碰撞法向的分量不会在这个近似模型里被直接交换。
    v1new = np.dot(vel1, collision_norm)
    v2new = np.dot(vel2, collision_norm)

    return v1new, v2new, collision_norm


@njit
def perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # 这个函数负责“碰撞已确认之后，双方动力学状态如何改写”。
    # 它以弹性碰撞近似为骨架，再叠加随机扰动，目标是让两架无人机可靠分离，而不是精确复现真实桨机碰撞细节。
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
    vel_change = (v2new - v1new) * collision_norm
    dyn1_vel_shift = vel_change
    dyn2_vel_shift = -vel_change

    # 这里反复采样噪声，直到两架无人机在碰撞法向上朝相反方向离开。
    # 这样可以减少“碰撞后还继续挤在一起，下一步再次判碰”的情况。
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

    # 双方速度幅值都限制在碰撞前最大速度量级附近，避免一次修正引入离谱高速度。
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    # 角速度扰动一正一负地分给双方，表示碰撞带来的对向自旋冲击。
    new_omega = compute_new_omega()
    omega1 += new_omega
    omega2 -= new_omega

    return vel1, omega1, vel2, omega2


@njit
def calculate_collision_matrix(positions, collision_threshold):
    """
    drone_col_matrix: set collided quadrotors' id to 1
    curr_drone_collisions: [i, j]
    distance_matrix: [i, j, dist]
    """
    # 这是多机环境每个 step 都会调用的核心碰撞扫描器。
    # 它一次性枚举所有 pair，既产出硬碰撞对，也保留距离表，供接近惩罚和日志统计继续使用。
    num_agents = len(positions)
    item_num = int(num_agents * (num_agents - 1) / 2)
    count = int(0)

    # 这里大量使用固定形状数组而不是动态 list，主要是为了兼容 numba 并降低多机高频循环开销。
    drone_col_matrix = -1000 * np.ones(num_agents)
    curr_drone_collisions = -1000. * np.ones((item_num, 2))
    distance_matrix = -1000. * np.ones((item_num, 3))

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance_matrix[count] = [i, j,
                                      ((positions[i][0] - positions[j][0]) ** 2 +
                                       (positions[i][1] - positions[j][1]) ** 2 +
                                       (positions[i][2] - positions[j][2]) ** 2) ** 0.5]

            if distance_matrix[count][2] <= collision_threshold:
                # 一旦小于机体碰撞阈值，就把双方标记成发生过硬碰撞，
                # 后面环境会基于这个结果累计碰撞次数、施加离散惩罚并决定是否做物理碰撞修正。
                drone_col_matrix[i] = 1
                drone_col_matrix[j] = 1
                curr_drone_collisions[count] = [i, j]

            count += 1

    return drone_col_matrix, curr_drone_collisions, distance_matrix


@njit
def calculate_drone_proximity_penalties(distance_matrix, collision_falloff_threshold, dt, max_penalty, num_agents):
    # 这部分对应的不是“已经撞上”的离散惩罚，而是距离进入 falloff 半径后的连续接近惩罚。
    # 多机环境会把这里的逐 agent 惩罚叠加到 reward 上，让策略在真正碰撞前就学会互相避让。
    penalties = np.zeros(num_agents)
    penalty_ratio = -max_penalty / collision_falloff_threshold
    for i, j, dist in distance_matrix:
        penalty = penalty_ratio * dist + max_penalty
        penalties[int(i)] += penalty
        penalties[int(j)] += penalty

    return dt * penalties
