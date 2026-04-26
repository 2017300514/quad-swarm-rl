#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件负责“无人机撞到障碍物之后，动力学状态怎样被立即改写”。
# 上游输入来自多机环境在每个 step 中识别出的碰撞无人机和对应障碍中心位置；
# 下游输出直接写回 `QuadrotorDynamics` 的速度与角速度，并间接影响下一步观测、奖励和碰撞统计。
# 这一层不判断有没有碰撞，碰撞检测本身来自 `obstacles/utils.py` 和 `MultiObstacles.collision_detection()`。

import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


@njit
def compute_col_norm_and_new_vel_obst(pos, vel, obstacle_pos):
    # 这里先构造障碍物指向无人机的碰撞法向。
    # 由于当前障碍物被视作贯穿房间高度的柱体，所以只看 xy 平面，不考虑 z 方向分量。
    collision_norm = pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    collision_norm[2] = 0.0
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # 这里取速度在碰撞法向上的投影，作为“沿障碍法向被弹开”的基础方向信息。
    vnew = np.dot(vel, collision_norm)

    return vnew, collision_norm


def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size):
    # 上层环境在确认某架无人机撞上障碍后，会调用这个函数直接修改它的动力学状态。
    # 这里并不追求精确刚体碰撞，而是构造一个稳定的“弹开 + 自旋扰动”近似，避免无人机卡在障碍边缘。
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(drone_dyn.pos, drone_dyn.vel, obstacle_pos)
    vel_magn = np.linalg.norm(drone_dyn.vel)
    new_vel = vel_magn * collision_norm

    # 这一段随机噪声的目标不是纯随机化，
    # 而是保证修正后的速度大致沿着远离障碍的方向离开，避免下一步还继续往障碍内部钻。
    vel_noise = np.zeros(3)
    for i in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.1, size=3)
        tmp_vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.05, size=3)
        if np.dot(new_vel + tmp_vel_noise, collision_norm) > 0:
            vel_noise = tmp_vel_noise
            break

    max_vel_magn = np.linalg.norm(drone_dyn.vel)
    # 如果机体已经穿进障碍半径内部，就用更强硬的速度修正把它推出去，
    # 否则按常规范围裁剪，保留更多原始速度量级。
    if np.linalg.norm(drone_dyn.pos - obstacle_pos) < obstacle_size / 2:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise, low=1.0, high=1.0)
    else:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise)

    # 角速度也会被额外扰动，模拟碰撞带来的姿态冲击。
    # 多机环境后面会基于这个新 `omega` 继续计算观测、姿态稳定奖励和后续动力学演化。
    new_omega = compute_new_omega(magn_scale=1.0)
    drone_dyn.omega += new_omega
