# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
from numba import njit

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `compute_col_norm_and_new_vel_obst`。
def compute_col_norm_and_new_vel_obst(pos, vel, obstacle_pos):
    # 保存或更新 `collision_norm` 的值。
    collision_norm = pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    # 保存或更新 `collision_norm[2]` 的值。
    collision_norm[2] = 0.0
    # 保存或更新 `coll_norm_mag` 的值。
    coll_norm_mag = np.linalg.norm(collision_norm)
    # 执行这一行逻辑。
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    # 保存或更新 `vnew` 的值。
    vnew = np.dot(vel, collision_norm)

    # 返回当前函数的结果。
    return vnew, collision_norm


# 定义函数 `perform_collision_with_obstacle`。
def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size):
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    # 同时更新 `vnew`, `collision_norm` 等变量。
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(drone_dyn.pos, drone_dyn.vel, obstacle_pos)
    # 保存或更新 `vel_magn` 的值。
    vel_magn = np.linalg.norm(drone_dyn.vel)
    # 保存或更新 `new_vel` 的值。
    new_vel = vel_magn * collision_norm

    # 保存或更新 `vel_noise` 的值。
    vel_noise = np.zeros(3)
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(3):
        # 保存或更新 `cons_rand_val` 的值。
        cons_rand_val = np.random.normal(loc=0, scale=0.1, size=3)
        # 保存或更新 `tmp_vel_noise` 的值。
        tmp_vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.05, size=3)
        # 根据条件决定是否进入当前分支。
        if np.dot(new_vel + tmp_vel_noise, collision_norm) > 0:
            # 保存或更新 `vel_noise` 的值。
            vel_noise = tmp_vel_noise
            # 提前结束当前循环。
            break

    # 保存或更新 `max_vel_magn` 的值。
    max_vel_magn = np.linalg.norm(drone_dyn.vel)
    # In case drone that is inside the obstacle
    # 根据条件决定是否进入当前分支。
    if np.linalg.norm(drone_dyn.pos - obstacle_pos) < obstacle_size / 2:
        # 保存或更新 `drone_dyn.vel` 的值。
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise, low=1.0, high=1.0)
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `drone_dyn.vel` 的值。
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise)

    # Random forces for omega
    # 保存或更新 `new_omega` 的值。
    new_omega = compute_new_omega(magn_scale=1.0)
    # 保存或更新 `drone_dyn.omega` 的值。
    drone_dyn.omega += new_omega
