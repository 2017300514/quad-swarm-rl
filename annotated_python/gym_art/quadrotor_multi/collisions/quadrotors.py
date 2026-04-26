# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/quadrotors.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
from numba import njit

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `compute_col_norm_and_new_velocities`。
def compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2):
    # Ge the collision normal, i.e difference in position
    # 保存或更新 `collision_norm` 的值。
    collision_norm = pos1 - pos2
    # 保存或更新 `coll_norm_mag` 的值。
    coll_norm_mag = np.linalg.norm(collision_norm)
    # 执行这一行逻辑。
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    # 保存或更新 `v1new` 的值。
    v1new = np.dot(vel1, collision_norm)
    # 保存或更新 `v2new` 的值。
    v2new = np.dot(vel2, collision_norm)

    # 返回当前函数的结果。
    return v1new, v2new, collision_norm


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `perform_collision_between_drones`。
def perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    # 同时更新 `v1new`, `v2new`, `collision_norm` 等变量。
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
    # 保存或更新 `vel_change` 的值。
    vel_change = (v2new - v1new) * collision_norm
    # 保存或更新 `dyn1_vel_shift` 的值。
    dyn1_vel_shift = vel_change
    # 保存或更新 `dyn2_vel_shift` 的值。
    dyn2_vel_shift = -vel_change

    # Make sure new vel direction would be opposite to the original vel direction
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for _ in range(3):
        # 保存或更新 `cons_rand_val` 的值。
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        # 保存或更新 `vel1_noise` 的值。
        vel1_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        # 保存或更新 `vel2_noise` 的值。
        vel2_noise = -cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)

        # 保存或更新 `dyn1_vel_shift` 的值。
        dyn1_vel_shift = vel_change + vel1_noise
        # 保存或更新 `dyn2_vel_shift` 的值。
        dyn2_vel_shift = -vel_change + vel2_noise

        # 保存或更新 `dyn1_new_vel_dir` 的值。
        dyn1_new_vel_dir = np.dot(vel1 + dyn1_vel_shift, collision_norm)
        # 保存或更新 `dyn2_new_vel_dir` 的值。
        dyn2_new_vel_dir = np.dot(vel2 + dyn2_vel_shift, collision_norm)

        # 根据条件决定是否进入当前分支。
        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            # 提前结束当前循环。
            break

    # Get new vel
    # 保存或更新 `max_vel_magn` 的值。
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    # 保存或更新 `vel1` 的值。
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    # 保存或更新 `vel2` 的值。
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    # Get new omega
    # 保存或更新 `new_omega` 的值。
    new_omega = compute_new_omega()
    # 保存或更新 `omega1` 的值。
    omega1 += new_omega
    # 保存或更新 `omega2` 的值。
    omega2 -= new_omega

    # 返回当前函数的结果。
    return vel1, omega1, vel2, omega2


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `calculate_collision_matrix`。
def calculate_collision_matrix(positions, collision_threshold):
    # 下面开始文档字符串说明。
    """
    drone_col_matrix: set collided quadrotors' id to 1
    curr_drone_collisions: [i, j]
    distance_matrix: [i, j, dist]
    """
    # 保存或更新 `num_agents` 的值。
    num_agents = len(positions)
    # 保存或更新 `item_num` 的值。
    item_num = int(num_agents * (num_agents - 1) / 2)
    # 保存或更新 `count` 的值。
    count = int(0)

    # 保存或更新 `drone_col_matrix` 的值。
    drone_col_matrix = -1000 * np.ones(num_agents)
    # 保存或更新 `curr_drone_collisions` 的值。
    curr_drone_collisions = -1000. * np.ones((item_num, 2))
    # 保存或更新 `distance_matrix` 的值。
    distance_matrix = -1000. * np.ones((item_num, 3))

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(num_agents):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j in range(i + 1, num_agents):
            # 保存或更新 `distance_matrix[count]` 的值。
            distance_matrix[count] = [i, j,
                                      ((positions[i][0] - positions[j][0]) ** 2 +
                                       (positions[i][1] - positions[j][1]) ** 2 +
                                       (positions[i][2] - positions[j][2]) ** 2) ** 0.5]

            # 根据条件决定是否进入当前分支。
            if distance_matrix[count][2] <= collision_threshold:
                # 保存或更新 `drone_col_matrix[i]` 的值。
                drone_col_matrix[i] = 1
                # 保存或更新 `drone_col_matrix[j]` 的值。
                drone_col_matrix[j] = 1
                # 保存或更新 `curr_drone_collisions[count]` 的值。
                curr_drone_collisions[count] = [i, j]

            # 保存或更新 `count` 的值。
            count += 1

    # 返回当前函数的结果。
    return drone_col_matrix, curr_drone_collisions, distance_matrix


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `calculate_drone_proximity_penalties`。
def calculate_drone_proximity_penalties(distance_matrix, collision_falloff_threshold, dt, max_penalty, num_agents):
    # 保存或更新 `penalties` 的值。
    penalties = np.zeros(num_agents)
    # 保存或更新 `penalty_ratio` 的值。
    penalty_ratio = -max_penalty / collision_falloff_threshold
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, j, dist in distance_matrix:
        # 保存或更新 `penalty` 的值。
        penalty = penalty_ratio * dist + max_penalty
        # 保存或更新 `penalties[int(i)]` 的值。
        penalties[int(i)] += penalty
        # 保存或更新 `penalties[int(j)]` 的值。
        penalties[int(j)] += penalty

    # 返回当前函数的结果。
    return dt * penalties
