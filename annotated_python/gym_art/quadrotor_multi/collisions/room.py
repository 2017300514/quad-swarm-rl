# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/room.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numba as nb
import numpy as np
from numba import njit


# 定义函数 `perform_collision_with_wall`。
def perform_collision_with_wall(drone_dyn, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                # 保存或更新 `lowest_speed` 的值。
                                lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # Decrease drone's speed after collision with wall
    # 保存或更新 `drone_speed` 的值。
    drone_speed = np.linalg.norm(drone_dyn.vel)
    # 保存或更新 `real_speed` 的值。
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    # 保存或更新 `real_speed` 的值。
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    # 保存或更新 `drone_pos` 的值。
    drone_pos = drone_dyn.pos
    # 执行这一行逻辑。
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    # 执行这一行逻辑。
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    # 保存或更新 `direction` 的值。
    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    # 根据条件决定是否进入当前分支。
    if x_list[0]:
        # 保存或更新 `direction[0]` 的值。
        direction[0] = np.random.uniform(low=0.1, high=1.0)
    # 当上一分支不满足时，继续判断新的条件。
    elif x_list[1]:
        # 保存或更新 `direction[0]` 的值。
        direction[0] = np.random.uniform(low=-1.0, high=-0.1)

    # 根据条件决定是否进入当前分支。
    if y_list[0]:
        # 保存或更新 `direction[1]` 的值。
        direction[1] = np.random.uniform(low=0.1, high=1.0)
    # 当上一分支不满足时，继续判断新的条件。
    elif y_list[1]:
        # 保存或更新 `direction[1]` 的值。
        direction[1] = np.random.uniform(low=-1.0, high=-0.1)

    # 保存或更新 `direction[2]` 的值。
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)

    # 保存或更新 `direction_mag` 的值。
    direction_mag = np.linalg.norm(direction)
    # 保存或更新 `direction_norm` 的值。
    direction_norm = direction / (direction_mag + eps)

    # 保存或更新 `drone_dyn.vel` 的值。
    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    # 保存或更新 `omega_max` 的值。
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    # 保存或更新 `new_omega` 的值。
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    # 保存或更新 `new_omega` 的值。
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    # 保存或更新 `new_omega_mag` 的值。
    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    # 保存或更新 `new_omega` 的值。
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    # 保存或更新 `drone_dyn.omega` 的值。
    drone_dyn.omega += new_omega


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `perform_collision_with_wall_numba`。
def perform_collision_with_wall_numba(vel, pos, omega, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                      # 保存或更新 `lowest_speed` 的值。
                                      lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # Decrease drone's speed after collision with wall
    # 保存或更新 `drone_speed` 的值。
    drone_speed = np.linalg.norm(vel)
    # 保存或更新 `real_speed` 的值。
    real_speed = nb.random.uniform(damp_low_speed_ratio * drone_speed, damp_high_speed_ratio * drone_speed)
    # 保存或更新 `real_speed` 的值。
    real_speed = np.clip(real_speed, 0.1, 6.0)

    # 保存或更新 `drone_pos` 的值。
    drone_pos = pos
    # 执行这一行逻辑。
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    # 执行这一行逻辑。
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    # 保存或更新 `direction` 的值。
    direction = np.random.uniform(-1.0, 1.0, size=(3,))
    # 根据条件决定是否进入当前分支。
    if x_list[0]:
        # 保存或更新 `direction[0]` 的值。
        direction[0] = np.random.uniform(0.1, 1.0)
    # 当上一分支不满足时，继续判断新的条件。
    elif x_list[1]:
        # 保存或更新 `direction[0]` 的值。
        direction[0] = np.random.uniform(-1.0, -0.1)

    # 根据条件决定是否进入当前分支。
    if y_list[0]:
        # 保存或更新 `direction[1]` 的值。
        direction[1] = np.random.uniform(0.1, 1.0)
    # 当上一分支不满足时，继续判断新的条件。
    elif y_list[1]:
        # 保存或更新 `direction[1]` 的值。
        direction[1] = np.random.uniform(-1.0, -0.1)

    # 保存或更新 `direction[2]` 的值。
    direction[2] = np.random.uniform(-1.0, -0.5)

    # 保存或更新 `direction_mag` 的值。
    direction_mag = np.linalg.norm(direction)
    # 保存或更新 `direction_norm` 的值。
    direction_norm = direction / (direction_mag + eps)

    # 保存或更新 `vel` 的值。
    vel = real_speed * direction_norm

    # Random forces for omega
    # 保存或更新 `omega_max` 的值。
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    # 保存或更新 `new_omega` 的值。
    new_omega = np.random.uniform(-1, 1, size=(3,))  # random direction in 3D space
    # 保存或更新 `new_omega` 的值。
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    # 保存或更新 `new_omega_mag` 的值。
    new_omega_mag = np.random.uniform(omega_max / 2, omega_max)  # random magnitude of the force
    # 保存或更新 `new_omega` 的值。
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    # 保存或更新 `omega` 的值。
    omega += new_omega

    # 返回当前函数的结果。
    return vel, omega


# 定义函数 `perform_collision_with_ceiling`。
def perform_collision_with_ceiling(drone_dyn, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                   # 保存或更新 `lowest_speed` 的值。
                                   lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # 保存或更新 `drone_speed` 的值。
    drone_speed = np.linalg.norm(drone_dyn.vel)
    # 保存或更新 `real_speed` 的值。
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    # 保存或更新 `real_speed` 的值。
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    # 保存或更新 `direction` 的值。
    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    # 保存或更新 `direction[2]` 的值。
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)
    # 保存或更新 `direction_mag` 的值。
    direction_mag = np.linalg.norm(direction)
    # 保存或更新 `direction_norm` 的值。
    direction_norm = direction / (direction_mag + eps)

    # 保存或更新 `drone_dyn.vel` 的值。
    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    # 保存或更新 `omega_max` 的值。
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    # 保存或更新 `new_omega` 的值。
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    # 保存或更新 `new_omega` 的值。
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    # 保存或更新 `new_omega_mag` 的值。
    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    # 保存或更新 `new_omega` 的值。
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    # 保存或更新 `drone_dyn.omega` 的值。
    drone_dyn.omega += new_omega
