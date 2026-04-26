# 中文注释副本；原始文件：gym_art/quadrotor_multi/aerodynamics/downwash.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np


# 定义函数 `perform_downwash`。
def perform_downwash(drones_dyn, dt):
    # based on some data from Neural-Swarm: https://arxiv.org/pdf/2003.02992.pdf, Fig. 3
    # quadrotor weights: 34 grams
    # 0.5 m, force = 4 grams ; 0.4 m, force = 6 grams
    # 0.3 m, force = 8 grams ; 0.2 m, force = 10 grams
    # force function: f(x) = -20x + 14
    # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17, x in [0, 0.7]
    # Use cylinder to simulate the downwash area
    # The downwash area is a cylinder with radius of 2 arm ~ 10 cm and height of 1.0 m
    # 保存或更新 `xy_downwash` 的值。
    xy_downwash = 0.1
    # 保存或更新 `z_downwash` 的值。
    z_downwash = 0.7
    # get pos
    # 保存或更新 `dyns_pos` 的值。
    dyns_pos = np.array([d.pos for d in drones_dyn])
    # get z_axis
    # 保存或更新 `dyns_z_axis` 的值。
    dyns_z_axis = np.array([d.rot[:, -1] for d in drones_dyn])
    # drone num
    # 保存或更新 `dyns_num` 的值。
    dyns_num = len(drones_dyn)
    # 保存或更新 `applied_downwash_list` 的值。
    applied_downwash_list = np.zeros(dyns_num)
    # check if neighbors drones are within teh downwash areas, if yes, apply downwash
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(dyns_num):
        # 保存或更新 `z_axis` 的值。
        z_axis = dyns_z_axis[i]
        # 保存或更新 `neighbor_pos` 的值。
        neighbor_pos = dyns_pos - dyns_pos[i]
        # 保存或更新 `neighbor_pos_dist` 的值。
        neighbor_pos_dist = np.linalg.norm(neighbor_pos, axis=1)
        # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17
        # x in [0, 0.7], a(x) in [0.0, 0.41]
        # acc = (1 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.03, high=0.03)
        # 保存或更新 `acc` 的值。
        acc = (6 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.1, high=0.1)
        # 保存或更新 `acc` 的值。
        acc = np.maximum(1e-6, acc)

        # omega downwash given neighbor_pos_dist
        # 0.3 * (x - 1)^2 + random(-0.01, 0.01)
        # 保存或更新 `omega_downwash` 的值。
        omega_downwash = 0.3 * (neighbor_pos_dist - 1) ** 2 + np.random.uniform(low=-0.01, high=0.01)
        # 保存或更新 `omega_downwash` 的值。
        omega_downwash = np.maximum(1e-6, omega_downwash)

        # 保存或更新 `rel_dists_z` 的值。
        rel_dists_z = np.dot(neighbor_pos, z_axis)
        # 保存或更新 `rel_dists_xy` 的值。
        rel_dists_xy = np.sqrt(neighbor_pos_dist ** 2 - rel_dists_z ** 2)

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j in range(dyns_num):
            # 根据条件决定是否进入当前分支。
            if i == j:
                # 跳过本轮循环剩余逻辑，进入下一轮。
                continue

            # 根据条件决定是否进入当前分支。
            if -z_downwash < rel_dists_z[j] < 0 and rel_dists_xy[j] < xy_downwash:
                # 同时更新 `down_z_axis_norm`, `dir_omega_norm` 等变量。
                down_z_axis_norm, dir_omega_norm = get_vel_omega_norm(z_axis=z_axis)
                # 保存或更新 `drones_dyn[j].vel` 的值。
                drones_dyn[j].vel += acc[j] * down_z_axis_norm * dt
                # 保存或更新 `drones_dyn[j].omega` 的值。
                drones_dyn[j].omega += omega_downwash[j] * dir_omega_norm * dt
                # 保存或更新 `applied_downwash_list[j]` 的值。
                applied_downwash_list[j] = 1.0

    # 返回当前函数的结果。
    return applied_downwash_list


# 定义函数 `get_vel_omega_norm`。
def get_vel_omega_norm(z_axis):
    # vel_norm
    # 保存或更新 `noise_z_axis` 的值。
    noise_z_axis = z_axis + np.random.uniform(low=-0.1, high=0.1, size=3)
    # 保存或更新 `noise_z_axis_mag` 的值。
    noise_z_axis_mag = np.linalg.norm(noise_z_axis)
    # 执行这一行逻辑。
    noise_z_axis_norm = noise_z_axis / (noise_z_axis_mag + 1e-6 if noise_z_axis_mag == 0.0 else noise_z_axis_mag)
    # 保存或更新 `down_z_axis_norm` 的值。
    down_z_axis_norm = -1.0 * noise_z_axis_norm

    # omega norm
    # 保存或更新 `dir_omega` 的值。
    dir_omega = np.random.uniform(low=-1, high=1, size=3)
    # 保存或更新 `dir_omega_mag` 的值。
    dir_omega_mag = np.linalg.norm(dir_omega)
    # 执行这一行逻辑。
    dir_omega_norm = dir_omega / (dir_omega_mag + 1e-6 if dir_omega_mag == 0.0 else dir_omega_mag)

    # 返回当前函数的结果。
    return down_z_axis_norm, dir_omega_norm
