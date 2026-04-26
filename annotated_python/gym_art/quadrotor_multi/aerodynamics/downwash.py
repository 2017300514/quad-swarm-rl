import numpy as np

# 这个模块给多机环境补充一个简化的下洗流模型。
# 它夹在动力学状态更新链路里，用经验公式近似“上方无人机把下方邻机往下推、姿态也扰乱”的效应。


def perform_downwash(drones_dyn, dt):
    # 遍历所有无人机对，只对“位于正下方圆柱区域内”的邻机施加额外扰动。
    # 返回的 0/1 标记会被环境上层拿去记录哪些 agent 当前受到了 downwash。
    # based on some data from Neural-Swarm: https://arxiv.org/pdf/2003.02992.pdf, Fig. 3
    # quadrotor weights: 34 grams
    # 0.5 m, force = 4 grams ; 0.4 m, force = 6 grams
    # 0.3 m, force = 8 grams ; 0.2 m, force = 10 grams
    # force function: f(x) = -20x + 14
    # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17, x in [0, 0.7]
    # Use cylinder to simulate the downwash area
    # The downwash area is a cylinder with radius of 2 arm ~ 10 cm and height of 1.0 m
    xy_downwash = 0.1
    z_downwash = 0.7
    # 把位置和机体 z 轴先抽出来，后面做批量相对几何判断。
    dyns_pos = np.array([d.pos for d in drones_dyn])
    dyns_z_axis = np.array([d.rot[:, -1] for d in drones_dyn])
    dyns_num = len(drones_dyn)
    applied_downwash_list = np.zeros(dyns_num)
    # 如果某架机体落在另一架机体的下洗圆柱区内，就根据相对距离衰减出额外加速度和角速度扰动。
    for i in range(dyns_num):
        z_axis = dyns_z_axis[i]
        neighbor_pos = dyns_pos - dyns_pos[i]
        neighbor_pos_dist = np.linalg.norm(neighbor_pos, axis=1)
        # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17
        # x in [0, 0.7], a(x) in [0.0, 0.41]
        # acc = (1 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.03, high=0.03)
        acc = (6 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.1, high=0.1)
        acc = np.maximum(1e-6, acc)

        # omega downwash given neighbor_pos_dist
        # 0.3 * (x - 1)^2 + random(-0.01, 0.01)
        omega_downwash = 0.3 * (neighbor_pos_dist - 1) ** 2 + np.random.uniform(low=-0.01, high=0.01)
        omega_downwash = np.maximum(1e-6, omega_downwash)

        rel_dists_z = np.dot(neighbor_pos, z_axis)
        rel_dists_xy = np.sqrt(neighbor_pos_dist ** 2 - rel_dists_z ** 2)

        for j in range(dyns_num):
            if i == j:
                continue

            if -z_downwash < rel_dists_z[j] < 0 and rel_dists_xy[j] < xy_downwash:
                # 这里直接改写下方无人机的线速度和角速度，相当于在本步积分里叠加了一个气动扰动项。
                down_z_axis_norm, dir_omega_norm = get_vel_omega_norm(z_axis=z_axis)
                drones_dyn[j].vel += acc[j] * down_z_axis_norm * dt
                drones_dyn[j].omega += omega_downwash[j] * dir_omega_norm * dt
                applied_downwash_list[j] = 1.0

    return applied_downwash_list


def get_vel_omega_norm(z_axis):
    # 速度方向基本沿上机喷流的反向，但加入一点噪声，避免训练里每次命中都是完全确定性的推挤。
    noise_z_axis = z_axis + np.random.uniform(low=-0.1, high=0.1, size=3)
    noise_z_axis_mag = np.linalg.norm(noise_z_axis)
    noise_z_axis_norm = noise_z_axis / (noise_z_axis_mag + 1e-6 if noise_z_axis_mag == 0.0 else noise_z_axis_mag)
    down_z_axis_norm = -1.0 * noise_z_axis_norm

    # 角速度方向不做精确流体建模，只提供一个随机扰动方向，让姿态也感受到不稳定性。
    dir_omega = np.random.uniform(low=-1, high=1, size=3)
    dir_omega_mag = np.linalg.norm(dir_omega)
    dir_omega_norm = dir_omega / (dir_omega_mag + 1e-6 if dir_omega_mag == 0.0 else dir_omega_mag)

    return down_z_axis_norm, dir_omega_norm
