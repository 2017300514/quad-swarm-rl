#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件放的是障碍物模块最底层的数值工具：
# 3x3 局部 SDF 风格观测、无人机与障碍物碰撞判定，以及障碍生成网格的中心点枚举。
# 上游输入主要是无人机 xy 坐标和障碍物中心坐标；下游消费者包括 `MultiObstacles`、多机环境 reset 逻辑和障碍单元测试。

import numpy as np
from numba import njit

@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)
    # 这里实现的是论文里那种固定 9 维、与障碍数量无关的局部障碍表示。
    # 对每架无人机，在自身周围 `resolution` 间隔的 3x3 网格上，记录每个格点到最近障碍边界的距离。

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                # `g_id` 固定了 3x3 网格展平成 9 维向量时的槽位顺序，
                # 这样模型始终能把同一位置的局部几何语义对应到同一输入维度。
                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    return quads_sdf_obs


@njit
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    # 这里返回的是“每架无人机撞到的第一个障碍物编号”，未碰撞则记为 -1。
    # 上层再把它转成碰撞 agent 列表和 agent->obstacle 的映射。
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    # 这个函数把矩形障碍生成区域离散成规则网格中心点。
    # 场景生成器随后会在这些候选中心上采样，决定障碍物实际落在哪些格子里。
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
