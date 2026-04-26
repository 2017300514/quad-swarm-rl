#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是多机环境里的障碍物观测包装层。
# 上游输入来自场景生成阶段给出的障碍物中心坐标，以及 `QuadrotorEnvMulti` 每一步同步过来的无人机位置；
# 下游输出是拼接到 agent 观测尾部的 9 维局部障碍特征，以及“哪架无人机撞上了哪个障碍”的索引结果。
# 结合 `obstacles/utils.py` 一起看时，可以把这里理解为障碍物模块的环境接口层。

import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection


class MultiObstacles:
    # 这个类长期保存当前 episode 的障碍物布局。
    # 多机环境在 reset 时创建它，在每一步 step 中复用它来生成障碍观测和检测障碍碰撞。
    def __init__(self, obstacle_size=1.0, quad_radius=0.046):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        # 0.1m 分辨率对应论文里 3x3 局部 SDF 风格障碍观测的采样步长。
        self.resolution = 0.1

    def reset(self, obs, quads_pos, pos_arr):
        # reset 时先冻结本局障碍物中心坐标。
        # 后续 step 不再重新生成布局，只是基于当前无人机位置不断重算局部障碍观测。
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        # 每架无人机都拿到固定 9 维障碍观测：
        # 以自身 xy 为中心，在 3x3 网格上查询到最近障碍边界的距离。
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                          quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                          resolution=self.resolution)

        # 障碍观测直接拼到已有自观测/邻居观测后面。
        # 模型侧会按固定切片把这一段送进障碍编码器。
        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos):
        # step 阶段不改障碍布局，只根据最新无人机位置重算局部障碍距离。
        # 这让障碍物观测与主环境里的位置推进保持逐步同步。
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                          quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                          resolution=self.resolution)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        # 碰撞检测只看 xy 平面距离，因为这一版障碍是柱状/圆盘式占据。
        # 返回值不是布尔矩阵，而是“发生碰撞的无人机 id”以及“它撞到的障碍 id”，方便上层统计奖励和物理修正。
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
