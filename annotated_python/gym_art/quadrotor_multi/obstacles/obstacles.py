# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection


# 定义类 `MultiObstacles`。
class MultiObstacles:
    # 定义函数 `__init__`。
    def __init__(self, obstacle_size=1.0, quad_radius=0.046):
        # 保存或更新 `size` 的值。
        self.size = obstacle_size
        # 保存或更新 `obstacle_radius` 的值。
        self.obstacle_radius = obstacle_size / 2.0
        # 保存或更新 `quad_radius` 的值。
        self.quad_radius = quad_radius
        # 保存或更新 `pos_arr` 的值。
        self.pos_arr = []
        # 保存或更新 `resolution` 的值。
        self.resolution = 0.1

    # 定义函数 `reset`。
    def reset(self, obs, quads_pos, pos_arr):
        # 保存或更新 `pos_arr` 的值。
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        # 保存或更新 `quads_sdf_obs` 的值。
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        # 保存或更新 `quads_sdf_obs` 的值。
        quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                          quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                          resolution=self.resolution)

        # 保存或更新 `obs` 的值。
        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        # 返回当前函数的结果。
        return obs

    # 定义函数 `step`。
    def step(self, obs, quads_pos):
        # 保存或更新 `quads_sdf_obs` 的值。
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        # 保存或更新 `quads_sdf_obs` 的值。
        quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                          quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                          resolution=self.resolution)

        # 保存或更新 `obs` 的值。
        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        # 返回当前函数的结果。
        return obs

    # 定义函数 `collision_detection`。
    def collision_detection(self, pos_quads):
        # 保存或更新 `quad_collisions` 的值。
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        # 保存或更新 `collided_quads_id` 的值。
        collided_quads_id = np.where(quad_collisions > -1)[0]
        # 保存或更新 `collided_obstacles_id` 的值。
        collided_obstacles_id = quad_collisions[collided_quads_id]
        # 保存或更新 `quad_obst_pair` 的值。
        quad_obst_pair = {}
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, key in enumerate(collided_quads_id):
            # 保存或更新 `quad_obst_pair[key]` 的值。
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        # 返回当前函数的结果。
        return collided_quads_id, quad_obst_pair
