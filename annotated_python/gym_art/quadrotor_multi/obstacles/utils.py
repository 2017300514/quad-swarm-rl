# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
from numba import njit

# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `get_surround_sdfs`。
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    # 保存或更新 `sdf_map` 的值。
    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    # 保存或更新 `sdf_map` 的值。
    sdf_map *= resolution

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, q_pos in enumerate(quad_poses):
        # 同时更新 `q_pos_x`, `q_pos_y` 等变量。
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                # 保存或更新 `grid_pos` 的值。
                grid_pos = np.array([g_x, g_y])

                # 保存或更新 `min_dist` 的值。
                min_dist = 100.0
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for o_pos in obst_poses:
                    # 保存或更新 `dist` 的值。
                    dist = np.linalg.norm(grid_pos - o_pos)
                    # 根据条件决定是否进入当前分支。
                    if dist < min_dist:
                        # 保存或更新 `min_dist` 的值。
                        min_dist = dist

                # 保存或更新 `g_id` 的值。
                g_id = g_i * 3 + g_j
                # 保存或更新 `quads_sdf_obs[i, g_id]` 的值。
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    # 返回当前函数的结果。
    return quads_sdf_obs


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `collision_detection`。
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    # 保存或更新 `quad_num` 的值。
    quad_num = len(quad_poses)
    # 保存或更新 `collide_threshold` 的值。
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    # 保存或更新 `quad_collisions` 的值。
    quad_collisions = -1 * np.ones(quad_num)
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, q_pos in enumerate(quad_poses):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j, o_pos in enumerate(obst_poses):
            # 保存或更新 `dist` 的值。
            dist = np.linalg.norm(q_pos - o_pos)
            # 根据条件决定是否进入当前分支。
            if dist <= collide_threshold:
                # 保存或更新 `quad_collisions[i]` 的值。
                quad_collisions[i] = j
                # 提前结束当前循环。
                break

    # 返回当前函数的结果。
    return quad_collisions


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `get_cell_centers`。
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    # 保存或更新 `count` 的值。
    count = 0
    # 保存或更新 `i_len` 的值。
    i_len = obst_area_length / grid_size
    # 保存或更新 `j_len` 的值。
    j_len = obst_area_width / grid_size
    # 保存或更新 `cell_centers` 的值。
    cell_centers = np.zeros((int(i_len * j_len), 2))
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in np.arange(0, obst_area_length, grid_size):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            # 保存或更新 `cell_centers[count][0]` 的值。
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            # 保存或更新 `cell_centers[count][1]` 的值。
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            # 保存或更新 `count` 的值。
            count += 1

    # 返回当前函数的结果。
    return cell_centers


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 导入当前模块依赖。
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    # 调用 `unit_test` 执行当前处理。
    unit_test()
    # 调用 `speed_test` 执行当前处理。
    speed_test()
