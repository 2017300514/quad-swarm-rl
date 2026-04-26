# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/unit_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_cell_centers


# 定义函数 `test_get_surround_sdfs`。
def test_get_surround_sdfs():
    # 保存或更新 `quad_poses` 的值。
    quad_poses = np.array([[0., 0.]])
    # 保存或更新 `obst_poses` 的值。
    obst_poses = np.array([[0.2, 0.]])
    # 保存或更新 `quads_sdf_obs` 的值。
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    # get_surround_sdfs
    # 保存或更新 `dist` 的值。
    dist = []
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, x in enumerate([-0.1, 0, 0.1]):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j, y in enumerate([-0.1, 0, 0.1]):
            # 保存或更新 `tmp` 的值。
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            # 调用 `append` 执行当前处理。
            dist.append(tmp)

    # 保存或更新 `test_res` 的值。
    test_res = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius=0.3, resolution=0.1)
    # 保存或更新 `true_res` 的值。
    true_res = np.array(dist)
    # 断言当前条件成立，用于保护运行假设。
    assert test_res.all() == true_res.all()
    # 返回当前函数的结果。
    return


# 定义函数 `test_collision_detection`。
def test_collision_detection():
    # 保存或更新 `quad_poses` 的值。
    quad_poses = np.array([[0., 0.]])
    # 保存或更新 `obst_poses` 的值。
    obst_poses = np.array([[0.2, 0.]])
    # collision_detection
    # 保存或更新 `quad_collisions` 的值。
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3)
    # 保存或更新 `test_res` 的值。
    test_res = np.where(quad_collisions > -1)[0]
    # 保存或更新 `true_res` 的值。
    true_res = np.array([0])
    # 断言当前条件成立，用于保护运行假设。
    assert test_res.all() == true_res.all()
    # 返回当前函数的结果。
    return


# 定义函数 `test_get_cell_centers`。
def test_get_cell_centers():
    # 保存或更新 `obst_area_length` 的值。
    obst_area_length = 8.0
    # 保存或更新 `obst_area_width` 的值。
    obst_area_width = 8.0
    # 保存或更新 `grid_size` 的值。
    grid_size = 1.0
    # 保存或更新 `test_res` 的值。
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    # 保存或更新 `true_res` 的值。
    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    # 断言当前条件成立，用于保护运行假设。
    assert test_res.all() == true_res.all()
    # 返回当前函数的结果。
    return


# 定义函数 `unit_test`。
def unit_test():
    # 调用 `test_get_surround_sdfs` 执行当前处理。
    test_get_surround_sdfs()
    # 调用 `test_collision_detection` 执行当前处理。
    test_collision_detection()
    # 调用 `test_get_cell_centers` 执行当前处理。
    test_get_cell_centers()
    # 调用 `print` 执行当前处理。
    print('Pass unit test!')
    # 返回当前函数的结果。
    return


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 调用 `unit_test` 执行当前处理。
    unit_test()
