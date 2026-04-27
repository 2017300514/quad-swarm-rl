# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/unit_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件给 obstacle 工具函数做最小单元测试。
# 它对应前面已经注释过的 `obstacles/utils.py`，重点检查 9 维局部 SDF、无人机-障碍碰撞索引，
# 以及规则栅格中心枚举这三条最基础的几何约定有没有偏。

import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_cell_centers


# 这里直接手算 3x3 局部网格上每个采样点到障碍圆的距离，
# 用来对拍 `get_surround_sdfs` 是否仍然遵守论文里 9 维 SDF 观测的几何定义。
def test_get_surround_sdfs():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    # get_surround_sdfs
    dist = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        for j, y in enumerate([-0.1, 0, 0.1]):
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            dist.append(tmp)

    test_res = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius=0.3, resolution=0.1)
    true_res = np.array(dist)
    assert test_res.all() == true_res.all()
    return


# 这个测试只看“哪一架无人机被标成撞上障碍”，不关心后续动力学修正。
def test_collision_detection():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    # collision_detection
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert test_res.all() == true_res.all()
    return


# obstacle map 的自由单元中心是很多 spawn/goal 采样逻辑的基础，
# 所以这里验证它的枚举顺序和坐标偏移是否与测试手算结果一致。
def test_get_cell_centers():
    obst_area_length = 8.0
    obst_area_width = 8.0
    grid_size = 1.0
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    assert test_res.all() == true_res.all()
    return


# 手工执行时的总入口，方便不走 pytest 也能直接把三项几何检查跑一遍。
def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_get_cell_centers()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
