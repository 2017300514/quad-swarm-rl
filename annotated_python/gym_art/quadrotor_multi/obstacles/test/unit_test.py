# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/unit_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import numpy as np

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_cell_centers


# `test_get_surround_sdfs` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
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
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_res.all() == true_res.all()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


# `test_collision_detection` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def test_collision_detection():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    # collision_detection
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_res.all() == true_res.all()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


# `test_get_cell_centers` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def test_get_cell_centers():
    obst_area_length = 8.0
    obst_area_width = 8.0
    grid_size = 1.0
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_res.all() == true_res.all()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


# `unit_test` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_get_cell_centers()
    print('Pass unit test!')
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


if __name__ == "__main__":
    unit_test()
