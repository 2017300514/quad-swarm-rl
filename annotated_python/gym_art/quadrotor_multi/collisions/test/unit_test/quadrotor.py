# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件处理机体、障碍物或房间边界的碰撞几何与碰撞后状态更新，是训练中安全相关奖励和终止判定的重要来源。
# 这里的输出会回流到动力学状态、奖励项和碰撞统计中。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import numpy as np

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix


# `test_calculate_collision_matrix` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def test_calculate_collision_matrix():
    positions = np.ones((8, 3))
    positions[7][0] = 3
    positions[7][1] = 3
    positions[7][2] = 6
    # 实际碰撞阈值不是裸半径，而是按机臂长度缩放得到，确保不同尺寸动力学参数下碰撞判定仍有物理一致性。
    collision_threshold = 0.2
    # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
    num_agents = 8

    item_num = int(num_agents * (num_agents - 1) / 2)
    test_drone_col_matrix, test_curr_drone_collisions, test_distance_matrix = \
        calculate_collision_matrix(positions=positions, collision_threshold=collision_threshold)

    true_drone_col_matrix = -1000 * np.ones(len(positions))
    true_curr_drone_collisions = -1000 * np.ones((item_num, 2))
    true_distance_matrix = -1000 * np.ones((item_num, 3))
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            true_distance_matrix[count] = [i, j, np.linalg.norm(positions[i] - positions[j])]
            if np.linalg.norm(positions[i] - positions[j]) <= collision_threshold:
                true_drone_col_matrix[i] = 1
                true_drone_col_matrix[j] = 1
                true_curr_drone_collisions[count] = [i, j]
            count += 1

    test_curr_drone_collisions = test_curr_drone_collisions.astype(int)
    test_curr_drone_collisions = np.delete(test_curr_drone_collisions, np.unique(
        np.where(test_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    true_curr_drone_collisions = true_curr_drone_collisions.astype(int)
    true_curr_drone_collisions = np.delete(true_curr_drone_collisions, np.unique(
        np.where(true_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_drone_col_matrix.all() == true_drone_col_matrix.all()

    for i in range(len(test_curr_drone_collisions)):
        if test_curr_drone_collisions[i] not in true_curr_drone_collisions:
            raise ValueError

    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_distance_matrix.all() == true_distance_matrix.all()

    # print('drone_col_matrix:    ', drone_col_matrix)
    # print('curr_drone_collisions:    ', curr_drone_collisions)
    # print('distance_matrix:    ', distance_matrix)

    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


# `unit_test` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def unit_test():
    test_calculate_collision_matrix()
    print('Pass unit test!')
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


if __name__ == "__main__":
    unit_test()
