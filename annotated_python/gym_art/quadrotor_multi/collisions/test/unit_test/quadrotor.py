# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix


# 定义函数 `test_calculate_collision_matrix`。
def test_calculate_collision_matrix():
    # 保存或更新 `positions` 的值。
    positions = np.ones((8, 3))
    # 保存或更新 `positions[7][0]` 的值。
    positions[7][0] = 3
    # 保存或更新 `positions[7][1]` 的值。
    positions[7][1] = 3
    # 保存或更新 `positions[7][2]` 的值。
    positions[7][2] = 6
    # 保存或更新 `collision_threshold` 的值。
    collision_threshold = 0.2
    # 保存或更新 `num_agents` 的值。
    num_agents = 8

    # 保存或更新 `item_num` 的值。
    item_num = int(num_agents * (num_agents - 1) / 2)
    # 同时更新 `test_drone_col_matrix`, `test_curr_drone_collisions`, `test_distance_matrix` 等变量。
    test_drone_col_matrix, test_curr_drone_collisions, test_distance_matrix = \
        # 保存或更新 `calculate_collision_matrix(positions` 的值。
        calculate_collision_matrix(positions=positions, collision_threshold=collision_threshold)

    # 保存或更新 `true_drone_col_matrix` 的值。
    true_drone_col_matrix = -1000 * np.ones(len(positions))
    # 保存或更新 `true_curr_drone_collisions` 的值。
    true_curr_drone_collisions = -1000 * np.ones((item_num, 2))
    # 保存或更新 `true_distance_matrix` 的值。
    true_distance_matrix = -1000 * np.ones((item_num, 3))
    # 保存或更新 `count` 的值。
    count = 0
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(len(positions)):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for j in range(i + 1, len(positions)):
            # 保存或更新 `true_distance_matrix[count]` 的值。
            true_distance_matrix[count] = [i, j, np.linalg.norm(positions[i] - positions[j])]
            # 根据条件决定是否进入当前分支。
            if np.linalg.norm(positions[i] - positions[j]) <= collision_threshold:
                # 保存或更新 `true_drone_col_matrix[i]` 的值。
                true_drone_col_matrix[i] = 1
                # 保存或更新 `true_drone_col_matrix[j]` 的值。
                true_drone_col_matrix[j] = 1
                # 保存或更新 `true_curr_drone_collisions[count]` 的值。
                true_curr_drone_collisions[count] = [i, j]
            # 保存或更新 `count` 的值。
            count += 1

    # 保存或更新 `test_curr_drone_collisions` 的值。
    test_curr_drone_collisions = test_curr_drone_collisions.astype(int)
    # 保存或更新 `test_curr_drone_collisions` 的值。
    test_curr_drone_collisions = np.delete(test_curr_drone_collisions, np.unique(
        np.where(test_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    # 保存或更新 `true_curr_drone_collisions` 的值。
    true_curr_drone_collisions = true_curr_drone_collisions.astype(int)
    # 保存或更新 `true_curr_drone_collisions` 的值。
    true_curr_drone_collisions = np.delete(true_curr_drone_collisions, np.unique(
        np.where(true_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    # 断言当前条件成立，用于保护运行假设。
    assert test_drone_col_matrix.all() == true_drone_col_matrix.all()

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(len(test_curr_drone_collisions)):
        # 根据条件决定是否进入当前分支。
        if test_curr_drone_collisions[i] not in true_curr_drone_collisions:
            # 主动抛出异常以中止或提示错误。
            raise ValueError

    # 断言当前条件成立，用于保护运行假设。
    assert test_distance_matrix.all() == true_distance_matrix.all()

    # print('drone_col_matrix:    ', drone_col_matrix)
    # print('curr_drone_collisions:    ', curr_drone_collisions)
    # print('distance_matrix:    ', distance_matrix)

    # 返回当前函数的结果。
    return


# 定义函数 `unit_test`。
def unit_test():
    # 调用 `test_calculate_collision_matrix` 执行当前处理。
    test_calculate_collision_matrix()
    # 调用 `print` 执行当前处理。
    print('Pass unit test!')
    # 返回当前函数的结果。
    return


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 调用 `unit_test` 执行当前处理。
    unit_test()
