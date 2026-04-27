# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件处理机体、障碍物或房间边界的碰撞几何与碰撞后状态更新，是训练中安全相关奖励和终止判定的重要来源。
# 这里的测试针对多机碰撞检测的底层输出：
# 哪些无人机被标成发生碰撞、哪些 agent pair 被记录下来，以及两两距离表是否完整。

import numpy as np

from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix


# 这里故意让前 7 架无人机挤在同一点，最后 1 架远离，
# 这样真值矩阵非常容易手算，适合检查 pairwise collision 扫描是否仍然正确。
def test_calculate_collision_matrix():
    positions = np.ones((8, 3))
    positions[7][0] = 3
    positions[7][1] = 3
    positions[7][2] = 6
    collision_threshold = 0.2
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

    assert test_drone_col_matrix.all() == true_drone_col_matrix.all()

    # pair 列表不要求排序完全一致，但要求检测出的每一对都真的在手算真值里。
    for i in range(len(test_curr_drone_collisions)):
        if test_curr_drone_collisions[i] not in true_curr_drone_collisions:
            raise ValueError

    assert test_distance_matrix.all() == true_distance_matrix.all()

    # print('drone_col_matrix:    ', drone_col_matrix)
    # print('curr_drone_collisions:    ', curr_drone_collisions)
    # print('distance_matrix:    ', distance_matrix)

    return


# 手工运行时的总入口。
def unit_test():
    test_calculate_collision_matrix()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
