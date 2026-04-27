# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件处理机体、障碍物或房间边界的碰撞几何与碰撞后状态更新，是训练中安全相关奖励和终止判定的重要来源。
# 这里的测试只盯住障碍碰撞里的最基础一环：给定无人机位置、速度和圆柱障碍中心，
# 反弹法向和法向速度分量是否按预期算出来。

import numpy as np

from gym_art.quadrotor_multi.collisions.obstacles import compute_col_norm_and_new_vel_obst


# 这个构型把障碍中心放在无人机的右前方，
# 所以正确的碰撞法向应该是朝左后方的 45 度单位向量，法向速度也应该对应投影后的负值。
def test_compute_col_norm_and_new_vel_obst():
    quad_pos = np.array([0., 0., 0.])
    quad_vel = np.array([1., 0., 0.])

    obst_pos = np.array([0.5, 0.5, 5.])

    true_vnew = -np.sqrt(2) / 2.
    true_collision_norm = np.array([-np.sqrt(2) / 2., -np.sqrt(2) / 2., 0.])

    test_vnew, test_col_norm = compute_col_norm_and_new_vel_obst(pos=quad_pos, vel=quad_vel, obstacle_pos=obst_pos)
    assert np.around(test_vnew, decimals=6) == np.around(true_vnew, decimals=6)
    assert test_col_norm.all() == true_collision_norm.all()
    return


# 手工运行时的总入口。
def unit_test():
    test_compute_col_norm_and_new_vel_obst()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
