# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.collisions.obstacles import compute_col_norm_and_new_vel_obst


# 定义函数 `test_compute_col_norm_and_new_vel_obst`。
def test_compute_col_norm_and_new_vel_obst():
    # 保存或更新 `quad_pos` 的值。
    quad_pos = np.array([0., 0., 0.])
    # 保存或更新 `quad_vel` 的值。
    quad_vel = np.array([1., 0., 0.])

    # 保存或更新 `obst_pos` 的值。
    obst_pos = np.array([0.5, 0.5, 5.])

    # 保存或更新 `true_vnew` 的值。
    true_vnew = -np.sqrt(2) / 2.
    # 保存或更新 `true_collision_norm` 的值。
    true_collision_norm = np.array([-np.sqrt(2) / 2., -np.sqrt(2) / 2., 0.])

    # 同时更新 `test_vnew`, `test_col_norm` 等变量。
    test_vnew, test_col_norm = compute_col_norm_and_new_vel_obst(pos=quad_pos, vel=quad_vel, obstacle_pos=obst_pos)
    # 断言当前条件成立，用于保护运行假设。
    assert np.around(test_vnew, decimals=6) == np.around(true_vnew, decimals=6)
    # 断言当前条件成立，用于保护运行假设。
    assert test_col_norm.all() == true_collision_norm.all()
    # 返回当前函数的结果。
    return


# 定义函数 `unit_test`。
def unit_test():
    # 调用 `test_compute_col_norm_and_new_vel_obst` 执行当前处理。
    test_compute_col_norm_and_new_vel_obst()
    # 调用 `print` 执行当前处理。
    print('Pass unit test!')
    # 返回当前函数的结果。
    return


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 调用 `unit_test` 执行当前处理。
    unit_test()
