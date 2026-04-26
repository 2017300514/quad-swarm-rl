# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件处理机体、障碍物或房间边界的碰撞几何与碰撞后状态更新，是训练中安全相关奖励和终止判定的重要来源。
# 这里的输出会回流到动力学状态、奖励项和碰撞统计中。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import numpy as np

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.collisions.obstacles import compute_col_norm_and_new_vel_obst


# `test_compute_col_norm_and_new_vel_obst` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def test_compute_col_norm_and_new_vel_obst():
    quad_pos = np.array([0., 0., 0.])
    quad_vel = np.array([1., 0., 0.])

    obst_pos = np.array([0.5, 0.5, 5.])

    true_vnew = -np.sqrt(2) / 2.
    true_collision_norm = np.array([-np.sqrt(2) / 2., -np.sqrt(2) / 2., 0.])

    test_vnew, test_col_norm = compute_col_norm_and_new_vel_obst(pos=quad_pos, vel=quad_vel, obstacle_pos=obst_pos)
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert np.around(test_vnew, decimals=6) == np.around(true_vnew, decimals=6)
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert test_col_norm.all() == true_collision_norm.all()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


# `unit_test` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def unit_test():
    test_compute_col_norm_and_new_vel_obst()
    print('Pass unit test!')
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


if __name__ == "__main__":
    unit_test()
