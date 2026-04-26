# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/speed_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import timeit
import numpy as np


# `test_speed_get_cell_centers` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def test_speed_get_cell_centers():
    SETUP_CODE = '''from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers'''

    TEST_CODE = '''get_cell_centers(obst_area_length=8.0, obst_area_width=8.0, grid_size=1.0)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e4))

    # print('times:   ', times)
    print('get_cell_centers - Mean Time:   ', np.mean(times[1:]))


# `speed_test` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def speed_test():
    test_speed_get_cell_centers()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return


if __name__ == "__main__":
    test_speed_get_cell_centers()
