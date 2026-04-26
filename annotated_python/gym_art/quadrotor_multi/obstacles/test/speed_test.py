# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/speed_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import timeit
import numpy as np


# 定义函数 `test_speed_get_cell_centers`。
def test_speed_get_cell_centers():
    # 保存或更新 `SETUP_CODE` 的值。
    SETUP_CODE = '''from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers'''

    # 保存或更新 `TEST_CODE` 的值。
    TEST_CODE = '''get_cell_centers(obst_area_length=8.0, obst_area_width=8.0, grid_size=1.0)'''

    # 保存或更新 `times` 的值。
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e4))

    # print('times:   ', times)
    # 调用 `print` 执行当前处理。
    print('get_cell_centers - Mean Time:   ', np.mean(times[1:]))


# 定义函数 `speed_test`。
def speed_test():
    # 调用 `test_speed_get_cell_centers` 执行当前处理。
    test_speed_get_cell_centers()
    # 返回当前函数的结果。
    return


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 调用 `test_speed_get_cell_centers` 执行当前处理。
    test_speed_get_cell_centers()
