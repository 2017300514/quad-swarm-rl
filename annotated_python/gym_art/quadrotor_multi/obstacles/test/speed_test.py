# 中文注释副本；原始文件：gym_art/quadrotor_multi/obstacles/test/speed_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件只做一个很窄的性能回归检查：
# `get_cell_centers` 会在 obstacle map 采样时频繁调用，这里用 `timeit` 粗略盯住它的生成开销。

import timeit
import numpy as np


# 固定在一个 8x8、步长 1.0 的小网格上 benchmark，
# 目的是比较后续改动前后的相对时间，而不是给出通用性能结论。
def test_speed_get_cell_centers():
    SETUP_CODE = '''from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers'''

    TEST_CODE = '''get_cell_centers(obst_area_length=8.0, obst_area_width=8.0, grid_size=1.0)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e4))

    # print('times:   ', times)
    print('get_cell_centers - Mean Time:   ', np.mean(times[1:]))


# 给手工运行保留的薄封装。
def speed_test():
    test_speed_get_cell_centers()
    return


if __name__ == "__main__":
    test_speed_get_cell_centers()
