# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/test/speed_test/quadrotor.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import sys
import timeit
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.collisions.quadrotors import compute_col_norm_and_new_velocities
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


# 定义函数 `normal_perform_collision_between_drones`。
def normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    # 同时更新 `v1new`, `v2new`, `collision_norm` 等变量。
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
    # 保存或更新 `vel_change` 的值。
    vel_change = (v2new - v1new) * collision_norm
    # 保存或更新 `dyn1_vel_shift` 的值。
    dyn1_vel_shift = vel_change
    # 保存或更新 `dyn2_vel_shift` 的值。
    dyn2_vel_shift = -vel_change

    # Make sure new vel direction would be opposite to the original vel direction
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for _ in range(3):
        # 保存或更新 `cons_rand_val` 的值。
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        # 保存或更新 `vel1_noise` 的值。
        vel1_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        # 保存或更新 `vel2_noise` 的值。
        vel2_noise = -cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)

        # 保存或更新 `dyn1_vel_shift` 的值。
        dyn1_vel_shift = vel_change + vel1_noise
        # 保存或更新 `dyn2_vel_shift` 的值。
        dyn2_vel_shift = -vel_change + vel2_noise

        # 保存或更新 `dyn1_new_vel_dir` 的值。
        dyn1_new_vel_dir = np.dot(vel1 + dyn1_vel_shift, collision_norm)
        # 保存或更新 `dyn2_new_vel_dir` 的值。
        dyn2_new_vel_dir = np.dot(vel2 + dyn2_vel_shift, collision_norm)

        # 根据条件决定是否进入当前分支。
        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            # 提前结束当前循环。
            break

    # Get new vel
    # 保存或更新 `max_vel_magn` 的值。
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    # 保存或更新 `vel1` 的值。
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    # 保存或更新 `vel2` 的值。
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    # Get new omega
    # 保存或更新 `new_omega` 的值。
    new_omega = compute_new_omega()
    # 保存或更新 `omega1` 的值。
    omega1 += new_omega
    # 保存或更新 `omega2` 的值。
    omega2 -= new_omega

    # 返回当前函数的结果。
    return vel1, omega1, vel2, omega2


# 定义函数 `test_perform_collision_between_drones`。
def test_perform_collision_between_drones():
    # 保存或更新 `SETUP_CODE` 的值。
    SETUP_CODE = '''from __main__ import normal_perform_collision_between_drones; import numpy as np'''

    # 保存或更新 `TEST_CODE` 的值。
    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    # timeit.repeat statement
    # 保存或更新 `times` 的值。
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e3))

    # 调用 `print` 执行当前处理。
    print('Speed: perform_collision_between_drones')
    # 调用 `print` 执行当前处理。
    print('Normal:   ', times)
    # 调用 `print` 执行当前处理。
    print('Mean:   ', np.mean(times))


    # 保存或更新 `SETUP_CODE` 的值。
    SETUP_CODE = '''from gym_art.quadrotor_multi.collisions.quadrotors import perform_collision_between_drones; import numpy as np'''

    # 保存或更新 `TEST_CODE` 的值。
    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    # 保存或更新 `times` 的值。
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=6,
                          number=int(1e3))

    # 调用 `print` 执行当前处理。
    print('Optimized:   ', times)
    # 调用 `print` 执行当前处理。
    print('Mean:   ', np.mean(times[1:]))


# 定义函数 `speed_test`。
def speed_test():
    # 调用 `test_perform_collision_between_drones` 执行当前处理。
    test_perform_collision_between_drones()
    # 返回当前函数的结果。
    return


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 调用 `exit` 执行当前处理。
    sys.exit(speed_test())
