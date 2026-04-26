# 中文注释副本；原始文件：gym_art/quadrotor_multi/collisions/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
from numba import njit

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import EPS


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `compute_new_vel`。
def compute_new_vel(max_vel_magn, vel, vel_shift, low=0.2, high=0.8):
    # 保存或更新 `vel_decay_ratio` 的值。
    vel_decay_ratio = np.random.uniform(low, high)
    # 保存或更新 `vel_new` 的值。
    vel_new = vel + vel_shift
    # 保存或更新 `vel_new_mag` 的值。
    vel_new_mag = np.linalg.norm(vel_new)
    # 执行这一行逻辑。
    vel_new_dir = vel_new / (vel_new_mag + EPS if vel_new_mag == 0.0 else vel_new_mag)
    # 保存或更新 `vel_new_mag` 的值。
    vel_new_mag = min(vel_new_mag * vel_decay_ratio, max_vel_magn)
    # 保存或更新 `vel_new` 的值。
    vel_new = vel_new_dir * vel_new_mag

    # 保存或更新 `vel_shift` 的值。
    vel_shift = vel_new - vel
    # 保存或更新 `vel` 的值。
    vel += vel_shift
    # 返回当前函数的结果。
    return vel


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `compute_new_omega`。
def compute_new_omega(magn_scale=20.0):
    # Random forces for omega
    # This will amount to max 3.5 revolutions per second
    # 保存或更新 `omega_max` 的值。
    omega_max = magn_scale * np.pi
    # 保存或更新 `omega` 的值。
    omega = np.random.uniform(-1, 1, size=(3,))
    # 保存或更新 `omega_mag` 的值。
    omega_mag = np.linalg.norm(omega)

    # 执行这一行逻辑。
    omega_dir = omega / (omega_mag + EPS if omega_mag == 0.0 else omega_mag)
    # 保存或更新 `omega_mag` 的值。
    omega_mag = np.random.uniform(omega_max / 2, omega_max)
    # 保存或更新 `omega` 的值。
    omega = omega_dir * omega_mag

    # 返回当前函数的结果。
    return omega


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 定义函数 `main`。
    def main():
        # 导入当前模块依赖。
        import timeit
        # 保存或更新 `SETUP_CODE` 的值。
        SETUP_CODE = '''from __main__ import calculate_collision_matrix; import numpy as np'''

        # 保存或更新 `TEST_CODE` 的值。
        TEST_CODE = '''calculate_collision_matrix(positions=np.ones((8, 3)), arm=0.05, hitbox_radius=2)'''

        # timeit.repeat statement
        # 保存或更新 `times` 的值。
        times = timeit.repeat(setup=SETUP_CODE,
                              stmt=TEST_CODE,
                              repeat=5,
                              number=int(1e4))

        # printing minimum exec. time
        # 调用 `print` 执行当前处理。
        print('times:   ', times)
        # 调用 `print` 执行当前处理。
        print('mean times:   ', np.mean(times[1:]))


    # 根据条件决定是否进入当前分支。
    if __name__ == '__main__':
        # 导入当前模块依赖。
        import sys

        # 调用 `exit` 执行当前处理。
        sys.exit(main())
