# 中文注释副本；原始文件：gym_art/quadrotor_multi/sensor_noise.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

#!/usr/bin/env python
# 导入当前模块依赖。
import numpy as np
from numpy.random import normal
from numpy.random import uniform
from math import exp
from numba import njit

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import quatXquat, quat2R, quat2R_numba, quatXquat_numba


# 定义函数 `quat_from_small_angle`。
def quat_from_small_angle(theta):
    # 断言当前条件成立，用于保护运行假设。
    assert theta.shape == (3,)

    # 保存或更新 `q_squared` 的值。
    q_squared = np.linalg.norm(theta) ** 2 / 4.0
    # 根据条件决定是否进入当前分支。
    if q_squared < 1:
        # 保存或更新 `q_theta` 的值。
        q_theta = np.array([(1 - q_squared) ** 0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `w` 的值。
        w = 1.0 / (1 + q_squared) ** 0.5
        # 保存或更新 `f` 的值。
        f = 0.5 * w
        # 保存或更新 `q_theta` 的值。
        q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])

    # 保存或更新 `q_theta` 的值。
    q_theta = q_theta / np.linalg.norm(q_theta)
    # 返回当前函数的结果。
    return q_theta


# 保存或更新 `quat_from_small_angle_numba` 的值。
quat_from_small_angle_numba = njit()(quat_from_small_angle)


# 下面开始文档字符串说明。
'''
http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
'''


# 定义函数 `rot2quat`。
def rot2quat(rot):
    # 断言当前条件成立，用于保护运行假设。
    assert rot.shape == (3, 3)

    # 保存或更新 `trace` 的值。
    trace = np.trace(rot)
    # 根据条件决定是否进入当前分支。
    if trace > 0:
        # 保存或更新 `S` 的值。
        S = (trace + 1.0) ** 0.5 * 2
        # 保存或更新 `qw` 的值。
        qw = 0.25 * S
        # 保存或更新 `qx` 的值。
        qx = (rot[2][1] - rot[1][2]) / S
        # 保存或更新 `qy` 的值。
        qy = (rot[0][2] - rot[2][0]) / S
        # 保存或更新 `qz` 的值。
        qz = (rot[1][0] - rot[0][1]) / S
    # 当上一分支不满足时，继续判断新的条件。
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        # 保存或更新 `S` 的值。
        S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2]) ** 0.5 * 2
        # 保存或更新 `qw` 的值。
        qw = (rot[2][1] - rot[1][2]) / S
        # 保存或更新 `qx` 的值。
        qx = 0.25 * S
        # 保存或更新 `qy` 的值。
        qy = (rot[0][1] + rot[1][0]) / S
        # 保存或更新 `qz` 的值。
        qz = (rot[0][2] + rot[2][0]) / S
    # 当上一分支不满足时，继续判断新的条件。
    elif rot[1][1] > rot[2][2]:
        # 保存或更新 `S` 的值。
        S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2]) ** 0.5 * 2
        # 保存或更新 `qw` 的值。
        qw = (rot[0][2] - rot[2][0]) / S
        # 保存或更新 `qx` 的值。
        qx = (rot[0][1] + rot[1][0]) / S
        # 保存或更新 `qy` 的值。
        qy = 0.25 * S
        # 保存或更新 `qz` 的值。
        qz = (rot[1][2] + rot[2][1]) / S
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `S` 的值。
        S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1]) ** 0.5 * 2
        # 保存或更新 `qw` 的值。
        qw = (rot[1][0] - rot[0][1]) / S
        # 保存或更新 `qx` 的值。
        qx = (rot[0][2] + rot[2][0]) / S
        # 保存或更新 `qy` 的值。
        qy = (rot[1][2] + rot[2][1]) / S
        # 保存或更新 `qz` 的值。
        qz = 0.25 * S

    # 返回当前函数的结果。
    return np.array([qw, qx, qy, qz])


# 保存或更新 `rot2quat_numba` 的值。
rot2quat_numba = njit()(rot2quat)


# 定义类 `SensorNoise`。
class SensorNoise:
    # 定义函数 `__init__`。
    def __init__(self, pos_norm_std=0.005, pos_unif_range=0.,
                 vel_norm_std=0.01, vel_unif_range=0.,
                 quat_norm_std=0., quat_unif_range=0., gyro_norm_std=0.,
                 gyro_noise_density=0.000175, gyro_random_walk=0.0105,
                 gyro_bias_correlation_time=1000., bypass=False,
                 acc_static_noise_std=0.002, acc_dynamic_noise_ratio=0.005,
                 # 保存或更新 `use_numba` 的值。
                 use_numba=False):
        # 下面开始文档字符串说明。
        """
        Args:
            pos_norm_std (float): std of pos gaus noise component
            pos_unif_range (float): range of pos unif noise component
            vel_norm_std (float): std of linear vel gaus noise component 
            vel_unif_range (float): range of linear vel unif noise component
            quat_norm_std (float): std of rotational quaternion noisy angle gaus component
            quat_unif_range (float): range of rotational quaternion noisy angle gaus component
            gyro_gyro_noise_density: gyroscope noise, MPU-9250 spec
            gyro_random_walk: gyroscope noise, MPU-9250 spec
            gyro_bias_correlation_time: gyroscope noise, MPU-9250 spec
            # gyro_gyro_turn_on_bias_sigma: gyroscope noise, MPU-9250 spec (val 0.09)
            bypass: no noise
        """

        # 保存或更新 `pos_norm_std` 的值。
        self.pos_norm_std = pos_norm_std
        # 保存或更新 `pos_unif_range` 的值。
        self.pos_unif_range = pos_unif_range

        # 保存或更新 `vel_norm_std` 的值。
        self.vel_norm_std = vel_norm_std
        # 保存或更新 `vel_unif_range` 的值。
        self.vel_unif_range = vel_unif_range

        # 保存或更新 `quat_norm_std` 的值。
        self.quat_norm_std = quat_norm_std
        # 保存或更新 `quat_unif_range` 的值。
        self.quat_unif_range = quat_unif_range

        # 保存或更新 `gyro_noise_density` 的值。
        self.gyro_noise_density = gyro_noise_density
        # 保存或更新 `gyro_random_walk` 的值。
        self.gyro_random_walk = gyro_random_walk
        # 保存或更新 `gyro_bias_correlation_time` 的值。
        self.gyro_bias_correlation_time = gyro_bias_correlation_time
        # 保存或更新 `gyro_norm_std` 的值。
        self.gyro_norm_std = gyro_norm_std
        # self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
        # 保存或更新 `gyro_bias` 的值。
        self.gyro_bias = np.zeros(3)

        # 保存或更新 `acc_static_noise_std` 的值。
        self.acc_static_noise_std = acc_static_noise_std
        # 保存或更新 `acc_dynamic_noise_ratio` 的值。
        self.acc_dynamic_noise_ratio = acc_dynamic_noise_ratio
        # 保存或更新 `bypass` 的值。
        self.bypass = bypass

    # 定义函数 `add_noise`。
    def add_noise(self, pos, vel, rot, omega, acc, dt):
        # 根据条件决定是否进入当前分支。
        if self.bypass:
            # 返回当前函数的结果。
            return pos, vel, rot, omega, acc
        # """
        # Args: 
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        # """
        # 断言当前条件成立，用于保护运行假设。
        assert pos.shape == (3,)
        # 断言当前条件成立，用于保护运行假设。
        assert vel.shape == (3,)
        # 断言当前条件成立，用于保护运行假设。
        assert omega.shape == (3,)

        # add noise to position measurement
        # 保存或更新 `noisy_pos` 的值。
        noisy_pos = pos + \
                    # 保存或更新 `normal(loc` 的值。
                    normal(loc=0., scale=self.pos_norm_std, size=3) + \
                    # 保存或更新 `uniform(low` 的值。
                    uniform(low=-self.pos_unif_range, high=self.pos_unif_range, size=3)

        # add noise to linear velocity
        # 保存或更新 `noisy_vel` 的值。
        noisy_vel = vel + \
                    # 保存或更新 `normal(loc` 的值。
                    normal(loc=0., scale=self.vel_norm_std, size=3) + \
                    # 保存或更新 `uniform(low` 的值。
                    uniform(low=-self.vel_unif_range, high=self.vel_unif_range, size=3)

        ## Noise in omega
        # 根据条件决定是否进入当前分支。
        if self.gyro_norm_std != 0.:
            # 保存或更新 `noisy_omega` 的值。
            noisy_omega = self.add_noise_to_omega(omega, dt)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `noisy_omega` 的值。
            noisy_omega = omega + \
                          # 保存或更新 `normal(loc` 的值。
                          normal(loc=0., scale=self.gyro_noise_density, size=3)

        # Noise in rotation
        # 保存或更新 `theta` 的值。
        theta = normal(0, self.quat_norm_std, size=3) + \
                # 保存或更新 `uniform(-quat_unif_range, quat_unif_range, size` 的值。
                uniform(-self.quat_unif_range, self.quat_unif_range, size=3)

        # 根据条件决定是否进入当前分支。
        if rot.shape == (3,):
            # Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = np.clip(rot + theta,
                                a_min=[-np.pi, -np.pi / 2, -np.pi],
                                a_max=[np.pi, np.pi / 2, np.pi])
        # 当上一分支不满足时，继续判断新的条件。
        elif rot.shape == (3, 3):
            # Rotation matrix
            # 保存或更新 `quat_theta` 的值。
            quat_theta = quat_from_small_angle(theta)
            # 保存或更新 `quat` 的值。
            quat = rot2quat(rot)
            # 保存或更新 `noisy_quat` 的值。
            noisy_quat = quatXquat(quat, quat_theta)
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = quat2R(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        # 当上一分支不满足时，继续判断新的条件。
        elif rot.shape == (4,):
            # Quaternion
            # 保存或更新 `quat_theta` 的值。
            quat_theta = quat_from_small_angle(theta)
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = quatXquat(rot, quat_theta)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        # Accelerometer noise
        # 保存或更新 `noisy_acc` 的值。
        noisy_acc = acc + normal(loc=0., scale=self.acc_static_noise_std, size=3) + \
                    # 保存或更新 `acc * normal(loc` 的值。
                    acc * normal(loc=0., scale=self.acc_dynamic_noise_ratio, size=3)

        # 返回当前函数的结果。
        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    # 定义函数 `add_noise_numba`。
    def add_noise_numba(self, pos, vel, rot, omega, acc, dt):
        # 根据条件决定是否进入当前分支。
        if self.bypass:
            # 返回当前函数的结果。
            return pos, vel, rot, omega, acc
        # """
        # Args:
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        # """
        # 断言当前条件成立，用于保护运行假设。
        assert pos.shape == (3,)
        # 断言当前条件成立，用于保护运行假设。
        assert vel.shape == (3,)
        # 断言当前条件成立，用于保护运行假设。
        assert omega.shape == (3,)

        # 同时更新 `noisy_pos`, `noisy_vel`, `noisy_omega`, `noisy_acc` 等变量。
        noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta = add_noise_to_vel_acc_pos_omega_rot(
            pos, vel, omega, acc,
            pos_rand_var=(self.pos_norm_std, self.pos_unif_range),
            vel_rand_var=(self.vel_norm_std, self.vel_unif_range),
            omega_rand_var=self.gyro_noise_density,
            acc_rand_var=(self.acc_static_noise_std, self.acc_dynamic_noise_ratio),
            rot_rand_var=(self.quat_norm_std, self.quat_unif_range),
        )

        # Noise in omega
        # 根据条件决定是否进入当前分支。
        if self.gyro_norm_std != 0.:
            # 保存或更新 `noisy_omega` 的值。
            noisy_omega = self.add_noise_to_omega(omega, dt)

        # 根据条件决定是否进入当前分支。
        if rot.shape == (3,):
            # Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = np.clip(rot + theta,
                                a_min=[-np.pi, -np.pi / 2, -np.pi],
                                a_max=[np.pi, np.pi / 2, np.pi])
        # 当上一分支不满足时，继续判断新的条件。
        elif rot.shape == (3, 3):
            # Rotation matrix
            # 保存或更新 `quat_theta` 的值。
            quat_theta = quat_from_small_angle_numba(theta)
            # 保存或更新 `quat` 的值。
            quat = rot2quat_numba(rot)
            # 保存或更新 `noisy_quat` 的值。
            noisy_quat = quatXquat_numba(quat, quat_theta)
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = quat2R_numba(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        # 当上一分支不满足时，继续判断新的条件。
        elif rot.shape == (4,):
            # Quaternion
            # 保存或更新 `quat_theta` 的值。
            quat_theta = quat_from_small_angle_numba(theta)
            # 保存或更新 `noisy_rot` 的值。
            noisy_rot = quatXquat_numba(rot, quat_theta)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        # 返回当前函数的结果。
        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    # copy from rotorS imu plugin
    # 定义函数 `add_noise_to_omega`。
    def add_noise_to_omega(self, omega, dt):
        # 断言当前条件成立，用于保护运行假设。
        assert omega.shape == (3,)

        # 保存或更新 `sigma_g_d` 的值。
        sigma_g_d = self.gyro_noise_density / (dt ** 0.5)
        # 保存或更新 `sigma_b_g_d` 的值。
        sigma_b_g_d = (-(sigma_g_d ** 2) * (self.gyro_bias_correlation_time / 2) * (
                    exp(-2 * dt / self.gyro_bias_correlation_time) - 1)) ** 0.5
        # 保存或更新 `pi_g_d` 的值。
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        # 保存或更新 `gyro_bias` 的值。
        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * normal(0, 1, 3)
        # 返回当前函数的结果。
        return omega + self.gyro_bias + self.gyro_random_walk * normal(0, 1,
                                                                       3)  # + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `add_noise_to_vel_acc_pos_omega_rot`。
def add_noise_to_vel_acc_pos_omega_rot(
        pos, vel, omega, acc, pos_rand_var, vel_rand_var, omega_rand_var,
        acc_rand_var, rot_rand_var
# 这里开始一个新的代码块。
):
    # add noise to position measurement
    # 保存或更新 `noisy_pos` 的值。
    noisy_pos = pos + \
                # 保存或更新 `normal(loc` 的值。
                normal(loc=0., scale=pos_rand_var[0], size=3) + \
                # 调用 `uniform` 执行当前处理。
                uniform(-pos_rand_var[1], pos_rand_var[1], 3)

    # Add noise to linear velocity
    # 保存或更新 `noisy_vel` 的值。
    noisy_vel = vel + \
                # 保存或更新 `normal(loc` 的值。
                normal(loc=0., scale=vel_rand_var[0], size=3) + \
                # 调用 `uniform` 执行当前处理。
                uniform(-vel_rand_var[1], vel_rand_var[1], 3)

    # Noise in omega
    # 保存或更新 `noisy_omega` 的值。
    noisy_omega = omega + \
                  # 保存或更新 `normal(loc` 的值。
                  normal(loc=0., scale=omega_rand_var, size=3)

    # Noise in rotation
    # 保存或更新 `theta` 的值。
    theta = normal(loc=0, scale=rot_rand_var[0], size=3) + \
            # 调用 `uniform` 执行当前处理。
            uniform(-rot_rand_var[1], rot_rand_var[1], 3)

    # Accelerometer noise
    # 保存或更新 `noisy_acc` 的值。
    noisy_acc = acc + normal(loc=0., scale=acc_rand_var[0], size=3) + \
                # 保存或更新 `(acc * normal(loc` 的值。
                (acc * normal(loc=0., scale=acc_rand_var[1], size=3))

    # 返回当前函数的结果。
    return noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 保存或更新 `sens` 的值。
    sens = SensorNoise()
    # 导入当前模块依赖。
    import time

    # 保存或更新 `start_time` 的值。
    start_time = time.time()
    # 调用 `add_noise` 执行当前处理。
    sens.add_noise(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), 0.005)
    # 调用 `print` 执行当前处理。
    print("Noise generation time: ", time.time() - start_time)
