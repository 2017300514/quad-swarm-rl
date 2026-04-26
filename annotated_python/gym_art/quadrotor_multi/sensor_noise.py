#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/sensor_noise.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件负责把动力学层里的理想状态扰动成“传感器看到的状态”。
# 上游输入来自 `QuadrotorDynamics` 的位置、速度、姿态、角速度和加速度计真值；
# 下游输出会被 `QuadrotorSingle` 的观测函数继续消费，从而让策略在训练时面对更接近真实机载传感器的输入。
# 这条链路也是 sim2real 相关实验的重要基础。

import numpy as np
from numpy.random import normal
from numpy.random import uniform
from math import exp
from numba import njit

from gym_art.quadrotor_multi.quad_utils import quatXquat, quat2R, quat2R_numba, quatXquat_numba


def quat_from_small_angle(theta):
    # 小角度扰动先转成四元数，再用于给姿态加随机旋转噪声。
    # 这样比直接在旋转矩阵上加噪更稳，也更容易和 numba 路径共享逻辑。
    assert theta.shape == (3,)

    q_squared = np.linalg.norm(theta) ** 2 / 4.0
    if q_squared < 1:
        q_theta = np.array([(1 - q_squared) ** 0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
    else:
        w = 1.0 / (1 + q_squared) ** 0.5
        f = 0.5 * w
        q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])

    q_theta = q_theta / np.linalg.norm(q_theta)
    return q_theta


quat_from_small_angle_numba = njit()(quat_from_small_angle)


'''
http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
'''


def rot2quat(rot):
    # 观测里如果当前姿态以旋转矩阵给出，这里先把它转成四元数，
    # 方便后面和小角度噪声四元数相乘，再转回矩阵。
    assert rot.shape == (3, 3)

    trace = np.trace(rot)
    if trace > 0:
        S = (trace + 1.0) ** 0.5 * 2
        qw = 0.25 * S
        qx = (rot[2][1] - rot[1][2]) / S
        qy = (rot[0][2] - rot[2][0]) / S
        qz = (rot[1][0] - rot[0][1]) / S
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2]) ** 0.5 * 2
        qw = (rot[2][1] - rot[1][2]) / S
        qx = 0.25 * S
        qy = (rot[0][1] + rot[1][0]) / S
        qz = (rot[0][2] + rot[2][0]) / S
    elif rot[1][1] > rot[2][2]:
        S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2]) ** 0.5 * 2
        qw = (rot[0][2] - rot[2][0]) / S
        qx = (rot[0][1] + rot[1][0]) / S
        qy = 0.25 * S
        qz = (rot[1][2] + rot[2][1]) / S
    else:
        S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1]) ** 0.5 * 2
        qw = (rot[1][0] - rot[0][1]) / S
        qx = (rot[0][2] + rot[2][0]) / S
        qy = (rot[1][2] + rot[2][1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


rot2quat_numba = njit()(rot2quat)


class SensorNoise:
    # 这个对象保存一整套传感器噪声参数，并在每次取观测时把理想状态扰动成 noisy state。
    # 它不改变真实动力学，只影响策略“看见什么”。
    def __init__(self, pos_norm_std=0.005, pos_unif_range=0.,
                 vel_norm_std=0.01, vel_unif_range=0.,
                 quat_norm_std=0., quat_unif_range=0., gyro_norm_std=0.,
                 gyro_noise_density=0.000175, gyro_random_walk=0.0105,
                 gyro_bias_correlation_time=1000., bypass=False,
                 acc_static_noise_std=0.002, acc_dynamic_noise_ratio=0.005,
                 use_numba=False):
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

        # 位置、速度、姿态、陀螺仪、加速度计噪声参数都在这里固定下来，
        # 后面 `add_noise()` / `add_noise_numba()` 只是重复按同一套参数采样。
        self.pos_norm_std = pos_norm_std
        self.pos_unif_range = pos_unif_range

        self.vel_norm_std = vel_norm_std
        self.vel_unif_range = vel_unif_range

        self.quat_norm_std = quat_norm_std
        self.quat_unif_range = quat_unif_range

        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.gyro_bias_correlation_time = gyro_bias_correlation_time
        self.gyro_norm_std = gyro_norm_std
        # self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
        self.gyro_bias = np.zeros(3)

        self.acc_static_noise_std = acc_static_noise_std
        self.acc_dynamic_noise_ratio = acc_dynamic_noise_ratio
        self.bypass = bypass

    def add_noise(self, pos, vel, rot, omega, acc, dt):
        if self.bypass:
            return pos, vel, rot, omega, acc
        # """
        # Args: 
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        # """
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert omega.shape == (3,)

        # 位置和速度都用“高斯 + 均匀”两部分噪声叠加，
        # 目的是同时模拟小幅连续误差和有限范围内的额外扰动。
        noisy_pos = pos + \
                    normal(loc=0., scale=self.pos_norm_std, size=3) + \
                    uniform(low=-self.pos_unif_range, high=self.pos_unif_range, size=3)

        # add noise to linear velocity
        noisy_vel = vel + \
                    normal(loc=0., scale=self.vel_norm_std, size=3) + \
                    uniform(low=-self.vel_unif_range, high=self.vel_unif_range, size=3)

        # 角速度噪声有两种路径：
        # 若启用更细的陀螺模型，则走 bias/random walk；
        # 否则退化成简单高斯噪声。
        if self.gyro_norm_std != 0.:
            noisy_omega = self.add_noise_to_omega(omega, dt)
        else:
            noisy_omega = omega + \
                          normal(loc=0., scale=self.gyro_noise_density, size=3)

        # 姿态噪声先采样一个小角度向量 `theta`，
        # 再按当前姿态表示形式分别处理成欧拉角扰动、旋转矩阵扰动或四元数扰动。
        theta = normal(0, self.quat_norm_std, size=3) + \
                uniform(-self.quat_unif_range, self.quat_unif_range, size=3)

        if rot.shape == (3,):
            # Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            noisy_rot = np.clip(rot + theta,
                                a_min=[-np.pi, -np.pi / 2, -np.pi],
                                a_max=[np.pi, np.pi / 2, np.pi])
        elif rot.shape == (3, 3):
            # Rotation matrix
            quat_theta = quat_from_small_angle(theta)
            quat = rot2quat(rot)
            noisy_quat = quatXquat(quat, quat_theta)
            noisy_rot = quat2R(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        elif rot.shape == (4,):
            # Quaternion
            quat_theta = quat_from_small_angle(theta)
            noisy_rot = quatXquat(rot, quat_theta)
        else:
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        # 加速度计既有静态噪声，也有与当前加速度成比例的动态噪声。
        noisy_acc = acc + normal(loc=0., scale=self.acc_static_noise_std, size=3) + \
                    acc * normal(loc=0., scale=self.acc_dynamic_noise_ratio, size=3)

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    def add_noise_numba(self, pos, vel, rot, omega, acc, dt):
        if self.bypass:
            return pos, vel, rot, omega, acc
        # """
        # Args:
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        # """
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert omega.shape == (3,)

        # numba 路径把最常见的随机采样部分提前编译，减少多机训练时的观测噪声开销。
        noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta = add_noise_to_vel_acc_pos_omega_rot(
            pos, vel, omega, acc,
            pos_rand_var=(self.pos_norm_std, self.pos_unif_range),
            vel_rand_var=(self.vel_norm_std, self.vel_unif_range),
            omega_rand_var=self.gyro_noise_density,
            acc_rand_var=(self.acc_static_noise_std, self.acc_dynamic_noise_ratio),
            rot_rand_var=(self.quat_norm_std, self.quat_unif_range),
        )

        # 如果启用了更真实的陀螺模型，这里再覆盖简单高斯版本的角速度噪声。
        if self.gyro_norm_std != 0.:
            noisy_omega = self.add_noise_to_omega(omega, dt)

        if rot.shape == (3,):
            # Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            noisy_rot = np.clip(rot + theta,
                                a_min=[-np.pi, -np.pi / 2, -np.pi],
                                a_max=[np.pi, np.pi / 2, np.pi])
        elif rot.shape == (3, 3):
            # Rotation matrix
            quat_theta = quat_from_small_angle_numba(theta)
            quat = rot2quat_numba(rot)
            noisy_quat = quatXquat_numba(quat, quat_theta)
            noisy_rot = quat2R_numba(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        elif rot.shape == (4,):
            # Quaternion
            quat_theta = quat_from_small_angle_numba(theta)
            noisy_rot = quatXquat_numba(rot, quat_theta)
        else:
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    # copy from rotorS imu plugin
    def add_noise_to_omega(self, omega, dt):
        # 这里实现的是带 bias correlation time 的陀螺仪噪声模型。
        # 它会让陀螺偏置随时间缓慢漂移，比独立同分布白噪声更接近真实 IMU。
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt ** 0.5)
        sigma_b_g_d = (-(sigma_g_d ** 2) * (self.gyro_bias_correlation_time / 2) * (
                    exp(-2 * dt / self.gyro_bias_correlation_time) - 1)) ** 0.5
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * normal(0, 1, 3)
        return omega + self.gyro_bias + self.gyro_random_walk * normal(0, 1,
                                                                       3)  # + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)


@njit
def add_noise_to_vel_acc_pos_omega_rot(
        pos, vel, omega, acc, pos_rand_var, vel_rand_var, omega_rand_var,
        acc_rand_var, rot_rand_var
):
    # 这是 `add_noise_numba()` 复用的底层采样核心。
    # 它把位置、速度、角速度、加速度和姿态小角度噪声一次性采出来，再交给上层按姿态表示形式重组。
    noisy_pos = pos + \
                normal(loc=0., scale=pos_rand_var[0], size=3) + \
                uniform(-pos_rand_var[1], pos_rand_var[1], 3)

    # Add noise to linear velocity
    noisy_vel = vel + \
                normal(loc=0., scale=vel_rand_var[0], size=3) + \
                uniform(-vel_rand_var[1], vel_rand_var[1], 3)

    # Noise in omega
    noisy_omega = omega + \
                  normal(loc=0., scale=omega_rand_var, size=3)

    # Noise in rotation
    theta = normal(loc=0, scale=rot_rand_var[0], size=3) + \
            uniform(-rot_rand_var[1], rot_rand_var[1], 3)

    # Accelerometer noise
    noisy_acc = acc + normal(loc=0., scale=acc_rand_var[0], size=3) + \
                (acc * normal(loc=0., scale=acc_rand_var[1], size=3))

    return noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta


if __name__ == "__main__":
    sens = SensorNoise()
    import time

    start_time = time.time()
    sens.add_noise(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), 0.005)
    print("Noise generation time: ", time.time() - start_time)
