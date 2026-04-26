#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/get_state.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件定义单机环境把底层动力学状态整理成策略观测向量的最后一步。
# 上游输入来自 `QuadrotorDynamics` 的真实状态和 `SensorNoise` 生成的带噪观测；
# 下游输出是 `QuadrotorSingle.state_vector()` 按不同 `obs_repr` 返回给策略网络的 self observation。
# 多机环境后面还会在这些自观测后拼上邻居观测和障碍观测。

import numpy as np


# NOTE: the state_* methods are static because otherwise getattr memorizes self
# 这里做成模块级函数，而不是类方法，是为了让 `QuadrotorSingle` 能用 `getattr(get_state, "state_" + obs_repr)` 直接切换观测表示。

def state_xyz_vxyz_R_omega(self):
    # 这是最基础的自观测布局：
    # 相对目标位置、线速度、旋转矩阵、角速度。
    # 它先经过传感器噪声模型，再拼成网络真正看到的输入。
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # 位置始终用“相对目标”的形式给策略，而不是绝对世界坐标，
    # 这样任务目标直接进入观测，策略不必自己再减一次 goal。
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])


def state_xyz_vxyz_R_omega_floor(self):
    # 这个版本在基础自观测后额外附上当前高度 `pos[2]`，
    # 主要用于让策略更直接感知离地关系。
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])


def state_xyz_vxyz_R_omega_wall(self):
    # 这个版本则把“离房间六个边界还有多远”编码进观测，
    # 让策略在不额外构建复杂墙面传感器的情况下，直接看到自身与房间边界的余量。
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # `wall_box_0` / `wall_box_1` 分别表示当前位置到房间最小边界和最大边界的距离，
    # 并裁到 5m 上限，避免过大房间尺寸把输入尺度拉得太散。
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, wall_box_0, wall_box_1])
