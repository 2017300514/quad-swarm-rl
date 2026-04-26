# 中文注释副本；原始文件：gym_art/quadrotor_multi/get_state.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np


# NOTE: the state_* methods are static because otherwise getattr memorizes self

# 定义函数 `state_xyz_vxyz_R_omega`。
def state_xyz_vxyz_R_omega(self):
    # 根据条件决定是否进入当前分支。
    if self.use_numba:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # 返回当前函数的结果。
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])


# 定义函数 `state_xyz_vxyz_R_omega_floor`。
def state_xyz_vxyz_R_omega_floor(self):
    # 根据条件决定是否进入当前分支。
    if self.use_numba:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # 返回当前函数的结果。
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])


# 定义函数 `state_xyz_vxyz_R_omega_wall`。
def state_xyz_vxyz_R_omega_wall(self):
    # 根据条件决定是否进入当前分支。
    if self.use_numba:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 同时更新 `pos`, `vel`, `rot`, `omega` 等变量。
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    # 保存或更新 `wall_box_0` 的值。
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    # 保存或更新 `wall_box_1` 的值。
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    # 返回当前函数的结果。
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, wall_box_0, wall_box_1])
