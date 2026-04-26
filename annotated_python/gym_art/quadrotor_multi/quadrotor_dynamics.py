#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_dynamics.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是单机环境 `QuadrotorSingle` 背后的物理推进核心。
# 上游输入主要来自控制器给出的四路归一化电机命令，以及随机化后的质量、惯量、推力和阻尼参数；
# 下游输出是位置、速度、姿态、角速度、加速度计读数，以及墙面/天花板/地面碰撞标记。
# 多机环境不会在这里直接处理邻居或奖励，它只消费这里更新后的单机状态，再去计算邻居观测、碰撞与奖励。

from copy import deepcopy

import numpy as np
from gymnasium import spaces
from numba import njit

from gym_art.quadrotor_multi.inertia import QuadLink, QuadLinkSimplified
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba, angvel2thrust_numba, numba_cross
from gym_art.quadrotor_multi.quad_utils import OUNoise, rand_uniform_rot3d, cross_vec_mx4, cross_mx4, npa, cross, \
    randyaw, to_xyhat, normalize

GRAV = 9.81  # default gravitational constant
EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


# WARN:
# linearity is set to 1 always, by means of check_quad_param_limits().
# The def. value of linearity for CF is set to 1 as well (due to firmware non-linearity compensation)
class QuadrotorDynamics:
    # 这个类维护一架无人机最底层的连续状态与动力学系数。
    # `QuadrotorSingle.step()` 每推进一次物理仿真，最终都会落到这里，把控制器输出转成新的姿态、速度和接触状态。
    """
    Simple simulation of quadrotor dynamics.
    mass unit: kilogram
    arm_length unit: meter
    inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    thrust_to_weight is the total, it will be divided among the 4 props
    torque_to_thrust is ratio of torque produced by prop to thrust
    thrust_noise_ratio is noise2signal ratio of the thrust noise, Ex: 0.05 = 5% of the current signal
      It is an approximate ratio, i.e. the upper bound could still be higher, due to how OU noise operates
    Coord frames: x configuration:
     - x axis between arms looking forward [x - configuration]
     - y axis pointing to the left
     - z axis up
    TODO:
    - only diagonal inertia is used at the moment
    """

    def __init__(self, model_params, room_box=None, dynamics_steps_num=1, dim_mode="3D", gravity=GRAV,
                 dynamics_simplification=False, use_numba=False, dt=1/200):
        # Pre-set Parameters
        self.dt = dt
        self.use_numba = use_numba

        # `dynamics_steps_num` 允许一个控制步内部细分多个物理积分小步，
        # 用来把策略动作频率和更高的动力学积分频率对齐。
        # `dynamics_simplification` 则决定是用完整刚体几何还是简化机体模型。
        self.dynamics_steps_num = dynamics_steps_num
        self.dynamics_simplification = dynamics_simplification
        # cw = 1 ; ccw = -1 [ccw, cw, ccw, cw]
        self.prop_ccw = np.array([-1., 1., -1., 1.])
        # Reference: https://docs.google.com/document/d/1wZMZQ6jilDbj0JtfeYt0TonjxoMPIgHwYbrFrMNls84/edit
        self.omega_max = 40.  # rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3.  # m/s
        self.gravity = gravity
        self.acc_max = 3. * GRAV
        self.since_last_svd = 0  # counter
        self.since_last_svd_limit = 0.5  # in sec - how often mandatory orthogonality should be applied
        self.eye = np.eye(3)
        # Initializing model
        self.thrust_noise = None
        self.update_model(model_params)
        # Sanity checks
        assert self.inertia.shape == (3,)

        # Dynamics used in step
        self.motor_tau_up = 4 * dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * dt / (self.motor_damp_time_down + EPS)

        # 房间边界在这里就写进动力学层，
        # 因为墙壁/天花板裁剪和地面接触都要在积分后第一时间处理。
        if room_box is None:
            self.room_box = np.array([[0., 0., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        # # Floor
        self.on_floor = False
        # # # If pos_z smaller than this threshold, we assume that drone collide with the floor
        self.floor_threshold = 0.05
        # # # Floor Fiction
        self.mu = 0.6
        # # # Collision with room
        self.crashed_wall = False
        self.crashed_ceiling = False
        self.crashed_floor = False

        # `control_mx` 描述不同维度模式下允许哪些电机自由度真正参与控制。
        # 它主要服务于简化实验，例如只做高度稳定或二维平面控制。
        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.control_mx = np.ones([4, 1])
        elif self.dim_mode == '2D':
            self.control_mx = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        elif self.dim_mode == '3D':
            self.control_mx = np.eye(4)
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)

    @staticmethod
    def angvel2thrust(w, linearity=0.424):
        """
        CrazyFlie: linearity=0.424
        Args:
            w: thrust_cmds_damp
            linearity (float): linearity factor factor [0 .. 1].
        """
        return (1 - linearity) * w ** 2 + linearity * w

    def update_model(self, model_params):
        # 这里把随机化后的模型参数翻译成后续每一步都会反复使用的动力学常数。
        # 对训练来说，真正重要的不是字典本身，而是这些值如何决定“同样的动作会产生多大的推力和角加速度”。
        if self.dynamics_simplification:
            self.model = QuadLinkSimplified(params=model_params["geom"])
        else:
            self.model = QuadLink(params=model_params["geom"])
        self.model_params = model_params

        # PARAMETERS FOR RANDOMIZATION
        self.mass = self.model.m
        self.inertia = np.diagonal(self.model.I_com)

        self.thrust_to_weight = self.model_params["motor"]["thrust_to_weight"]
        self.torque_to_thrust = self.model_params["motor"]["torque_to_thrust"]
        self.motor_linearity = self.model_params["motor"]["linearity"]
        self.C_rot_drag = self.model_params["motor"]["C_drag"]
        self.C_rot_roll = self.model_params["motor"]["C_roll"]
        self.motor_damp_time_up = self.model_params["motor"]["damp_time_up"]
        self.motor_damp_time_down = self.model_params["motor"]["damp_time_down"]

        self.thrust_noise_ratio = self.model_params["noise"]["thrust_noise_ratio"]
        self.vel_damp = self.model_params["damp"]["vel"]
        self.damp_omega_quadratic = self.model_params["damp"]["omega_quadratic"]

        # 这里开始计算依赖几何和电机参数的派生量，
        # 它们会在 step 中直接参与“电机命令 -> 推力/力矩 -> 刚体状态更新”的主链路。
        try:
            self.motor_assymetry = np.array(self.model_params["motor"]["assymetry"])
        except:
            self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
            print("WARNING: Motor assymetry was not setup. Setting assymetry to:", self.motor_assymetry)
        self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)  # re-normalizing to sum-up to 4
        self.thrust_max = GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.torque_max = self.torque_to_thrust * self.thrust_max  # propeller torque scales

        # Propeller positions in X configurations
        self.prop_pos = self.model.prop_pos

        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        self.prop_ccw_mx = np.zeros([3, 4])  # Matrix allows using matrix multiplication
        self.prop_ccw_mx[2, :] = self.prop_ccw

        # 这几组矩阵把“四个桨各自产生的力和力矩”预先整理成线性代数友好的形式，
        # 方便控制器、雅可比和积分器在高频循环里重复使用。
        # Prop crossproduct give torque directions
        self.G_omega_thrust = self.thrust_max * self.prop_crossproducts.T  # [3,4] @ [4,1]
        # additional torques along z-axis caused by propeller rotations
        self.C_omega_prop = self.torque_max * self.prop_ccw_mx  # [3,4] @ [4,1] = [3,1]
        self.G_omega = (1.0 / self.inertia)[:, None] * (self.G_omega_thrust + self.C_omega_prop)

        # Allows to sum-up thrusts as a linear matrix operation
        self.thrust_sum_mx = np.zeros([3, 4])  # [0,0,F_sum].T
        self.thrust_sum_mx[2, :] = 1  # [0,0,F_sum].T

        self.init_thrust_noise()

        self.arm = np.linalg.norm(self.model.motor_xyz[:2])

        # the ratio between max torque and inertia around each axis
        # the 0-1 matrix on the right is the way to sum-up
        self.torque_to_inertia = self.G_omega @ np.array([[0., 0., 0.], [0., 1., 1.], [1., 1., 0.], [1., 0., 1.]])
        self.torque_to_inertia = np.sum(self.torque_to_inertia, axis=1)
        # self.torque_to_inertia = self.torque_to_inertia / np.linalg.norm(self.torque_to_inertia)

        self.reset()

    def init_thrust_noise(self):
        # sigma = 0.2 gives roughly max noise of -1 ... 1
        # 推力噪声是 domain randomization 的一部分。
        # 它让同样的电机命令在不同 step 里出现相关扰动，避免策略只适应理想无噪声电机。
        if self.use_numba:
            self.thrust_noise = OUNoiseNumba(4, sigma=0.2 * self.thrust_noise_ratio)
        else:
            self.thrust_noise = OUNoise(4, sigma=0.2 * self.thrust_noise_ratio)

    # pos, vel, in world coordinates (meters)
    # rotation is 3x3 matrix (body coordinates) -> (world coordinates)dt
    # omega is angular velocity (radians/sec) in body coordinates, i.e. the gyroscope
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        # reset、回放恢复和碰撞修正都会用这个入口直接覆盖动力学状态。
        # 后续观测构造、奖励计算和渲染看到的，都是这里写入后的最新物理量。
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3, 3)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega.astype(np.float32))
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)
    @staticmethod
    def random_state(box, vel_max=15.0, omega_max=2 * np.pi):
        # 这个采样器主要服务于单机 reset 时的随机初始状态生成。
        # 它不是多机场景生成器本身，但会影响训练早期策略接触到的姿态和速度分布。
        box = np.array(box)
        pos = np.random.uniform(low=-box, high=box, size=(3,))

        vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
        vel_magn = np.random.uniform(low=0., high=vel_max)
        vel = vel_magn / (np.linalg.norm(vel) + EPS) * vel

        omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
        omega_magn = np.random.uniform(low=0., high=omega_max)
        omega = omega_magn / (np.linalg.norm(omega) + EPS) * omega

        rot = rand_uniform_rot3d()
        return pos, vel, rot, omega

    def step(self, thrust_cmds, dt):
        # 一个控制步内先采一次 OU 推力噪声，再按设定的小步数重复积分。
        # 这样同一策略动作可以对应更细的物理推进，而不是简单地一次欧拉更新结束。
        thrust_noise = self.thrust_noise.noise()

        if self.use_numba:
            [self.step1_numba(thrust_cmds, dt, thrust_noise) for _ in range(self.dynamics_steps_num)]
        else:
            [self.step1(thrust_cmds, dt, thrust_noise) for _ in range(self.dynamics_steps_num)]

    # Step function integrates based on current derivative values (best fits affine dynamics model)
    # thrust_cmds is motor thrusts given in normalized range [0, 1].
    # 1 represents the max possible thrust of the motor.
    # Frames:
    # pos - global
    # vel - global
    # rot - global
    # omega - body frame
    # goal_pos - global
    def step1(self, thrust_cmds, dt, thrust_noise):
        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)

        # 这一步先做电机一阶滞后，再叠加推力噪声。
        # 策略输出不是“立刻生效的理想推力”，而是会经历电机响应时间和平滑过程。
        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        motor_tau_down = np.array(self.motor_tau_down)
        motor_tau = self.motor_tau_up * np.ones([4, ])
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = motor_tau_down
        motor_tau[motor_tau > 1.] = 1.

        # 源码把控制量先开方再滤波，再平方回推力命令，
        # 近似模拟“电机转速响应再转回推力”的过程。
        # WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
        # since it likely means that you are using rotational velocities as an input instead of the thrust and hence
        # you are filtering square roots of angular velocities
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        # 噪声按当前命令幅值缩放，意味着高油门时允许更大的推力扰动。
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        # Prop crossproduct give torque directions
        self.torques = self.prop_crossproducts * thrusts[:, None]  # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        self.torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp

        # net torque: sum over propellers
        thrust_torque = np.sum(self.torques, axis=0)

        # 这里补充旋翼平面内速度导致的 drag / rolling 力矩。
        # 这些项不决定基本能不能飞，但会影响高速飞行和近地/大姿态时的细节稳定性。
        # See Ref[1] Sec:2.1 for details
        if self.C_rot_drag != 0 or self.C_rot_roll != 0:
            vel_body = self.rot.T @ self.vel
            v_rotor = vel_body + cross_vec_mx4(self.omega, self.model.prop_pos)
            v_rotor[:, 2] = 0.  # Projection to the rotor plane

            # Drag/Roll of rotors (both in body frame)
            rotor_drag_fi = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotor
            rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
            rotor_drag_ti = cross_mx4(rotor_drag_fi, self.model.prop_pos)
            rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

            rotor_roll_torque = \
                - self.C_rot_roll * self.prop_ccw[:, None] * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotor
            rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
            rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

            # Constraints (prevent numerical instabilities)
            vel_norm = np.linalg.norm(vel_body)
            rdf_norm = np.linalg.norm(rotor_drag_force)
            rdf_norm_clip = np.clip(rdf_norm, a_min=0., a_max=vel_norm * self.mass / (2 * dt))
            if rdf_norm > EPS:
                rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

            # omega_norm = np.linalg.norm(self.omega)
            rvt_norm = np.linalg.norm(rotor_visc_torque)
            rvt_norm_clipped = np.clip(rvt_norm, a_min=0., a_max=np.linalg.norm(self.omega * self.inertia) / (2 * dt))
            if rvt_norm > EPS:
                rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
        else:
            rotor_visc_torque = rotor_drag_force = np.zeros(3)

        # (Square) Damping using torques (in case we would like to add damping using torques)
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        self.torque = thrust_torque + rotor_visc_torque
        thrust = npa(0, 0, np.sum(thrusts))

        # 角速度先从机体系变到世界系，再用 Rodrigues 公式积分姿态矩阵。
        # 这一步产出的 `self.rot` 会立刻被观测、控制器和碰撞逻辑继续使用。
        # Integrating rotations (based on current values)
        omega_vec = np.matmul(self.rot, self.omega)  # Change from body to world frame
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            # See [7]
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = self.eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
            self.rot = dRdt @ self.rot

        # SVD is not strictly required anymore. Performing it rarely, just in case
        self.since_last_svd += dt
        if self.since_last_svd > self.since_last_svd_limit:
            # Perform SVD orthogonolization
            u, s, v = np.linalg.svd(self.rot)
            self.rot = np.matmul(u, v)
            self.since_last_svd = 0

        # 这里完成刚体角速度更新。
        # `self.torque` 来自四桨合力矩和旋翼粘性项，`self.omega_dot` 则是控制器下一步稳定姿态时的重要反馈量。
        # Damping using velocities (I find it more stable numerically)
        # This is only for linear damping of angular velocity.
        self.omega_dot = ((1.0 / self.inertia) * (cross(-self.omega, self.inertia * self.omega) + self.torque))

        # Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * self.omega_dot
        self.omega = np.clip(self.omega, a_min=-self.omega_max, a_max=self.omega_max)

        # 平移部分先更新位置，再立刻和房间边界、地面接触逻辑交互。
        # 也就是说，墙面/地板裁剪发生在同一个积分步内，而不是留给上层环境事后修正。
        # Computing position
        self.pos = self.pos + dt * self.vel

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        self.pos_before_clip = self.pos.copy()
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        self.crashed_wall = not np.array_equal(self.pos_before_clip[:2], self.pos[:2])
        self.crashed_ceiling = self.pos_before_clip[2] > self.pos[2]

        # `sum_thr_drag` 是机体系里的总升力与旋翼阻力合力；
        # 地面接触函数会把它再转回世界系，决定撞地后的支撑、摩擦和是否允许重新起飞。
        sum_thr_drag = thrust + rotor_drag_force
        self.floor_interaction(sum_thr_drag=sum_thr_drag)

        # Computing velocities
        self.vel = (1.0 - self.vel_damp) * self.vel + dt * self.acc

        # Accelerometer measures so-called "proper acceleration"
        # that includes gravity with the opposite sign
        self.accelerometer = np.matmul(self.rot.T, self.acc + [0, 0, self.gravity])

    def step1_numba(self, thrust_cmds, dt, thrust_noise):
        # numba 路径与 `step1()` 语义一致，只是把高频数值部分提前编译以提高多机训练吞吐。
        # 它拆成三个阶段：刚体积分、地面接触、速度/加速度计更新。
        self.thrust_rot_damp, self.thrust_cmds_damp, self.torques, self.torque, self.rot, self.since_last_svd, \
            self.omega_dot, self.omega, self.pos, thrust, rotor_drag_force, self.vel = \
            calculate_torque_integrate_rotations_and_update_omega(
                thrust_cmds=thrust_cmds, motor_tau_up=self.motor_tau_up, motor_tau_down=self.motor_tau_down,
                thrust_cmds_damp=self.thrust_cmds_damp, thrust_rot_damp=self.thrust_rot_damp, thr_noise=thrust_noise,
                thrust_max=self.thrust_max, motor_linearity=self.motor_linearity,
                prop_crossproducts=self.prop_crossproducts, torque_max=self.torque_max, prop_ccw=self.prop_ccw,
                rot=self.rot, omega=np.float64(self.omega), dt=self.dt, since_last_svd=self.since_last_svd,
                since_last_svd_limit=self.since_last_svd_limit, inertia=self.inertia, eye=self.eye,
                omega_max=self.omega_max, damp_omega_quadratic=self.damp_omega_quadratic, pos=self.pos, vel=self.vel)

        pos_before_clip = np.array(self.pos)

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        # Detect collision with walls
        self.crashed_wall = not np.array_equal(pos_before_clip[:2], self.pos[:2])
        self.crashed_ceiling = pos_before_clip[2] > self.pos[2]

        # Set constant variables up for numba
        sum_thr_drag = thrust + rotor_drag_force
        grav_arr = np.float64([0, 0, self.gravity])

        # self.floor_interaction(sum_thr_drag=sum_thr_drag)
        self.pos, self.vel, self.acc, self.omega, self.rot, self.thrust_cmds_damp, self.thrust_rot_damp, \
            self.on_floor, self.crashed_floor = floor_interaction_numba(
                pos=self.pos, vel=self.vel, rot=self.rot, omega=self.omega, mu=self.mu, mass=self.mass,
                sum_thr_drag=sum_thr_drag, thrust_cmds_damp=self.thrust_cmds_damp, thrust_rot_damp=self.thrust_rot_damp,
                floor_threshold=self.arm, on_floor=self.on_floor)

        # compute_velocity_and_acceleration(vel, vel_damp, dt, rot_tpose, grav_arr, acc):
        self.vel, self.accelerometer = compute_velocity_and_acceleration(vel=self.vel, vel_damp=self.vel_damp, dt=dt,
                                                                         rot_tpose=self.rot.T, grav_arr=grav_arr,
                                                                         acc=self.acc)

    def reset(self):
        # reset 只清电机内部滞后状态，不负责位置姿态初始化；
        # 真正的初始位置/速度写入由 `set_state()` 完成。
        self.thrust_cmds_damp = np.zeros([4])
        self.thrust_rot_damp = np.zeros([4])

    def floor_interaction(self, sum_thr_drag):
        # 这里集中处理“撞地”和“地面滑动/重新起飞”。
        # 多机环境的其它碰撞是离散碰撞模块负责，而地面接触直接内嵌在动力学层，因为它会持续改变姿态、速度和推力累积状态。
        self.crashed_floor = False
        if self.pos[2] <= self.floor_threshold:
            self.pos = np.array((self.pos[0], self.pos[1], self.floor_threshold))
            force = np.matmul(self.rot, sum_thr_drag)
            if self.on_floor:
                # 已经在地面上时，把姿态压回竖直基准，并用摩擦限制水平滑动。
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                c, s = np.cos(theta), np.sin(theta)
                self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                # Add friction if drone is on the floor
                force_xy = np.array([force[0], force[1]])
                force_xy_magn = np.linalg.norm(force_xy)
                friction_xy_magn = self.mu * (self.mass * GRAV - force[2])

                if np.linalg.norm(self.vel) == 0.0:
                    force_xy_magn = max(force_xy_magn - friction_xy_magn, 0.)
                    if force_xy_magn == 0.:
                        force[0] = 0.
                        force[1] = 0.
                    else:
                        force_angle = np.arctan2(force[1], force[0])
                        force_xy_dir = np.array([np.cos(force_angle), np.sin(force_angle)])
                        force_xy = force_xy_magn * force_xy_dir
                        force[0] = force_xy[0]
                        force[1] = force_xy[1]
                else:
                    # vel > 0, friction direction is opposite to velocity direction
                    friction_xy_angle = np.arctan2(-1.0 * self.vel[1], -1.0 * self.vel[0])
                    friction_xy_dir = np.array([np.cos(friction_xy_angle), np.sin(friction_xy_angle)])
                    force[0] = force[0] - friction_xy_dir[0] * friction_xy_magn
                    force[1] = force[1] - friction_xy_dir[1] * friction_xy_magn
            else:
                # 这一分支表示“本步首次撞地”。
                # 这里会清掉速度、角速度和电机累积量，避免上一时刻的冲量在地面状态下继续传播。
                self.on_floor = True
                self.crashed_floor = True
                # Set vel to [0, 0, 0]
                self.vel, self.acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
                self.omega = np.zeros(3, dtype=np.float32)
                # Set rot
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                c, s = np.cos(theta), np.sin(theta)
                if self.rot[2, 2] < 0:
                    self.rot = randyaw()
                    while np.dot(self.rot[:, 0], to_xyhat(-self.pos)) < 0.5:
                        self.rot = randyaw()
                else:
                    self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                self.set_state(self.pos, self.vel, self.rot, self.omega)

                # reset momentum / accumulation of thrust
                self.thrust_cmds_damp = np.zeros([4])
                self.thrust_rot_damp = np.zeros([4])

            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force
            self.acc[2] = np.maximum(0, self.acc[2])
        else:
            # self.pos[2] > self.floor_threshold
            if self.on_floor:
                # Drone is in the air, while on_floor flag still True
                self.on_floor = False

            # Computing accelerations
            force = np.matmul(self.rot, sum_thr_drag)
            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force

    def look_at(self):
        # 渲染器通过这个相机位姿追踪当前无人机。
        # 它不参与训练，但可视化调试时会直接反映动力学状态是否异常。
        degrees_down = 45.0
        R = self.rot
        # camera slightly below COM
        eye = self.pos + np.matmul(R, [0, 0, -0.02])
        theta = np.radians(degrees_down)
        to, _ = normalize(np.cos(theta) * R[:, 0] - np.sin(theta) * R[:, 2])
        center = eye + to
        up = cross(to, R[:, 1])
        return eye, center, up

    def state_vector(self):
        # 这是最底层的完整物理状态展平形式。
        # 单机环境的若干 `state_*` 观测函数会从这些量中挑选并重组出训练输入。
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    @staticmethod
    def action_space():
        low = np.zeros(4)
        high = np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    def __deepcopy__(self, memo):
        """Certain numba-optimized instance attributes can't be naively copied."""

        cls = self.__class__
        copied_dynamics = cls.__new__(cls)
        memo[id(self)] = copied_dynamics

        skip_copying = {"thrust_noise"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_dynamics, k, deepcopy(v, memo))

        copied_dynamics.init_thrust_noise()
        return copied_dynamics


@njit
def calculate_torque_integrate_rotations_and_update_omega(
        thrust_cmds, motor_tau_up, motor_tau_down, thrust_cmds_damp, thrust_rot_damp, thr_noise, thrust_max,
        motor_linearity, prop_crossproducts, torque_max, prop_ccw, rot, omega, dt, since_last_svd, since_last_svd_limit,
        inertia, eye, omega_max, damp_omega_quadratic, pos, vel
):
    # 这是 `step1()` 前半段的 numba 版本：
    # 从电机滞后、推力/力矩计算一直做到姿态和角速度积分，但还不处理地面接触。
    thrust_cmds = np.clip(thrust_cmds, 0., 1.)
    motor_tau = motor_tau_up * np.ones(4)
    motor_tau[thrust_cmds < thrust_cmds_damp] = np.array(motor_tau_down)
    motor_tau[motor_tau > 1.] = 1.

    # Since NN commands thrusts we need to convert to rot vel and back
    thrust_rot = thrust_cmds ** 0.5
    thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp ** 2

    # Adding noise
    thrust_noise = thrust_cmds * thr_noise
    thrust_cmds_damp = np.clip(thrust_cmds_damp + thrust_noise, 0.0, 1.0)
    thrusts = thrust_max * angvel2thrust_numba(thrust_cmds_damp, motor_linearity)

    # Prop cross-product gives torque directions
    torques = prop_crossproducts * np.reshape(thrusts, (-1, 1))

    # Additional torques along z-axis caused by propeller rotations
    torques[:, 2] += torque_max * prop_ccw * thrust_cmds_damp

    # Net torque: sum over propellers
    thrust_torque = np.sum(torques, 0)

    # Rotor drag and Rolling forces and moments
    rotor_visc_torque = rotor_drag_force = np.zeros(3)

    # (Square) Damping using torques (in case we would like to add damping using torques)
    torque = thrust_torque + rotor_visc_torque
    thrust = np.array([0., 0., np.sum(thrusts)])

    # 这一段与 Python 路径一致，负责把机体系角速度积分成新的旋转矩阵。
    # Integrating rotations (based on current values)
    omega_vec = rot @ omega
    wx, wy, wz = omega_vec
    omega_norm = np.linalg.norm(omega_vec)
    if omega_norm != 0:
        K = np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]]) / omega_norm
        rot_angle = omega_norm * dt
        dRdt = eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        rot = dRdt @ rot

    # SVD is not strictly required anymore. Performing it rarely, just in case
    since_last_svd += dt
    if since_last_svd > since_last_svd_limit:
        u, s, v = np.linalg.svd(rot)
        rot = u @ v
        since_last_svd = 0

    # COMPUTING OMEGA UPDATE
    # Linear damping
    omega_dot = ((1.0 / inertia) * (numba_cross(-omega, inertia * omega) + torque))

    # Quadratic damping
    omega_damp_quadratic = np.clip(damp_omega_quadratic * omega ** 2, 0.0, 1.0)
    omega = omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
    omega = np.clip(omega, -omega_max, omega_max)

    # Computing position
    pos = pos + dt * vel

    return thrust_rot_damp, thrust_cmds_damp, torques, torque, rot, since_last_svd, omega_dot, omega, pos, thrust, \
        rotor_drag_force, vel


@njit
def floor_interaction_numba(pos, vel, rot, omega, mu, mass, sum_thr_drag, thrust_cmds_damp, thrust_rot_damp,
                            floor_threshold, on_floor):
    # 这是地面交互的 numba 版本。
    # 语义与 `floor_interaction()` 保持一致：首次撞地时清动量，落地后持续施加摩擦与姿态约束。
    crashed_floor = False
    if pos[2] <= floor_threshold:
        pos = np.array((pos[0], pos[1], floor_threshold))
        force = rot @ sum_thr_drag
        if on_floor:
            # Drone is on the floor, and on_floor flag still True
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # Add friction if drone is on the floor
            friction_xy_magn = mu * (mass * GRAV - force[2])

            if np.linalg.norm(vel) < EPS:
                force_xy = np.array([force[0], force[1]])
                force_xy_magn = np.linalg.norm(force_xy)
                force_xy_magn = max(force_xy_magn - friction_xy_magn, 0.)
                if force_xy_magn == 0.:
                    force[0] = 0.
                    force[1] = 0.
                else:
                    force_angle = np.arctan2(force[1], force[0])
                    force_xy_dir = np.array([np.cos(force_angle), np.sin(force_angle)])
                    force_xy = force_xy_magn * force_xy_dir
                    force[0] = force_xy[0]
                    force[1] = force_xy[1]
            else:
                # vel > 0, friction direction is opposite to velocity direction
                friction_xy_angle = np.arctan2(vel[1], vel[0])
                friction_xy_dir = np.array([np.cos(friction_xy_angle), np.sin(friction_xy_angle)])
                force[0] = force[0] - friction_xy_dir[0] * friction_xy_magn
                force[1] = force[1] - friction_xy_dir[1] * friction_xy_magn
        else:
            # Previous step, drone still in the air, but in this step, it hits the floor
            # In previous step, self.on_floor = False, self.crashed_floor = False
            on_floor = True
            crashed_floor = True
            # Set vel to [0, 0, 0]
            vel, acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            omega = np.zeros(3, dtype=np.float64)
            # Set rot
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            c, s = np.cos(theta), np.sin(theta)
            if rot[2, 2] < 0:
                theta = np.random.uniform(-np.pi, np.pi)
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            else:
                rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # reset momentum / accumulation of thrust
            thrust_cmds_damp = np.zeros(4)
            thrust_rot_damp = np.zeros(4)

        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force
        acc[2] = np.maximum(0, acc[2])
    else:
        # self.pos[2] > self.floor_threshold
        if on_floor:
            # Drone is in the air, while on_floor flag still True
            on_floor = False

        # Computing accelerations
        force = rot @ sum_thr_drag
        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force

    return pos, vel, acc, omega, rot, thrust_cmds_damp, thrust_rot_damp, on_floor, crashed_floor


@njit
def compute_velocity_and_acceleration(vel, vel_damp, dt, rot_tpose, grav_arr, acc):
    # 最后一步把世界系加速度写回速度，并生成机体系加速度计读数。
    # 这些读数会被观测函数和 sim2real 相关模块继续消费。
    vel = (1.0 - vel_damp) * vel + dt * acc

    # Accelerometer measures so-called "proper acceleration" that includes gravity with the opposite sign
    accm = rot_tpose @ (acc + grav_arr)
    return vel, accm
