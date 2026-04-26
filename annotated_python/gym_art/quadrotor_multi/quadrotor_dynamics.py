# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_dynamics.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from copy import deepcopy

# 导入当前模块依赖。
import numpy as np
from gymnasium import spaces
from numba import njit

# 导入当前模块依赖。
from gym_art.quadrotor_multi.inertia import QuadLink, QuadLinkSimplified
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba, angvel2thrust_numba, numba_cross
from gym_art.quadrotor_multi.quad_utils import OUNoise, rand_uniform_rot3d, cross_vec_mx4, cross_mx4, npa, cross, \
    # 执行这一行逻辑。
    randyaw, to_xyhat, normalize

# 保存或更新 `GRAV` 的值。
GRAV = 9.81  # default gravitational constant
# 保存或更新 `EPS` 的值。
EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


# WARN:
# linearity is set to 1 always, by means of check_quad_param_limits().
# The def. value of linearity for CF is set to 1 as well (due to firmware non-linearity compensation)
# 定义类 `QuadrotorDynamics`。
class QuadrotorDynamics:
    # 下面开始文档字符串说明。
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

    # 定义函数 `__init__`。
    def __init__(self, model_params, room_box=None, dynamics_steps_num=1, dim_mode="3D", gravity=GRAV,
                 # 保存或更新 `dynamics_simplification` 的值。
                 dynamics_simplification=False, use_numba=False, dt=1/200):
        # Pre-set Parameters
        # 保存或更新 `dt` 的值。
        self.dt = dt
        # 保存或更新 `use_numba` 的值。
        self.use_numba = use_numba

        # Dynamics
        # 保存或更新 `dynamics_steps_num` 的值。
        self.dynamics_steps_num = dynamics_steps_num
        # 保存或更新 `dynamics_simplification` 的值。
        self.dynamics_simplification = dynamics_simplification
        # cw = 1 ; ccw = -1 [ccw, cw, ccw, cw]
        # 保存或更新 `prop_ccw` 的值。
        self.prop_ccw = np.array([-1., 1., -1., 1.])
        # Reference: https://docs.google.com/document/d/1wZMZQ6jilDbj0JtfeYt0TonjxoMPIgHwYbrFrMNls84/edit
        # 保存或更新 `omega_max` 的值。
        self.omega_max = 40.  # rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        # 保存或更新 `vxyz_max` 的值。
        self.vxyz_max = 3.  # m/s
        # 保存或更新 `gravity` 的值。
        self.gravity = gravity
        # 保存或更新 `acc_max` 的值。
        self.acc_max = 3. * GRAV
        # 保存或更新 `since_last_svd` 的值。
        self.since_last_svd = 0  # counter
        # 保存或更新 `since_last_svd_limit` 的值。
        self.since_last_svd_limit = 0.5  # in sec - how often mandatory orthogonality should be applied
        # 保存或更新 `eye` 的值。
        self.eye = np.eye(3)
        # Initializing model
        # 保存或更新 `thrust_noise` 的值。
        self.thrust_noise = None
        # 调用 `update_model` 执行当前处理。
        self.update_model(model_params)
        # Sanity checks
        # 断言当前条件成立，用于保护运行假设。
        assert self.inertia.shape == (3,)

        # Dynamics used in step
        # 保存或更新 `motor_tau_up` 的值。
        self.motor_tau_up = 4 * dt / (self.motor_damp_time_up + EPS)
        # 保存或更新 `motor_tau_down` 的值。
        self.motor_tau_down = 4 * dt / (self.motor_damp_time_down + EPS)

        # Room
        # 根据条件决定是否进入当前分支。
        if room_box is None:
            # 保存或更新 `room_box` 的值。
            self.room_box = np.array([[0., 0., 0.], [10., 10., 10.]])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `room_box` 的值。
            self.room_box = np.array(room_box).copy()

        # # Floor
        # 保存或更新 `on_floor` 的值。
        self.on_floor = False
        # # # If pos_z smaller than this threshold, we assume that drone collide with the floor
        # 保存或更新 `floor_threshold` 的值。
        self.floor_threshold = 0.05
        # # # Floor Fiction
        # 保存或更新 `mu` 的值。
        self.mu = 0.6
        # # # Collision with room
        # 保存或更新 `crashed_wall` 的值。
        self.crashed_wall = False
        # 保存或更新 `crashed_ceiling` 的值。
        self.crashed_ceiling = False
        # 保存或更新 `crashed_floor` 的值。
        self.crashed_floor = False

        # Selecting 1D, Planar or Full 3D modes
        # 保存或更新 `dim_mode` 的值。
        self.dim_mode = dim_mode
        # 根据条件决定是否进入当前分支。
        if self.dim_mode == '1D':
            # 保存或更新 `control_mx` 的值。
            self.control_mx = np.ones([4, 1])
        # 当上一分支不满足时，继续判断新的条件。
        elif self.dim_mode == '2D':
            # 保存或更新 `control_mx` 的值。
            self.control_mx = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        # 当上一分支不满足时，继续判断新的条件。
        elif self.dim_mode == '3D':
            # 保存或更新 `control_mx` 的值。
            self.control_mx = np.eye(4)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)

    # 为下面的函数或方法附加装饰器行为。
    @staticmethod
    # 定义函数 `angvel2thrust`。
    def angvel2thrust(w, linearity=0.424):
        # 下面开始文档字符串说明。
        """
        CrazyFlie: linearity=0.424
        Args:
            w: thrust_cmds_damp
            linearity (float): linearity factor factor [0 .. 1].
        """
        # 返回当前函数的结果。
        return (1 - linearity) * w ** 2 + linearity * w

    # 定义函数 `update_model`。
    def update_model(self, model_params):
        # 根据条件决定是否进入当前分支。
        if self.dynamics_simplification:
            # 保存或更新 `model` 的值。
            self.model = QuadLinkSimplified(params=model_params["geom"])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `model` 的值。
            self.model = QuadLink(params=model_params["geom"])
        # 保存或更新 `model_params` 的值。
        self.model_params = model_params

        # PARAMETERS FOR RANDOMIZATION
        # 保存或更新 `mass` 的值。
        self.mass = self.model.m
        # 保存或更新 `inertia` 的值。
        self.inertia = np.diagonal(self.model.I_com)

        # 保存或更新 `thrust_to_weight` 的值。
        self.thrust_to_weight = self.model_params["motor"]["thrust_to_weight"]
        # 保存或更新 `torque_to_thrust` 的值。
        self.torque_to_thrust = self.model_params["motor"]["torque_to_thrust"]
        # 保存或更新 `motor_linearity` 的值。
        self.motor_linearity = self.model_params["motor"]["linearity"]
        # 保存或更新 `C_rot_drag` 的值。
        self.C_rot_drag = self.model_params["motor"]["C_drag"]
        # 保存或更新 `C_rot_roll` 的值。
        self.C_rot_roll = self.model_params["motor"]["C_roll"]
        # 保存或更新 `motor_damp_time_up` 的值。
        self.motor_damp_time_up = self.model_params["motor"]["damp_time_up"]
        # 保存或更新 `motor_damp_time_down` 的值。
        self.motor_damp_time_down = self.model_params["motor"]["damp_time_down"]

        # 保存或更新 `thrust_noise_ratio` 的值。
        self.thrust_noise_ratio = self.model_params["noise"]["thrust_noise_ratio"]
        # 保存或更新 `vel_damp` 的值。
        self.vel_damp = self.model_params["damp"]["vel"]
        # 保存或更新 `damp_omega_quadratic` 的值。
        self.damp_omega_quadratic = self.model_params["damp"]["omega_quadratic"]

        # COMPUTED (Dependent) PARAMETERS
        # 尝试执行下面的逻辑，并为异常情况做准备。
        try:
            # 保存或更新 `motor_assymetry` 的值。
            self.motor_assymetry = np.array(self.model_params["motor"]["assymetry"])
        # 捕获前面代码可能抛出的异常。
        except:
            # 保存或更新 `motor_assymetry` 的值。
            self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
            # 调用 `print` 执行当前处理。
            print("WARNING: Motor assymetry was not setup. Setting assymetry to:", self.motor_assymetry)
        # 保存或更新 `motor_assymetry` 的值。
        self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)  # re-normalizing to sum-up to 4
        # 保存或更新 `thrust_max` 的值。
        self.thrust_max = GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        # 保存或更新 `torque_max` 的值。
        self.torque_max = self.torque_to_thrust * self.thrust_max  # propeller torque scales

        # Propeller positions in X configurations
        # 保存或更新 `prop_pos` 的值。
        self.prop_pos = self.model.prop_pos

        # unit: meters^2 ??? maybe wrong
        # 保存或更新 `prop_crossproducts` 的值。
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        # 保存或更新 `prop_ccw_mx` 的值。
        self.prop_ccw_mx = np.zeros([3, 4])  # Matrix allows using matrix multiplication
        # 保存或更新 `prop_ccw_mx[2, :]` 的值。
        self.prop_ccw_mx[2, :] = self.prop_ccw

        # Forced dynamics auxiliary matrices
        # Prop crossproduct give torque directions
        # 保存或更新 `G_omega_thrust` 的值。
        self.G_omega_thrust = self.thrust_max * self.prop_crossproducts.T  # [3,4] @ [4,1]
        # additional torques along z-axis caused by propeller rotations
        # 保存或更新 `C_omega_prop` 的值。
        self.C_omega_prop = self.torque_max * self.prop_ccw_mx  # [3,4] @ [4,1] = [3,1]
        # 保存或更新 `G_omega` 的值。
        self.G_omega = (1.0 / self.inertia)[:, None] * (self.G_omega_thrust + self.C_omega_prop)

        # Allows to sum-up thrusts as a linear matrix operation
        # 保存或更新 `thrust_sum_mx` 的值。
        self.thrust_sum_mx = np.zeros([3, 4])  # [0,0,F_sum].T
        # 保存或更新 `thrust_sum_mx[2, :]` 的值。
        self.thrust_sum_mx[2, :] = 1  # [0,0,F_sum].T

        # 调用 `init_thrust_noise` 执行当前处理。
        self.init_thrust_noise()

        # 保存或更新 `arm` 的值。
        self.arm = np.linalg.norm(self.model.motor_xyz[:2])

        # the ratio between max torque and inertia around each axis
        # the 0-1 matrix on the right is the way to sum-up
        # 保存或更新 `torque_to_inertia` 的值。
        self.torque_to_inertia = self.G_omega @ np.array([[0., 0., 0.], [0., 1., 1.], [1., 1., 0.], [1., 0., 1.]])
        # 保存或更新 `torque_to_inertia` 的值。
        self.torque_to_inertia = np.sum(self.torque_to_inertia, axis=1)
        # self.torque_to_inertia = self.torque_to_inertia / np.linalg.norm(self.torque_to_inertia)

        # 调用 `reset` 执行当前处理。
        self.reset()

    # 定义函数 `init_thrust_noise`。
    def init_thrust_noise(self):
        # sigma = 0.2 gives roughly max noise of -1 ... 1
        # 根据条件决定是否进入当前分支。
        if self.use_numba:
            # 保存或更新 `thrust_noise` 的值。
            self.thrust_noise = OUNoiseNumba(4, sigma=0.2 * self.thrust_noise_ratio)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `thrust_noise` 的值。
            self.thrust_noise = OUNoise(4, sigma=0.2 * self.thrust_noise_ratio)

    # pos, vel, in world coordinates (meters)
    # rotation is 3x3 matrix (body coordinates) -> (world coordinates)dt
    # omega is angular velocity (radians/sec) in body coordinates, i.e. the gyroscope
    # 定义函数 `set_state`。
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for v in (position, velocity, omega):
            # 断言当前条件成立，用于保护运行假设。
            assert v.shape == (3,)
        # 断言当前条件成立，用于保护运行假设。
        assert thrusts.shape == (4,)
        # 断言当前条件成立，用于保护运行假设。
        assert rotation.shape == (3, 3)
        # 保存或更新 `pos` 的值。
        self.pos = deepcopy(position)
        # 保存或更新 `vel` 的值。
        self.vel = deepcopy(velocity)
        # 保存或更新 `acc` 的值。
        self.acc = np.zeros(3)
        # 保存或更新 `accelerometer` 的值。
        self.accelerometer = np.array([0, 0, GRAV])
        # 保存或更新 `rot` 的值。
        self.rot = deepcopy(rotation)
        # 保存或更新 `omega` 的值。
        self.omega = deepcopy(omega.astype(np.float32))
        # 保存或更新 `thrusts` 的值。
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)
    # 为下面的函数或方法附加装饰器行为。
    @staticmethod
    # 定义函数 `random_state`。
    def random_state(box, vel_max=15.0, omega_max=2 * np.pi):
        # 保存或更新 `box` 的值。
        box = np.array(box)
        # 保存或更新 `pos` 的值。
        pos = np.random.uniform(low=-box, high=box, size=(3,))

        # 保存或更新 `vel` 的值。
        vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
        # 保存或更新 `vel_magn` 的值。
        vel_magn = np.random.uniform(low=0., high=vel_max)
        # 保存或更新 `vel` 的值。
        vel = vel_magn / (np.linalg.norm(vel) + EPS) * vel

        # 保存或更新 `omega` 的值。
        omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
        # 保存或更新 `omega_magn` 的值。
        omega_magn = np.random.uniform(low=0., high=omega_max)
        # 保存或更新 `omega` 的值。
        omega = omega_magn / (np.linalg.norm(omega) + EPS) * omega

        # 保存或更新 `rot` 的值。
        rot = rand_uniform_rot3d()
        # 返回当前函数的结果。
        return pos, vel, rot, omega

    # 定义函数 `step`。
    def step(self, thrust_cmds, dt):
        # 保存或更新 `thrust_noise` 的值。
        thrust_noise = self.thrust_noise.noise()

        # 根据条件决定是否进入当前分支。
        if self.use_numba:
            # 执行这一行逻辑。
            [self.step1_numba(thrust_cmds, dt, thrust_noise) for _ in range(self.dynamics_steps_num)]
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 执行这一行逻辑。
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
    # 定义函数 `step1`。
    def step1(self, thrust_cmds, dt, thrust_noise):
        # 保存或更新 `thrust_cmds` 的值。
        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)

        # Filtering the thruster and adding noise
        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        # 保存或更新 `motor_tau_down` 的值。
        motor_tau_down = np.array(self.motor_tau_down)
        # 保存或更新 `motor_tau` 的值。
        motor_tau = self.motor_tau_up * np.ones([4, ])
        # 保存或更新 `motor_tau[thrust_cmds < thrust_cmds_damp]` 的值。
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = motor_tau_down
        # 保存或更新 `motor_tau[motor_tau > 1.]` 的值。
        motor_tau[motor_tau > 1.] = 1.

        # Since NN commands thrusts we need to convert to rot vel and back
        # WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
        # since it likely means that you are using rotational velocities as an input instead of the thrust and hence
        # you are filtering square roots of angular velocities
        # 保存或更新 `thrust_rot` 的值。
        thrust_rot = thrust_cmds ** 0.5
        # 保存或更新 `thrust_rot_damp` 的值。
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        # 保存或更新 `thrust_cmds_damp` 的值。
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        # Adding noise
        # 保存或更新 `thrust_noise` 的值。
        thrust_noise = thrust_cmds * thrust_noise
        # 保存或更新 `thrust_cmds_damp` 的值。
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        # 保存或更新 `thrusts` 的值。
        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        # Prop crossproduct give torque directions
        # 保存或更新 `torques` 的值。
        self.torques = self.prop_crossproducts * thrusts[:, None]  # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        # 保存或更新 `torques[:, 2]` 的值。
        self.torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp

        # net torque: sum over propellers
        # 保存或更新 `thrust_torque` 的值。
        thrust_torque = np.sum(self.torques, axis=0)

        # Rotor drag and Rolling forces and moments
        # See Ref[1] Sec:2.1 for details
        # 根据条件决定是否进入当前分支。
        if self.C_rot_drag != 0 or self.C_rot_roll != 0:
            # 保存或更新 `vel_body` 的值。
            vel_body = self.rot.T @ self.vel
            # 保存或更新 `v_rotor` 的值。
            v_rotor = vel_body + cross_vec_mx4(self.omega, self.model.prop_pos)
            # 保存或更新 `v_rotor[:, 2]` 的值。
            v_rotor[:, 2] = 0.  # Projection to the rotor plane

            # Drag/Roll of rotors (both in body frame)
            # 保存或更新 `rotor_drag_fi` 的值。
            rotor_drag_fi = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotor
            # 保存或更新 `rotor_drag_force` 的值。
            rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
            # 保存或更新 `rotor_drag_ti` 的值。
            rotor_drag_ti = cross_mx4(rotor_drag_fi, self.model.prop_pos)
            # 保存或更新 `rotor_drag_torque` 的值。
            rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

            # 保存或更新 `rotor_roll_torque` 的值。
            rotor_roll_torque = \
                # 执行这一行逻辑。
                - self.C_rot_roll * self.prop_ccw[:, None] * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotor
            # 保存或更新 `rotor_roll_torque` 的值。
            rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
            # 保存或更新 `rotor_visc_torque` 的值。
            rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

            # Constraints (prevent numerical instabilities)
            # 保存或更新 `vel_norm` 的值。
            vel_norm = np.linalg.norm(vel_body)
            # 保存或更新 `rdf_norm` 的值。
            rdf_norm = np.linalg.norm(rotor_drag_force)
            # 保存或更新 `rdf_norm_clip` 的值。
            rdf_norm_clip = np.clip(rdf_norm, a_min=0., a_max=vel_norm * self.mass / (2 * dt))
            # 根据条件决定是否进入当前分支。
            if rdf_norm > EPS:
                # 保存或更新 `rotor_drag_force` 的值。
                rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

            # omega_norm = np.linalg.norm(self.omega)
            # 保存或更新 `rvt_norm` 的值。
            rvt_norm = np.linalg.norm(rotor_visc_torque)
            # 保存或更新 `rvt_norm_clipped` 的值。
            rvt_norm_clipped = np.clip(rvt_norm, a_min=0., a_max=np.linalg.norm(self.omega * self.inertia) / (2 * dt))
            # 根据条件决定是否进入当前分支。
            if rvt_norm > EPS:
                # 保存或更新 `rotor_visc_torque` 的值。
                rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `rotor_visc_torque` 的值。
            rotor_visc_torque = rotor_drag_force = np.zeros(3)

        # (Square) Damping using torques (in case we would like to add damping using torques)
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        # 保存或更新 `torque` 的值。
        self.torque = thrust_torque + rotor_visc_torque
        # 保存或更新 `thrust` 的值。
        thrust = npa(0, 0, np.sum(thrusts))

        # ROTATIONAL DYNAMICS
        # Integrating rotations (based on current values)
        # 保存或更新 `omega_vec` 的值。
        omega_vec = np.matmul(self.rot, self.omega)  # Change from body to world frame
        # 同时更新 `wx`, `wy`, `wz` 等变量。
        wx, wy, wz = omega_vec
        # 保存或更新 `omega_norm` 的值。
        omega_norm = np.linalg.norm(omega_vec)
        # 根据条件决定是否进入当前分支。
        if omega_norm != 0:
            # See [7]
            # 保存或更新 `K` 的值。
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            # 保存或更新 `rot_angle` 的值。
            rot_angle = omega_norm * dt
            # 保存或更新 `dRdt` 的值。
            dRdt = self.eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
            # 保存或更新 `rot` 的值。
            self.rot = dRdt @ self.rot

        # SVD is not strictly required anymore. Performing it rarely, just in case
        # 保存或更新 `since_last_svd` 的值。
        self.since_last_svd += dt
        # 根据条件决定是否进入当前分支。
        if self.since_last_svd > self.since_last_svd_limit:
            # Perform SVD orthogonolization
            # 同时更新 `u`, `s`, `v` 等变量。
            u, s, v = np.linalg.svd(self.rot)
            # 保存或更新 `rot` 的值。
            self.rot = np.matmul(u, v)
            # 保存或更新 `since_last_svd` 的值。
            self.since_last_svd = 0

        # COMPUTING OMEGA UPDATE
        # Damping using velocities (I find it more stable numerically)
        # This is only for linear damping of angular velocity.
        # 保存或更新 `omega_dot` 的值。
        self.omega_dot = ((1.0 / self.inertia) * (cross(-self.omega, self.inertia * self.omega) + self.torque))

        # Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        # 保存或更新 `omega_damp_quadratic` 的值。
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * self.omega ** 2, a_min=0.0, a_max=1.0)
        # 保存或更新 `omega` 的值。
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * self.omega_dot
        # 保存或更新 `omega` 的值。
        self.omega = np.clip(self.omega, a_min=-self.omega_max, a_max=self.omega_max)

        # TRANSLATIONAL DYNAMICS
        # Computing position
        # 保存或更新 `pos` 的值。
        self.pos = self.pos + dt * self.vel

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        # 保存或更新 `pos_before_clip` 的值。
        self.pos_before_clip = self.pos.copy()
        # 保存或更新 `pos` 的值。
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        # 保存或更新 `crashed_wall` 的值。
        self.crashed_wall = not np.array_equal(self.pos_before_clip[:2], self.pos[:2])
        # 保存或更新 `crashed_ceiling` 的值。
        self.crashed_ceiling = self.pos_before_clip[2] > self.pos[2]

        # 保存或更新 `sum_thr_drag` 的值。
        sum_thr_drag = thrust + rotor_drag_force
        # 保存或更新 `floor_interaction(sum_thr_drag` 的值。
        self.floor_interaction(sum_thr_drag=sum_thr_drag)

        # Computing velocities
        # 保存或更新 `vel` 的值。
        self.vel = (1.0 - self.vel_damp) * self.vel + dt * self.acc

        # Accelerometer measures so-called "proper acceleration"
        # that includes gravity with the opposite sign
        # 保存或更新 `accelerometer` 的值。
        self.accelerometer = np.matmul(self.rot.T, self.acc + [0, 0, self.gravity])

    # 定义函数 `step1_numba`。
    def step1_numba(self, thrust_cmds, dt, thrust_noise):
        # 执行这一行逻辑。
        self.thrust_rot_damp, self.thrust_cmds_damp, self.torques, self.torque, self.rot, self.since_last_svd, \
            # 同时更新 `omega_dot`, `omega`, `pos`, `thrust` 等变量。
            self.omega_dot, self.omega, self.pos, thrust, rotor_drag_force, self.vel = \
            # 调用 `calculate_torque_integrate_rotations_and_update_omega` 执行当前处理。
            calculate_torque_integrate_rotations_and_update_omega(
                thrust_cmds=thrust_cmds, motor_tau_up=self.motor_tau_up, motor_tau_down=self.motor_tau_down,
                thrust_cmds_damp=self.thrust_cmds_damp, thrust_rot_damp=self.thrust_rot_damp, thr_noise=thrust_noise,
                thrust_max=self.thrust_max, motor_linearity=self.motor_linearity,
                prop_crossproducts=self.prop_crossproducts, torque_max=self.torque_max, prop_ccw=self.prop_ccw,
                rot=self.rot, omega=np.float64(self.omega), dt=self.dt, since_last_svd=self.since_last_svd,
                since_last_svd_limit=self.since_last_svd_limit, inertia=self.inertia, eye=self.eye,
                omega_max=self.omega_max, damp_omega_quadratic=self.damp_omega_quadratic, pos=self.pos, vel=self.vel)

        # 保存或更新 `pos_before_clip` 的值。
        pos_before_clip = np.array(self.pos)

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        # 保存或更新 `pos` 的值。
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        # Detect collision with walls
        # 保存或更新 `crashed_wall` 的值。
        self.crashed_wall = not np.array_equal(pos_before_clip[:2], self.pos[:2])
        # 保存或更新 `crashed_ceiling` 的值。
        self.crashed_ceiling = pos_before_clip[2] > self.pos[2]

        # Set constant variables up for numba
        # 保存或更新 `sum_thr_drag` 的值。
        sum_thr_drag = thrust + rotor_drag_force
        # 保存或更新 `grav_arr` 的值。
        grav_arr = np.float64([0, 0, self.gravity])

        # self.floor_interaction(sum_thr_drag=sum_thr_drag)
        # 执行这一行逻辑。
        self.pos, self.vel, self.acc, self.omega, self.rot, self.thrust_cmds_damp, self.thrust_rot_damp, \
            # 同时更新 `on_floor`, `crashed_floor` 等变量。
            self.on_floor, self.crashed_floor = floor_interaction_numba(
                pos=self.pos, vel=self.vel, rot=self.rot, omega=self.omega, mu=self.mu, mass=self.mass,
                sum_thr_drag=sum_thr_drag, thrust_cmds_damp=self.thrust_cmds_damp, thrust_rot_damp=self.thrust_rot_damp,
                floor_threshold=self.arm, on_floor=self.on_floor)

        # compute_velocity_and_acceleration(vel, vel_damp, dt, rot_tpose, grav_arr, acc):
        # 同时更新 `vel`, `accelerometer` 等变量。
        self.vel, self.accelerometer = compute_velocity_and_acceleration(vel=self.vel, vel_damp=self.vel_damp, dt=dt,
                                                                         rot_tpose=self.rot.T, grav_arr=grav_arr,
                                                                         acc=self.acc)

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `thrust_cmds_damp` 的值。
        self.thrust_cmds_damp = np.zeros([4])
        # 保存或更新 `thrust_rot_damp` 的值。
        self.thrust_rot_damp = np.zeros([4])

    # 定义函数 `floor_interaction`。
    def floor_interaction(self, sum_thr_drag):
        # Change pos, omega, rot, acc
        # 保存或更新 `crashed_floor` 的值。
        self.crashed_floor = False
        # 根据条件决定是否进入当前分支。
        if self.pos[2] <= self.floor_threshold:
            # 保存或更新 `pos` 的值。
            self.pos = np.array((self.pos[0], self.pos[1], self.floor_threshold))
            # 保存或更新 `force` 的值。
            force = np.matmul(self.rot, sum_thr_drag)
            # 根据条件决定是否进入当前分支。
            if self.on_floor:
                # Drone is on the floor, and on_floor flag still True
                # 保存或更新 `theta` 的值。
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                # 同时更新 `c`, `s` 等变量。
                c, s = np.cos(theta), np.sin(theta)
                # 保存或更新 `rot` 的值。
                self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                # Add friction if drone is on the floor
                # 保存或更新 `force_xy` 的值。
                force_xy = np.array([force[0], force[1]])
                # 保存或更新 `force_xy_magn` 的值。
                force_xy_magn = np.linalg.norm(force_xy)
                # 保存或更新 `friction_xy_magn` 的值。
                friction_xy_magn = self.mu * (self.mass * GRAV - force[2])

                # 根据条件决定是否进入当前分支。
                if np.linalg.norm(self.vel) == 0.0:
                    # 保存或更新 `force_xy_magn` 的值。
                    force_xy_magn = max(force_xy_magn - friction_xy_magn, 0.)
                    # 根据条件决定是否进入当前分支。
                    if force_xy_magn == 0.:
                        # 保存或更新 `force[0]` 的值。
                        force[0] = 0.
                        # 保存或更新 `force[1]` 的值。
                        force[1] = 0.
                    # 当前置条件都不满足时，执行兜底分支。
                    else:
                        # 保存或更新 `force_angle` 的值。
                        force_angle = np.arctan2(force[1], force[0])
                        # 保存或更新 `force_xy_dir` 的值。
                        force_xy_dir = np.array([np.cos(force_angle), np.sin(force_angle)])
                        # 保存或更新 `force_xy` 的值。
                        force_xy = force_xy_magn * force_xy_dir
                        # 保存或更新 `force[0]` 的值。
                        force[0] = force_xy[0]
                        # 保存或更新 `force[1]` 的值。
                        force[1] = force_xy[1]
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # vel > 0, friction direction is opposite to velocity direction
                    # 保存或更新 `friction_xy_angle` 的值。
                    friction_xy_angle = np.arctan2(-1.0 * self.vel[1], -1.0 * self.vel[0])
                    # 保存或更新 `friction_xy_dir` 的值。
                    friction_xy_dir = np.array([np.cos(friction_xy_angle), np.sin(friction_xy_angle)])
                    # 保存或更新 `force[0]` 的值。
                    force[0] = force[0] - friction_xy_dir[0] * friction_xy_magn
                    # 保存或更新 `force[1]` 的值。
                    force[1] = force[1] - friction_xy_dir[1] * friction_xy_magn
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # Previous step, drone still in the air, but in this step, it hits the floor
                # In previous step, self.on_floor = False, self.crashed_floor = False
                # 保存或更新 `on_floor` 的值。
                self.on_floor = True
                # 保存或更新 `crashed_floor` 的值。
                self.crashed_floor = True
                # Set vel to [0, 0, 0]
                # 同时更新 `vel`, `acc` 等变量。
                self.vel, self.acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
                # 保存或更新 `omega` 的值。
                self.omega = np.zeros(3, dtype=np.float32)
                # Set rot
                # 保存或更新 `theta` 的值。
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                # 同时更新 `c`, `s` 等变量。
                c, s = np.cos(theta), np.sin(theta)
                # 根据条件决定是否进入当前分支。
                if self.rot[2, 2] < 0:
                    # 保存或更新 `rot` 的值。
                    self.rot = randyaw()
                    # 在条件成立时持续执行下面的循环体。
                    while np.dot(self.rot[:, 0], to_xyhat(-self.pos)) < 0.5:
                        # 保存或更新 `rot` 的值。
                        self.rot = randyaw()
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # 保存或更新 `rot` 的值。
                    self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                # 调用 `set_state` 执行当前处理。
                self.set_state(self.pos, self.vel, self.rot, self.omega)

                # reset momentum / accumulation of thrust
                # 保存或更新 `thrust_cmds_damp` 的值。
                self.thrust_cmds_damp = np.zeros([4])
                # 保存或更新 `thrust_rot_damp` 的值。
                self.thrust_rot_damp = np.zeros([4])

            # 保存或更新 `acc` 的值。
            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force
            # 保存或更新 `acc[2]` 的值。
            self.acc[2] = np.maximum(0, self.acc[2])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # self.pos[2] > self.floor_threshold
            # 根据条件决定是否进入当前分支。
            if self.on_floor:
                # Drone is in the air, while on_floor flag still True
                # 保存或更新 `on_floor` 的值。
                self.on_floor = False

            # Computing accelerations
            # 保存或更新 `force` 的值。
            force = np.matmul(self.rot, sum_thr_drag)
            # 保存或更新 `acc` 的值。
            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force

    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `degrees_down` 的值。
        degrees_down = 45.0
        # 保存或更新 `R` 的值。
        R = self.rot
        # camera slightly below COM
        # 保存或更新 `eye` 的值。
        eye = self.pos + np.matmul(R, [0, 0, -0.02])
        # 保存或更新 `theta` 的值。
        theta = np.radians(degrees_down)
        # 同时更新 `to`, `_` 等变量。
        to, _ = normalize(np.cos(theta) * R[:, 0] - np.sin(theta) * R[:, 2])
        # 保存或更新 `center` 的值。
        center = eye + to
        # 保存或更新 `up` 的值。
        up = cross(to, R[:, 1])
        # 返回当前函数的结果。
        return eye, center, up

    # 定义函数 `state_vector`。
    def state_vector(self):
        # 返回当前函数的结果。
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    # 为下面的函数或方法附加装饰器行为。
    @staticmethod
    # 定义函数 `action_space`。
    def action_space():
        # 保存或更新 `low` 的值。
        low = np.zeros(4)
        # 保存或更新 `high` 的值。
        high = np.ones(4)
        # 返回当前函数的结果。
        return spaces.Box(low, high, dtype=np.float32)

    # 定义函数 `__deepcopy__`。
    def __deepcopy__(self, memo):
        # 下面的文档字符串用于说明当前模块或代码块。
        """Certain numba-optimized instance attributes can't be naively copied."""

        # 保存或更新 `cls` 的值。
        cls = self.__class__
        # 保存或更新 `copied_dynamics` 的值。
        copied_dynamics = cls.__new__(cls)
        # 保存或更新 `memo[id(self)]` 的值。
        memo[id(self)] = copied_dynamics

        # 保存或更新 `skip_copying` 的值。
        skip_copying = {"thrust_noise"}

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for k, v in self.__dict__.items():
            # 根据条件决定是否进入当前分支。
            if k not in skip_copying:
                # 调用 `setattr` 执行当前处理。
                setattr(copied_dynamics, k, deepcopy(v, memo))

        # 调用 `init_thrust_noise` 执行当前处理。
        copied_dynamics.init_thrust_noise()
        # 返回当前函数的结果。
        return copied_dynamics


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `calculate_torque_integrate_rotations_and_update_omega`。
def calculate_torque_integrate_rotations_and_update_omega(
        thrust_cmds, motor_tau_up, motor_tau_down, thrust_cmds_damp, thrust_rot_damp, thr_noise, thrust_max,
        motor_linearity, prop_crossproducts, torque_max, prop_ccw, rot, omega, dt, since_last_svd, since_last_svd_limit,
        inertia, eye, omega_max, damp_omega_quadratic, pos, vel
# 这里开始一个新的代码块。
):
    # Filtering the thruster and adding noise
    # 保存或更新 `thrust_cmds` 的值。
    thrust_cmds = np.clip(thrust_cmds, 0., 1.)
    # 保存或更新 `motor_tau` 的值。
    motor_tau = motor_tau_up * np.ones(4)
    # 保存或更新 `motor_tau[thrust_cmds < thrust_cmds_damp]` 的值。
    motor_tau[thrust_cmds < thrust_cmds_damp] = np.array(motor_tau_down)
    # 保存或更新 `motor_tau[motor_tau > 1.]` 的值。
    motor_tau[motor_tau > 1.] = 1.

    # Since NN commands thrusts we need to convert to rot vel and back
    # 保存或更新 `thrust_rot` 的值。
    thrust_rot = thrust_cmds ** 0.5
    # 保存或更新 `thrust_rot_damp` 的值。
    thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
    # 保存或更新 `thrust_cmds_damp` 的值。
    thrust_cmds_damp = thrust_rot_damp ** 2

    # Adding noise
    # 保存或更新 `thrust_noise` 的值。
    thrust_noise = thrust_cmds * thr_noise
    # 保存或更新 `thrust_cmds_damp` 的值。
    thrust_cmds_damp = np.clip(thrust_cmds_damp + thrust_noise, 0.0, 1.0)
    # 保存或更新 `thrusts` 的值。
    thrusts = thrust_max * angvel2thrust_numba(thrust_cmds_damp, motor_linearity)

    # Prop cross-product gives torque directions
    # 保存或更新 `torques` 的值。
    torques = prop_crossproducts * np.reshape(thrusts, (-1, 1))

    # Additional torques along z-axis caused by propeller rotations
    # 保存或更新 `torques[:, 2]` 的值。
    torques[:, 2] += torque_max * prop_ccw * thrust_cmds_damp

    # Net torque: sum over propellers
    # 保存或更新 `thrust_torque` 的值。
    thrust_torque = np.sum(torques, 0)

    # Rotor drag and Rolling forces and moments
    # 保存或更新 `rotor_visc_torque` 的值。
    rotor_visc_torque = rotor_drag_force = np.zeros(3)

    # (Square) Damping using torques (in case we would like to add damping using torques)
    # 保存或更新 `torque` 的值。
    torque = thrust_torque + rotor_visc_torque
    # 保存或更新 `thrust` 的值。
    thrust = np.array([0., 0., np.sum(thrusts)])

    # ROTATIONAL DYNAMICS
    # Integrating rotations (based on current values)
    # 保存或更新 `omega_vec` 的值。
    omega_vec = rot @ omega
    # 同时更新 `wx`, `wy`, `wz` 等变量。
    wx, wy, wz = omega_vec
    # 保存或更新 `omega_norm` 的值。
    omega_norm = np.linalg.norm(omega_vec)
    # 根据条件决定是否进入当前分支。
    if omega_norm != 0:
        # 保存或更新 `K` 的值。
        K = np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]]) / omega_norm
        # 保存或更新 `rot_angle` 的值。
        rot_angle = omega_norm * dt
        # 保存或更新 `dRdt` 的值。
        dRdt = eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        # 保存或更新 `rot` 的值。
        rot = dRdt @ rot

    # SVD is not strictly required anymore. Performing it rarely, just in case
    # 保存或更新 `since_last_svd` 的值。
    since_last_svd += dt
    # 根据条件决定是否进入当前分支。
    if since_last_svd > since_last_svd_limit:
        # 同时更新 `u`, `s`, `v` 等变量。
        u, s, v = np.linalg.svd(rot)
        # 保存或更新 `rot` 的值。
        rot = u @ v
        # 保存或更新 `since_last_svd` 的值。
        since_last_svd = 0

    # COMPUTING OMEGA UPDATE
    # Linear damping
    # 保存或更新 `omega_dot` 的值。
    omega_dot = ((1.0 / inertia) * (numba_cross(-omega, inertia * omega) + torque))

    # Quadratic damping
    # 保存或更新 `omega_damp_quadratic` 的值。
    omega_damp_quadratic = np.clip(damp_omega_quadratic * omega ** 2, 0.0, 1.0)
    # 保存或更新 `omega` 的值。
    omega = omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
    # 保存或更新 `omega` 的值。
    omega = np.clip(omega, -omega_max, omega_max)

    # Computing position
    # 保存或更新 `pos` 的值。
    pos = pos + dt * vel

    # 返回当前函数的结果。
    return thrust_rot_damp, thrust_cmds_damp, torques, torque, rot, since_last_svd, omega_dot, omega, pos, thrust, \
        # 执行这一行逻辑。
        rotor_drag_force, vel


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `floor_interaction_numba`。
def floor_interaction_numba(pos, vel, rot, omega, mu, mass, sum_thr_drag, thrust_cmds_damp, thrust_rot_damp,
                            # 这里开始一个新的代码块。
                            floor_threshold, on_floor):
    # Change pos, omega, rot, acc
    # 保存或更新 `crashed_floor` 的值。
    crashed_floor = False
    # 根据条件决定是否进入当前分支。
    if pos[2] <= floor_threshold:
        # 保存或更新 `pos` 的值。
        pos = np.array((pos[0], pos[1], floor_threshold))
        # 保存或更新 `force` 的值。
        force = rot @ sum_thr_drag
        # 根据条件决定是否进入当前分支。
        if on_floor:
            # Drone is on the floor, and on_floor flag still True
            # 保存或更新 `theta` 的值。
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            # 同时更新 `c`, `s` 等变量。
            c, s = np.cos(theta), np.sin(theta)
            # 保存或更新 `rot` 的值。
            rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # Add friction if drone is on the floor
            # 保存或更新 `friction_xy_magn` 的值。
            friction_xy_magn = mu * (mass * GRAV - force[2])

            # 根据条件决定是否进入当前分支。
            if np.linalg.norm(vel) < EPS:
                # 保存或更新 `force_xy` 的值。
                force_xy = np.array([force[0], force[1]])
                # 保存或更新 `force_xy_magn` 的值。
                force_xy_magn = np.linalg.norm(force_xy)
                # 保存或更新 `force_xy_magn` 的值。
                force_xy_magn = max(force_xy_magn - friction_xy_magn, 0.)
                # 根据条件决定是否进入当前分支。
                if force_xy_magn == 0.:
                    # 保存或更新 `force[0]` 的值。
                    force[0] = 0.
                    # 保存或更新 `force[1]` 的值。
                    force[1] = 0.
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # 保存或更新 `force_angle` 的值。
                    force_angle = np.arctan2(force[1], force[0])
                    # 保存或更新 `force_xy_dir` 的值。
                    force_xy_dir = np.array([np.cos(force_angle), np.sin(force_angle)])
                    # 保存或更新 `force_xy` 的值。
                    force_xy = force_xy_magn * force_xy_dir
                    # 保存或更新 `force[0]` 的值。
                    force[0] = force_xy[0]
                    # 保存或更新 `force[1]` 的值。
                    force[1] = force_xy[1]
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # vel > 0, friction direction is opposite to velocity direction
                # 保存或更新 `friction_xy_angle` 的值。
                friction_xy_angle = np.arctan2(vel[1], vel[0])
                # 保存或更新 `friction_xy_dir` 的值。
                friction_xy_dir = np.array([np.cos(friction_xy_angle), np.sin(friction_xy_angle)])
                # 保存或更新 `force[0]` 的值。
                force[0] = force[0] - friction_xy_dir[0] * friction_xy_magn
                # 保存或更新 `force[1]` 的值。
                force[1] = force[1] - friction_xy_dir[1] * friction_xy_magn
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # Previous step, drone still in the air, but in this step, it hits the floor
            # In previous step, self.on_floor = False, self.crashed_floor = False
            # 保存或更新 `on_floor` 的值。
            on_floor = True
            # 保存或更新 `crashed_floor` 的值。
            crashed_floor = True
            # Set vel to [0, 0, 0]
            # 同时更新 `vel`, `acc` 等变量。
            vel, acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            # 保存或更新 `omega` 的值。
            omega = np.zeros(3, dtype=np.float64)
            # Set rot
            # 保存或更新 `theta` 的值。
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            # 同时更新 `c`, `s` 等变量。
            c, s = np.cos(theta), np.sin(theta)
            # 根据条件决定是否进入当前分支。
            if rot[2, 2] < 0:
                # 保存或更新 `theta` 的值。
                theta = np.random.uniform(-np.pi, np.pi)
                # 同时更新 `c`, `s` 等变量。
                c, s = np.cos(theta), np.sin(theta)
                # 保存或更新 `rot` 的值。
                rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `rot` 的值。
                rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # reset momentum / accumulation of thrust
            # 保存或更新 `thrust_cmds_damp` 的值。
            thrust_cmds_damp = np.zeros(4)
            # 保存或更新 `thrust_rot_damp` 的值。
            thrust_rot_damp = np.zeros(4)

        # 保存或更新 `acc` 的值。
        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force
        # 保存或更新 `acc[2]` 的值。
        acc[2] = np.maximum(0, acc[2])
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # self.pos[2] > self.floor_threshold
        # 根据条件决定是否进入当前分支。
        if on_floor:
            # Drone is in the air, while on_floor flag still True
            # 保存或更新 `on_floor` 的值。
            on_floor = False

        # Computing accelerations
        # 保存或更新 `force` 的值。
        force = rot @ sum_thr_drag
        # 保存或更新 `acc` 的值。
        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force

    # 返回当前函数的结果。
    return pos, vel, acc, omega, rot, thrust_cmds_damp, thrust_rot_damp, on_floor, crashed_floor


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `compute_velocity_and_acceleration`。
def compute_velocity_and_acceleration(vel, vel_damp, dt, rot_tpose, grav_arr, acc):
    # Computing velocities
    # 保存或更新 `vel` 的值。
    vel = (1.0 - vel_damp) * vel + dt * acc

    # Accelerometer measures so-called "proper acceleration" that includes gravity with the opposite sign
    # 保存或更新 `accm` 的值。
    accm = rot_tpose @ (acc + grav_arr)
    # 返回当前函数的结果。
    return vel, accm
