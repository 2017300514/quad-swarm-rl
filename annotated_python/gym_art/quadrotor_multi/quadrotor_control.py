# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_control.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *

# 保存或更新 `GRAV` 的值。
GRAV = 9.81


# import line_profiler
# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
# 定义类 `ShiftedMotorControl`。
class ShiftedMotorControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # make it so the zero action corresponds to hovering
        # 保存或更新 `low` 的值。
        low = -1.0 * np.ones(4)
        # 保存或更新 `high` 的值。
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        # 返回当前函数的结果。
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step`。
    def step(self, dynamics, action, dt):
        # 保存或更新 `action` 的值。
        action = (action + 1.0) / dynamics.thrust_to_weight
        # 保存或更新 `action[action < 0]` 的值。
        action[action < 0] = 0
        # 保存或更新 `action[action > 1]` 的值。
        action[action > 1] = 1
        # 调用 `step` 执行当前处理。
        dynamics.step(action, dt)


# 定义类 `RawControl`。
class RawControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics, zero_action_middle=True):
        # 保存或更新 `zero_action_middle` 的值。
        self.zero_action_middle = zero_action_middle
        # print("RawControl: self.zero_action_middle", self.zero_action_middle)
        # 保存或更新 `action` 的值。
        self.action = None
        # 保存或更新 `step_func` 的值。
        self.step_func = self.step

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 根据条件决定是否进入当前分支。
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            # 保存或更新 `low` 的值。
            self.low = np.zeros(4)
            # 保存或更新 `bias` 的值。
            self.bias = 0.0
            # 保存或更新 `scale` 的值。
            self.scale = 1.0
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # Range of actions -1 .. 1
            # 保存或更新 `low` 的值。
            self.low = -np.ones(4)
            # 保存或更新 `bias` 的值。
            self.bias = 1.0
            # 保存或更新 `scale` 的值。
            self.scale = 0.5
        # 保存或更新 `high` 的值。
        self.high = np.ones(4)
        # 返回当前函数的结果。
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step`。
    def step(self, dynamics, action, goal, dt, observation=None):
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 调用 `step` 执行当前处理。
        dynamics.step(action, dt)
        # 保存或更新 `action` 的值。
        self.action = action.copy()

    # @profile
    # 定义函数 `step_tf`。
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 调用 `step` 执行当前处理。
        dynamics.step(action, dt)
        # 保存或更新 `action` 的值。
        self.action = action.copy()


# 定义类 `VerticalControl`。
class VerticalControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        # 保存或更新 `zero_action_middle` 的值。
        self.zero_action_middle = zero_action_middle

        # 保存或更新 `dim_mode` 的值。
        self.dim_mode = dim_mode
        # 根据条件决定是否进入当前分支。
        if self.dim_mode == '1D':
            # 保存或更新 `step` 的值。
            self.step = self.step1D
        # 当上一分支不满足时，继续判断新的条件。
        elif self.dim_mode == '3D':
            # 保存或更新 `step` 的值。
            self.step = self.step3D
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        # 保存或更新 `step_func` 的值。
        self.step_func = self.step

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 根据条件决定是否进入当前分支。
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            # 保存或更新 `low` 的值。
            self.low = np.zeros(1)
            # 保存或更新 `bias` 的值。
            self.bias = 0
            # 保存或更新 `scale` 的值。
            self.scale = 1.0
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # Range of actions -1 .. 1
            # 保存或更新 `low` 的值。
            self.low = -np.ones(1)
            # 保存或更新 `bias` 的值。
            self.bias = 1.0
            # 保存或更新 `scale` 的值。
            self.scale = 0.5
        # 保存或更新 `high` 的值。
        self.high = np.ones(1)
        # 返回当前函数的结果。
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step3D`。
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 调用 `step` 执行当前处理。
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step1D`。
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 调用 `step` 执行当前处理。
        dynamics.step(np.array([action[0]]), dt)


# 定义类 `VertPlaneControl`。
class VertPlaneControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        # 保存或更新 `zero_action_middle` 的值。
        self.zero_action_middle = zero_action_middle

        # 保存或更新 `dim_mode` 的值。
        self.dim_mode = dim_mode
        # 根据条件决定是否进入当前分支。
        if self.dim_mode == '2D':
            # 保存或更新 `step` 的值。
            self.step = self.step2D
        # 当上一分支不满足时，继续判断新的条件。
        elif self.dim_mode == '3D':
            # 保存或更新 `step` 的值。
            self.step = self.step3D
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        # 保存或更新 `step_func` 的值。
        self.step_func = self.step

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 根据条件决定是否进入当前分支。
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            # 保存或更新 `low` 的值。
            self.low = np.zeros(2)
            # 保存或更新 `bias` 的值。
            self.bias = 0
            # 保存或更新 `scale` 的值。
            self.scale = 1.0
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # Range of actions -1 .. 1
            # 保存或更新 `low` 的值。
            self.low = -np.ones(2)
            # 保存或更新 `bias` 的值。
            self.bias = 1.0
            # 保存或更新 `scale` 的值。
            self.scale = 0.5
        # 保存或更新 `high` 的值。
        self.high = np.ones(2)
        # 返回当前函数的结果。
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step3D`。
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 调用 `step` 执行当前处理。
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step2D`。
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        # 保存或更新 `action` 的值。
        action = self.scale * (action + self.bias)
        # 保存或更新 `action` 的值。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # 调用 `step` 执行当前处理。
        dynamics.step(np.array(action), dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
# 定义函数 `quadrotor_jacobian`。
def quadrotor_jacobian(dynamics):
    # 保存或更新 `torque` 的值。
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    # 保存或更新 `torque[2, :]` 的值。
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    # 保存或更新 `thrust` 的值。
    thrust = dynamics.thrust_max * np.ones((1, 4))
    # 保存或更新 `dw` 的值。
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    # 保存或更新 `dv` 的值。
    dv = thrust / dynamics.mass
    # 保存或更新 `J` 的值。
    J = np.vstack([dv, dw])
    # 保存或更新 `J_cond` 的值。
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    # 根据条件决定是否进入当前分支。
    if J_cond > 50:
        # 调用 `print` 执行当前处理。
        print("WARN: Jacobian conditioning is high: ", J_cond)
    # 返回当前函数的结果。
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
# 定义类 `OmegaThrustControl`。
class OmegaThrustControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics):
        # 保存或更新 `jacobian` 的值。
        jacobian = quadrotor_jacobian(dynamics)
        # 保存或更新 `Jinv` 的值。
        self.Jinv = np.linalg.inv(jacobian)

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 保存或更新 `circle_per_sec` 的值。
        circle_per_sec = 2 * np.pi
        # 保存或更新 `max_rp` 的值。
        max_rp = 5 * circle_per_sec
        # 保存或更新 `max_yaw` 的值。
        max_yaw = 1 * circle_per_sec
        # 保存或更新 `min_g` 的值。
        min_g = -1.0
        # 保存或更新 `max_g` 的值。
        max_g = dynamics.thrust_to_weight - 1.0
        # 保存或更新 `low` 的值。
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        # 保存或更新 `high` 的值。
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        # 返回当前函数的结果。
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step`。
    def step(self, dynamics, action, dt):
        # 保存或更新 `kp` 的值。
        kp = 5.0  # could be more aggressive
        # 保存或更新 `omega_err` 的值。
        omega_err = dynamics.omega - action[1:]
        # 保存或更新 `dw_des` 的值。
        dw_des = -kp * omega_err
        # 保存或更新 `acc_des` 的值。
        acc_des = GRAV * (action[0] + 1.0)
        # 保存或更新 `des` 的值。
        des = np.append(acc_des, dw_des)
        # 保存或更新 `thrusts` 的值。
        thrusts = np.matmul(self.Jinv, des)
        # 保存或更新 `thrusts[thrusts < 0]` 的值。
        thrusts[thrusts < 0] = 0
        # 保存或更新 `thrusts[thrusts > 1]` 的值。
        thrusts[thrusts > 1] = 1
        # 调用 `step` 执行当前处理。
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
# 定义类 `VelocityYawControl`。
class VelocityYawControl(object):
    # 定义函数 `__init__`。
    def __init__(self, dynamics):
        # 保存或更新 `jacobian` 的值。
        jacobian = quadrotor_jacobian(dynamics)
        # 保存或更新 `Jinv` 的值。
        self.Jinv = np.linalg.inv(jacobian)

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 保存或更新 `vmax` 的值。
        vmax = 20.0  # meters / sec
        # 保存或更新 `dymax` 的值。
        dymax = 4 * np.pi  # radians / sec
        # 保存或更新 `high` 的值。
        high = np.array([vmax, vmax, vmax, dymax])
        # 返回当前函数的结果。
        return spaces.Box(-high, high, dtype=np.float32)

    # @profile
    # 定义函数 `step`。
    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        # 保存或更新 `kp_v` 的值。
        kp_v = 5.0
        # 同时更新 `kp_a`, `kd_a` 等变量。
        kp_a, kd_a = 100.0, 50.0

        # 保存或更新 `e_v` 的值。
        e_v = dynamics.vel - action[:3]
        # 保存或更新 `acc_des` 的值。
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        # 保存或更新 `R` 的值。
        R = dynamics.rot
        # 同时更新 `zb_des`, `_` 等变量。
        zb_des, _ = normalize(acc_des)
        # 同时更新 `yb_des`, `_` 等变量。
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        # 保存或更新 `xb_des` 的值。
        xb_des = cross(yb_des, zb_des)
        # 保存或更新 `R_des` 的值。
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        # 定义函数 `vee`。
        def vee(R):
            # 返回当前函数的结果。
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        # 保存或更新 `e_R` 的值。
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        # 保存或更新 `omega_des` 的值。
        omega_des = np.array([0, 0, action[3]])
        # 保存或更新 `e_w` 的值。
        e_w = dynamics.omega - omega_des

        # 保存或更新 `dw_des` 的值。
        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # thrust_mag = np.dot(acc_des, dynamics.rot[:,2])
        # 保存或更新 `thrust_mag` 的值。
        thrust_mag = get_blas_funcs("thrust_mag", [acc_des, dynamics.rot[:, 2]])

        # 保存或更新 `des` 的值。
        des = np.append(thrust_mag, dw_des)
        # 保存或更新 `thrusts` 的值。
        thrusts = np.matmul(self.Jinv, des)
        # 保存或更新 `thrusts` 的值。
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        # 调用 `step` 执行当前处理。
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
# 定义类 `NonlinearPositionController`。
class NonlinearPositionController(object):
    # @profile
    # 定义函数 `__init__`。
    def __init__(self, dynamics, tf_control=True):
        # 导入当前模块依赖。
        import tensorflow as tf
        # 保存或更新 `jacobian` 的值。
        jacobian = quadrotor_jacobian(dynamics)
        # 保存或更新 `Jinv` 的值。
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        # 保存或更新 `action` 的值。
        self.action = None

        # 同时更新 `kp_p`, `kd_p` 等变量。
        self.kp_p, self.kd_p = 4.5, 3.5
        # 同时更新 `kp_a`, `kd_a` 等变量。
        self.kp_a, self.kd_a = 200.0, 50.0

        # 保存或更新 `rot_des` 的值。
        self.rot_des = np.eye(3)

        # 保存或更新 `tf_control` 的值。
        self.tf_control = tf_control
        # 根据条件决定是否进入当前分支。
        if tf_control:
            # 保存或更新 `step_func` 的值。
            self.step_func = self.step_tf
            # 保存或更新 `sess` 的值。
            self.sess = tf.Session()
            # 保存或更新 `thrusts_tf` 的值。
            self.thrusts_tf = self.step_graph_construct(Jinv_=self.Jinv, observation_provided=True)
            # 调用 `run` 执行当前处理。
            self.sess.run(tf.global_variables_initializer())
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `step_func` 的值。
            self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    # 定义函数 `step`。
    def step(self, dynamics, goal, dt, action=None, observation=None):
        # 保存或更新 `to_goal` 的值。
        to_goal = goal - dynamics.pos
        # goal_dist = np.sqrt(np.cumsum(np.square(to_goal)))[2]
        # 保存或更新 `goal_dist` 的值。
        goal_dist = (to_goal[0] ** 2 + to_goal[1] ** 2 + to_goal[2] ** 2) ** 0.5
        ##goal_dist = norm(to_goal)
        # 保存或更新 `e_p` 的值。
        e_p = -clamp_norm(to_goal, 4.0)
        # 保存或更新 `e_v` 的值。
        e_v = dynamics.vel
        # print('Mellinger: ', e_p, e_v, type(e_p), type(e_v))
        # 保存或更新 `acc_des` 的值。
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        # I don't need to control yaw
        # if goal_dist > 2.0 * dynamics.arm:
        #     # point towards goal
        #     xc_des = to_xyhat(to_goal)
        # else:
        #     # keep current
        #     xc_des = to_xyhat(dynamics.rot[:,0])

        # 保存或更新 `xc_des` 的值。
        xc_des = self.rot_des[:, 0]
        # xc_des = np.array([1.0, 0.0, 0.0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        # 同时更新 `zb_des`, `_` 等变量。
        zb_des, _ = normalize(acc_des)
        # 同时更新 `yb_des`, `_` 等变量。
        yb_des, _ = normalize(cross(zb_des, xc_des))
        # 保存或更新 `xb_des` 的值。
        xb_des = cross(yb_des, zb_des)
        # 保存或更新 `R_des` 的值。
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        # 保存或更新 `R` 的值。
        R = dynamics.rot

        # 定义函数 `vee`。
        def vee(R):
            # 返回当前函数的结果。
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        # 保存或更新 `e_R` 的值。
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        # 保存或更新 `e_R[2]` 的值。
        e_R[2] *= 0.2  # slow down yaw dynamics
        # 保存或更新 `e_w` 的值。
        e_w = dynamics.omega

        # 保存或更新 `dw_des` 的值。
        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # 保存或更新 `thrust_mag` 的值。
        thrust_mag = np.dot(acc_des, R[:, 2])

        # 保存或更新 `des` 的值。
        des = np.append(thrust_mag, dw_des)

        # print('Jinv:', self.Jinv)
        # 保存或更新 `thrusts` 的值。
        thrusts = np.matmul(self.Jinv, des)
        # 保存或更新 `thrusts[thrusts < 0]` 的值。
        thrusts[thrusts < 0] = 0
        # 保存或更新 `thrusts[thrusts > 1]` 的值。
        thrusts[thrusts > 1] = 1

        # 调用 `step` 执行当前处理。
        dynamics.step(thrusts, dt)
        # 保存或更新 `action` 的值。
        self.action = thrusts.copy()

    # 定义函数 `step_tf`。
    def step_tf(self, dynamics, goal, dt, action=None, observation=None):
        # print('step tf')
        # 根据条件决定是否进入当前分支。
        if not self.observation_provided:
            # 保存或更新 `xyz` 的值。
            xyz = np.expand_dims(dynamics.pos.astype(np.float32), axis=0)
            # 保存或更新 `Vxyz` 的值。
            Vxyz = np.expand_dims(dynamics.vel.astype(np.float32), axis=0)
            # 保存或更新 `Omega` 的值。
            Omega = np.expand_dims(dynamics.omega.astype(np.float32), axis=0)
            # 保存或更新 `R` 的值。
            R = np.expand_dims(dynamics.rot.astype(np.float32), axis=0)
            # print('step_tf: goal type: ', type(goal), goal[:3])
            # 保存或更新 `goal_xyz` 的值。
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)

            # 保存或更新 `result` 的值。
            result = self.sess.run([self.thrusts_tf], feed_dict={self.xyz_tf: xyz,
                                                                 self.Vxyz_tf: Vxyz,
                                                                 self.Omega_tf: Omega,
                                                                 self.R_tf: R,
                                                                 self.goal_xyz_tf: goal_xyz})

        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `print` 执行当前处理。
            print('obs fed: ', observation)
            # 保存或更新 `goal_xyz` 的值。
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)
            # 保存或更新 `result` 的值。
            result = self.sess.run([self.thrusts_tf], feed_dict={self.observation: observation,
                                                                 self.goal_xyz_tf: goal_xyz})
        # 保存或更新 `action` 的值。
        self.action = result[0].squeeze()
        # 调用 `step` 执行当前处理。
        dynamics.step(self.action, dt)

    # 定义函数 `step_graph_construct`。
    def step_graph_construct(self, Jinv_=None, observation_provided=False):
        # import tensorflow as tf
        # 保存或更新 `observation_provided` 的值。
        self.observation_provided = observation_provided
        # 使用上下文管理器包裹后续资源操作。
        with tf.variable_scope('MellingerControl'):

            # 根据条件决定是否进入当前分支。
            if not observation_provided:
                # Here we will provide all components independently
                # 保存或更新 `xyz_tf` 的值。
                self.xyz_tf = tf.placeholder(name='xyz', dtype=tf.float32, shape=(None, 3))
                # 保存或更新 `Vxyz_tf` 的值。
                self.Vxyz_tf = tf.placeholder(name='Vxyz', dtype=tf.float32, shape=(None, 3))
                # 保存或更新 `Omega_tf` 的值。
                self.Omega_tf = tf.placeholder(name='Omega', dtype=tf.float32, shape=(None, 3))
                # 保存或更新 `R_tf` 的值。
                self.R_tf = tf.placeholder(name='R', dtype=tf.float32, shape=(None, 3, 3))
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # Here we will provide observations directly and split them
                # 保存或更新 `observation` 的值。
                self.observation = tf.placeholder(name='obs', dtype=tf.float32, shape=(None, 3 + 3 + 9 + 3))
                # 同时更新 `xyz_tf`, `Vxyz_tf`, `R_flat`, `Omega_tf` 等变量。
                self.xyz_tf, self.Vxyz_tf, self.R_flat, self.Omega_tf = tf.split(self.observation, [3, 3, 9, 3], axis=1)
                # 保存或更新 `R_tf` 的值。
                self.R_tf = tf.reshape(self.R_flat, shape=[-1, 3, 3], name='R')

            # 保存或更新 `R` 的值。
            R = self.R_tf
            # R_flat = tf.placeholder(name='R_flat', type=tf.float32, shape=(None, 9))
            # R = tf.reshape(R_flat, shape=(-1, 3, 3), name='R')

            # GOAL = [x,y,z, Vx, Vy, Vz]
            # 保存或更新 `goal_xyz_tf` 的值。
            self.goal_xyz_tf = tf.placeholder(name='goal_xyz', dtype=tf.float32, shape=(None, 3))
            # goal_Vxyz = tf.placeholder(name='goal_Vxyz', type=tf.float32, shape=(None, 3))

            # Learnable gains with static initialization
            # 保存或更新 `kp_p` 的值。
            kp_p = tf.get_variable('kp_p', shape=[], initializer=tf.constant_initializer(4.5), trainable=True)  # 4.5
            # 保存或更新 `kd_p` 的值。
            kd_p = tf.get_variable('kd_p', shape=[], initializer=tf.constant_initializer(3.5), trainable=True)  # 3.5
            # 保存或更新 `kp_a` 的值。
            kp_a = tf.get_variable('kp_a', shape=[], initializer=tf.constant_initializer(200.0), trainable=True)  # 200.
            # 保存或更新 `kd_a` 的值。
            kd_a = tf.get_variable('kd_a', shape=[], initializer=tf.constant_initializer(50.0), trainable=True)  # 50.

            ## IN case you want to optimize them from random values
            # kp_p = tf.get_variable('kp_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 4.5
            # kd_p = tf.get_variable('kd_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 3.5
            # kp_a = tf.get_variable('kp_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 200.
            # kd_a = tf.get_variable('kd_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 50.

            # 保存或更新 `to_goal` 的值。
            to_goal = self.goal_xyz_tf - self.xyz_tf
            # 保存或更新 `e_p` 的值。
            e_p = -tf.clip_by_norm(to_goal, 4.0, name='e_p')
            # 保存或更新 `e_v` 的值。
            e_v = self.Vxyz_tf
            # 保存或更新 `acc_des` 的值。
            acc_des = -kp_p * e_p - kd_p * e_v + tf.constant([0, 0, 9.81], name='GRAV')
            # 调用 `print` 执行当前处理。
            print('acc_des shape: ', acc_des.get_shape().as_list())

            # 定义函数 `project_xy`。
            def project_xy(x, name='project_xy'):
                # print('x_shape:', x.get_shape().as_list())
                # x = tf.squeeze(x, axis=2)
                # 返回当前函数的结果。
                return tf.multiply(x, tf.constant([1., 1., 0.]), name=name)

            # goal_dist = tf.norm(to_goal, name='goal_xyz_dist')
            # 保存或更新 `xc_des` 的值。
            xc_des = project_xy(tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2), name='xc_des')
            # 调用 `print` 执行当前处理。
            print('xc_des shape: ', xc_des.get_shape().as_list())
            # xc_des = project_xy(R[:, 0])

            # rotation towards the ideal thrust direction
            # see Mellinger and Kumar 2011
            # 保存或更新 `zb_des` 的值。
            zb_des = tf.nn.l2_normalize(acc_des, axis=1, name='zb_dex')
            # 保存或更新 `yb_des` 的值。
            yb_des = tf.nn.l2_normalize(tf.cross(zb_des, xc_des), axis=1, name='yb_des')
            # 保存或更新 `xb_des` 的值。
            xb_des = tf.cross(yb_des, zb_des, name='xb_des')
            # 保存或更新 `R_des` 的值。
            R_des = tf.stack([xb_des, yb_des, zb_des], axis=2, name='R_des')

            # 调用 `print` 执行当前处理。
            print('zb_des shape: ', zb_des.get_shape().as_list())
            # 调用 `print` 执行当前处理。
            print('yb_des shape: ', yb_des.get_shape().as_list())
            # 调用 `print` 执行当前处理。
            print('xb_des shape: ', xb_des.get_shape().as_list())
            # 调用 `print` 执行当前处理。
            print('R_des shape: ', R_des.get_shape().as_list())

            # 定义函数 `transpose`。
            def transpose(x):
                # 返回当前函数的结果。
                return tf.transpose(x, perm=[0, 2, 1])

            # Rotational difference
            # 保存或更新 `Rdiff` 的值。
            Rdiff = tf.matmul(transpose(R_des), R) - tf.matmul(transpose(R), R_des, name='Rdiff')
            # 调用 `print` 执行当前处理。
            print('Rdiff shape: ', Rdiff.get_shape().as_list())

            # 定义函数 `tf_vee`。
            def tf_vee(R, name='vee'):
                # 返回当前函数的结果。
                return tf.squeeze(tf.stack([
                    tf.squeeze(tf.slice(R, [0, 2, 1], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 0, 2], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 1, 0], [-1, 1, 1]), axis=2)], axis=1, name=name), axis=2)

            # def vee(R):
            #     return np.array([R[2, 1], R[0, 2], R[1, 0]])

            # 保存或更新 `e_R` 的值。
            e_R = 0.5 * tf_vee(Rdiff, name='e_R')
            # 调用 `print` 执行当前处理。
            print('e_R shape: ', e_R.get_shape().as_list())
            # e_R[2] *= 0.2  # slow down yaw dynamics
            # 保存或更新 `e_w` 的值。
            e_w = self.Omega_tf

            # Control orientation
            # 保存或更新 `dw_des` 的值。
            dw_des = -kp_a * e_R - kd_a * e_w
            # 调用 `print` 执行当前处理。
            print('dw_des shape: ', dw_des.get_shape().as_list())

            # we want this acceleration, but we can only accelerate in one direction!
            # thrust_mag = np.dot(acc_des, R[:, 2])
            # 保存或更新 `acc_cur` 的值。
            acc_cur = tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2)
            # 调用 `print` 执行当前处理。
            print('acc_cur shape: ', acc_cur.get_shape().as_list())

            # 保存或更新 `acc_dot` 的值。
            acc_dot = tf.multiply(acc_des, acc_cur)
            # 调用 `print` 执行当前处理。
            print('acc_dot shape: ', acc_dot.get_shape().as_list())

            # 保存或更新 `thrust_mag` 的值。
            thrust_mag = tf.reduce_sum(acc_dot, axis=1, keepdims=True, name='thrust_mag')
            # 调用 `print` 执行当前处理。
            print('thrust_mag shape: ', thrust_mag.get_shape().as_list())

            # des = np.append(thrust_mag, dw_des)
            # 保存或更新 `des` 的值。
            des = tf.concat([thrust_mag, dw_des], axis=1, name='des')
            # 调用 `print` 执行当前处理。
            print('des shape: ', des.get_shape().as_list())

            # 根据条件决定是否进入当前分支。
            if Jinv_ is None:
                # Learn the jacobian inverse
                # 保存或更新 `Jinv` 的值。
                Jinv = tf.get_variable('Jinv', initializer=tf.random_normal(shape=[4, 4], mean=0.0, stddev=0.1),
                                       trainable=True)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # Jacobian inverse is provided
                # 保存或更新 `Jinv` 的值。
                Jinv = tf.constant(Jinv_.astype(np.float32), name='Jinv')
                # Jinv = tf.get_variable('Jinv', shape=[4,4], initializer=tf.constant_initializer())

            # 调用 `print` 执行当前处理。
            print('Jinv shape: ', Jinv.get_shape().as_list())
            ## Jacobian inverse for our quadrotor
            # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
            #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
            #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
            #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])

            # thrusts = np.matmul(self.Jinv, des)
            # 保存或更新 `thrusts` 的值。
            thrusts = tf.matmul(des, tf.transpose(Jinv), name='thrust')
            # 保存或更新 `thrusts` 的值。
            thrusts = tf.clip_by_value(thrusts, clip_value_min=0.0, clip_value_max=1.0, name='thrust_clipped')
            # 返回当前函数的结果。
            return thrusts

    # 定义函数 `action_space`。
    def action_space(self, dynamics):
        # 保存或更新 `circle_per_sec` 的值。
        circle_per_sec = 2 * np.pi
        # 保存或更新 `max_rp` 的值。
        max_rp = 5 * circle_per_sec
        # 保存或更新 `max_yaw` 的值。
        max_yaw = 1 * circle_per_sec
        # 保存或更新 `min_g` 的值。
        min_g = -1.0
        # 保存或更新 `max_g` 的值。
        max_g = dynamics.thrust_to_weight - 1.0
        # 保存或更新 `low` 的值。
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        # 保存或更新 `high` 的值。
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        # 返回当前函数的结果。
        return spaces.Box(low, high, dtype=np.float32)

# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition
