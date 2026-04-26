#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_control.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件位于“策略/参考目标”和“底层动力学”之间，负责把不同形式的动作解释成电机推力命令。
# 上游输入可能是强化学习策略输出、目标速度/目标位置，或用于调试的解析控制目标；
# 下游统一流向 `QuadrotorDynamics.step()`，因此这里决定了策略到底在学什么控制接口。
# 结合 `quadrotor_single.py` 一起看时，可以把这里理解为动作语义层，而 `quadrotor_dynamics.py` 是物理执行层。

from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *

GRAV = 9.81


# import line_profiler
# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
class ShiftedMotorControl(object):
    # 这种控制接口仍然是四电机直接控制，
    # 但把“悬停所需推力”平移到动作 0 附近，便于某些控制器或手工策略在零附近微调。
    def __init__(self, dynamics):
        pass

    def action_space(self, dynamics):
        # 上界不是固定 1，而是 `thrust_to_weight - 1`，
        # 因为零点已经被挪到悬停位置，正方向只剩下“相对悬停还能再加多少推力”。
        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        # 这里把“以悬停为中心的动作”重新映射回动力学层期望的 `[0, 1]` 四电机命令。
        action = (action + 1.0) / dynamics.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        dynamics.step(action, dt)


class RawControl(object):
    # 这是主训练链路最常用的控制接口。
    # 策略直接输出四路归一化动作，环境几乎不做解析控制，只做线性缩放后交给动力学层。
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        # print("RawControl: self.zero_action_middle", self.zero_action_middle)
        self.action = None
        self.step_func = self.step

    def action_space(self, dynamics):
        # `zero_action_middle=True` 时，策略输出空间是更常见的 `[-1, 1]`，
        # 但动力学层最终仍消费 `[0, 1]` 推力命令，所以这里要保存一套 bias/scale 供 step 重用。
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, goal, dt, observation=None):
        # `goal` 在 raw control 下其实不参与控制律，
        # 这也说明策略必须自己从观测里学会“怎样朝目标飞”，而不是靠解析控制器代劳。
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


class VerticalControl(object):
    # 这个接口把动作压成 1 维，只保留整体升力控制。
    # 它主要用于极简稳定性实验，而不是论文里的完整三维多机训练。
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # 3D 模式下把单个升力命令复制给四个电机，等价于只控制总推力、不控制姿态差动。
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)


class VertPlaneControl(object):
    # 这个接口保留两个自由度：一个控制左右成对电机的共同升力，另一个控制前后成对电机。
    # 它适合二维平面飞行实验，介于纯竖直控制和完整四电机控制之间。
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    # 这里构造的是“归一化电机推力 -> 总加速度/角加速度”的局部线性映射。
    # 后面的解析控制器会先算出想要的加速度和角加速度，再通过这个雅可比逆求每个电机该给多少推力。
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    # 这个控制器要求策略直接给“总升力偏置 + 目标角速度”。
    # 它比 raw control 更结构化，但仍然比位置控制器更贴近底层执行层。
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        # 先按 P 控制把目标角速度转成期望角加速度，
        # 再与期望总升力拼成 4 维目标，通过 `Jinv` 求回四电机推力。
        kp = 5.0  # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    # 这个控制器接收“目标线速度 + 目标偏航角速度”，
    # 再用解析姿态控制把它折算成电机推力，属于比 raw control 更高层的接口。
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0  # meters / sec
        dymax = 4 * np.pi  # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high, dtype=np.float32)

    # @profile
    def step(self, dynamics, action, dt):
        # 这里先把速度误差变成期望平动加速度，再构造与该加速度一致的期望姿态。
        # 最后仍然回到“期望总推力 + 期望角加速度 -> 四电机推力”的同一求解框架。
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # thrust_mag = np.dot(acc_des, dynamics.rot[:,2])
        thrust_mag = get_blas_funcs("thrust_mag", [acc_des, dynamics.rot[:, 2]])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class NonlinearPositionController(object):
    # 这是基于 Mellinger 控制思想的解析位置控制器。
    # 在训练代码里它更像调试/教师控制器或旧实验接口，不是当前论文主链路里的学习动作接口。
    # @profile
    def __init__(self, dynamics, tf_control=True):
        import tensorflow as tf
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        # 这四个增益分别对应位置 PD 和姿态 PD。
        # 它们决定解析控制器把位置误差、多快的速度误差和姿态误差转成多激进的推力修正。
        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.tf_control = tf_control
        if tf_control:
            # TensorFlow 路径把同一套控制律改写成图执行形式，
            # 主要为了旧版部署/实验接口，而不是训练主线的默认依赖。
            self.step_func = self.step_tf
            self.sess = tf.Session()
            self.thrusts_tf = self.step_graph_construct(Jinv_=self.Jinv, observation_provided=True)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, goal, dt, action=None, observation=None):
        # 位置控制的主线是：
        # 目标位置误差 -> 期望平动加速度 -> 期望姿态 -> 期望角加速度 -> 四电机推力。
        to_goal = goal - dynamics.pos
        # goal_dist = np.sqrt(np.cumsum(np.square(to_goal)))[2]
        goal_dist = (to_goal[0] ** 2 + to_goal[1] ** 2 + to_goal[2] ** 2) ** 0.5
        ##goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        # print('Mellinger: ', e_p, e_v, type(e_p), type(e_v))
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        # 这里默认不主动追踪目标偏航，只保持一个固定的机体 x 轴参考方向。
        # 对本项目来说，主任务是导航与避碰，偏航通常不是主要优化目标。
        # if goal_dist > 2.0 * dynamics.arm:
        #     # point towards goal
        #     xc_des = to_xyhat(to_goal)
        # else:
        #     # keep current
        #     xc_des = to_xyhat(dynamics.rot[:,0])

        xc_des = self.rot_des[:, 0]
        # xc_des = np.array([1.0, 0.0, 0.0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2  # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        # print('Jinv:', self.Jinv)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()

    def step_tf(self, dynamics, goal, dt, action=None, observation=None):
        # 图执行版本与 `step()` 控制律一致，只是把输入组织成 TensorFlow placeholder 供会话运行。
        # print('step tf')
        if not self.observation_provided:
            xyz = np.expand_dims(dynamics.pos.astype(np.float32), axis=0)
            Vxyz = np.expand_dims(dynamics.vel.astype(np.float32), axis=0)
            Omega = np.expand_dims(dynamics.omega.astype(np.float32), axis=0)
            R = np.expand_dims(dynamics.rot.astype(np.float32), axis=0)
            # print('step_tf: goal type: ', type(goal), goal[:3])
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)

            result = self.sess.run([self.thrusts_tf], feed_dict={self.xyz_tf: xyz,
                                                                 self.Vxyz_tf: Vxyz,
                                                                 self.Omega_tf: Omega,
                                                                 self.R_tf: R,
                                                                 self.goal_xyz_tf: goal_xyz})

        else:
            print('obs fed: ', observation)
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)
            result = self.sess.run([self.thrusts_tf], feed_dict={self.observation: observation,
                                                                 self.goal_xyz_tf: goal_xyz})
        self.action = result[0].squeeze()
        dynamics.step(self.action, dt)

    def step_graph_construct(self, Jinv_=None, observation_provided=False):
        # import tensorflow as tf
        # 这个函数把 Mellinger 控制律完整展开成计算图。
        # 当 sim2real 或旧实验流程希望在图里直接复用控制器时，会从这里取 thrust 输出。
        self.observation_provided = observation_provided
        with tf.variable_scope('MellingerControl'):

            if not observation_provided:
                # 这一分支要求外部分别喂入位置、速度、姿态和角速度。
                self.xyz_tf = tf.placeholder(name='xyz', dtype=tf.float32, shape=(None, 3))
                self.Vxyz_tf = tf.placeholder(name='Vxyz', dtype=tf.float32, shape=(None, 3))
                self.Omega_tf = tf.placeholder(name='Omega', dtype=tf.float32, shape=(None, 3))
                self.R_tf = tf.placeholder(name='R', dtype=tf.float32, shape=(None, 3, 3))
            else:
                # 这一分支直接消费单机观测向量，再按 `[xyz, vxyz, R, omega]` 拆回控制器所需状态。
                self.observation = tf.placeholder(name='obs', dtype=tf.float32, shape=(None, 3 + 3 + 9 + 3))
                self.xyz_tf, self.Vxyz_tf, self.R_flat, self.Omega_tf = tf.split(self.observation, [3, 3, 9, 3], axis=1)
                self.R_tf = tf.reshape(self.R_flat, shape=[-1, 3, 3], name='R')

            R = self.R_tf
            # R_flat = tf.placeholder(name='R_flat', type=tf.float32, shape=(None, 9))
            # R = tf.reshape(R_flat, shape=(-1, 3, 3), name='R')

            # GOAL = [x,y,z, Vx, Vy, Vz]
            self.goal_xyz_tf = tf.placeholder(name='goal_xyz', dtype=tf.float32, shape=(None, 3))
            # goal_Vxyz = tf.placeholder(name='goal_Vxyz', type=tf.float32, shape=(None, 3))

            # 这些增益虽然写成 TensorFlow variable，
            # 但默认初始化就是与 Python 路径相同的手工控制参数。
            kp_p = tf.get_variable('kp_p', shape=[], initializer=tf.constant_initializer(4.5), trainable=True)  # 4.5
            kd_p = tf.get_variable('kd_p', shape=[], initializer=tf.constant_initializer(3.5), trainable=True)  # 3.5
            kp_a = tf.get_variable('kp_a', shape=[], initializer=tf.constant_initializer(200.0), trainable=True)  # 200.
            kd_a = tf.get_variable('kd_a', shape=[], initializer=tf.constant_initializer(50.0), trainable=True)  # 50.

            ## IN case you want to optimize them from random values
            # kp_p = tf.get_variable('kp_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 4.5
            # kd_p = tf.get_variable('kd_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 3.5
            # kp_a = tf.get_variable('kp_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 200.
            # kd_a = tf.get_variable('kd_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 50.

            to_goal = self.goal_xyz_tf - self.xyz_tf
            e_p = -tf.clip_by_norm(to_goal, 4.0, name='e_p')
            e_v = self.Vxyz_tf
            acc_des = -kp_p * e_p - kd_p * e_v + tf.constant([0, 0, 9.81], name='GRAV')
            print('acc_des shape: ', acc_des.get_shape().as_list())

            def project_xy(x, name='project_xy'):
                # print('x_shape:', x.get_shape().as_list())
                # x = tf.squeeze(x, axis=2)
                return tf.multiply(x, tf.constant([1., 1., 0.]), name=name)

            # goal_dist = tf.norm(to_goal, name='goal_xyz_dist')
            xc_des = project_xy(tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2), name='xc_des')
            print('xc_des shape: ', xc_des.get_shape().as_list())
            # xc_des = project_xy(R[:, 0])

            # 这一段与 Python 版同构：先由期望加速度确定机体 z 轴，再补齐期望姿态矩阵。
            # see Mellinger and Kumar 2011
            zb_des = tf.nn.l2_normalize(acc_des, axis=1, name='zb_dex')
            yb_des = tf.nn.l2_normalize(tf.cross(zb_des, xc_des), axis=1, name='yb_des')
            xb_des = tf.cross(yb_des, zb_des, name='xb_des')
            R_des = tf.stack([xb_des, yb_des, zb_des], axis=2, name='R_des')

            print('zb_des shape: ', zb_des.get_shape().as_list())
            print('yb_des shape: ', yb_des.get_shape().as_list())
            print('xb_des shape: ', xb_des.get_shape().as_list())
            print('R_des shape: ', R_des.get_shape().as_list())

            def transpose(x):
                return tf.transpose(x, perm=[0, 2, 1])

            # `Rdiff -> e_R` 把姿态矩阵差异压缩成李代数形式的姿态误差向量，
            # 后面再配合角速度误差一起生成期望角加速度。
            Rdiff = tf.matmul(transpose(R_des), R) - tf.matmul(transpose(R), R_des, name='Rdiff')
            print('Rdiff shape: ', Rdiff.get_shape().as_list())

            def tf_vee(R, name='vee'):
                return tf.squeeze(tf.stack([
                    tf.squeeze(tf.slice(R, [0, 2, 1], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 0, 2], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 1, 0], [-1, 1, 1]), axis=2)], axis=1, name=name), axis=2)

            # def vee(R):
            #     return np.array([R[2, 1], R[0, 2], R[1, 0]])

            e_R = 0.5 * tf_vee(Rdiff, name='e_R')
            print('e_R shape: ', e_R.get_shape().as_list())
            # e_R[2] *= 0.2  # slow down yaw dynamics
            e_w = self.Omega_tf

            # 至此又回到统一接口：总推力 + 角加速度需求，通过 `Jinv` 反解四电机推力。
            dw_des = -kp_a * e_R - kd_a * e_w
            print('dw_des shape: ', dw_des.get_shape().as_list())

            # we want this acceleration, but we can only accelerate in one direction!
            # thrust_mag = np.dot(acc_des, R[:, 2])
            acc_cur = tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2)
            print('acc_cur shape: ', acc_cur.get_shape().as_list())

            acc_dot = tf.multiply(acc_des, acc_cur)
            print('acc_dot shape: ', acc_dot.get_shape().as_list())

            thrust_mag = tf.reduce_sum(acc_dot, axis=1, keepdims=True, name='thrust_mag')
            print('thrust_mag shape: ', thrust_mag.get_shape().as_list())

            # des = np.append(thrust_mag, dw_des)
            des = tf.concat([thrust_mag, dw_des], axis=1, name='des')
            print('des shape: ', des.get_shape().as_list())

            if Jinv_ is None:
                # Learn the jacobian inverse
                Jinv = tf.get_variable('Jinv', initializer=tf.random_normal(shape=[4, 4], mean=0.0, stddev=0.1),
                                       trainable=True)
            else:
                # Jacobian inverse is provided
                Jinv = tf.constant(Jinv_.astype(np.float32), name='Jinv')
                # Jinv = tf.get_variable('Jinv', shape=[4,4], initializer=tf.constant_initializer())

            print('Jinv shape: ', Jinv.get_shape().as_list())
            ## Jacobian inverse for our quadrotor
            # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
            #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
            #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
            #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])

            # thrusts = np.matmul(self.Jinv, des)
            thrusts = tf.matmul(des, tf.transpose(Jinv), name='thrust')
            thrusts = tf.clip_by_value(thrusts, clip_value_min=0.0, clip_value_max=1.0, name='thrust_clipped')
            return thrusts

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition
