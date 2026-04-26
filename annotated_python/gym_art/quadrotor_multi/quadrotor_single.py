#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_single.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件定义单架四旋翼的底层物理环境，是多机环境中每个 agent 的最小执行单元。
# 上游输入包括动力学参数、控制方式、房间尺寸、观测表示和邻居/障碍开关；
# 下游输出是单机级别的状态向量、基础奖励分解和 `QuadrotorDynamics` 状态更新。
# 多机环境 `QuadrotorEnvMulti` 会从这里收集每架无人机的原始状态，再继续拼接邻居观测、障碍观测和碰撞奖励。
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 

References:
[1] RotorS: https://www.researchgate.net/profile/Fadri_Furrer/publication/309291237_RotorS_-_A_Modular_Gazebo_MAV_Simulator_Framework/links/5a0169c4a6fdcc82a3183f8f/RotorS-A-Modular-Gazebo-MAV-Simulator-Framework.pdf
[2] CrazyFlie modelling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
[3] HummingBird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
[4] CrazyFlie thrusters transition functions: https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
[5] HummingBird modelling: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds
[6] Rotation b/w matrices: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
[7] Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
import copy

from gymnasium.utils import seeding

import gym_art.quadrotor_multi.get_state as get_state
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics
from gym_art.quadrotor_multi.sensor_noise import SensorNoise

GRAV = 9.81  # default gravitational constant


def compute_reward_weighted(dynamics, goal, action, dt, time_remain, rew_coeff, action_prev, on_floor=False):
    # 这里计算的是单机层面的基础奖励，不包含多机邻居碰撞、障碍碰撞或房间碰撞惩罚。
    # 换句话说，这个函数只回答“这架无人机自身有没有朝目标飞、控制是否平稳、姿态是否失控、是否摔地”。
    # 多机环境会在拿到这里的 `reward` 和 `rew_info` 后，再额外叠加邻居/障碍相关惩罚。

    # 到目标点的欧氏距离是主任务项。
    # 距离越大，位置代价越大，因此最终 reward 会更负。
    dist = np.linalg.norm(goal - dynamics.pos)
    cost_pos_raw = dist
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    # 控制量范数作为 effort 代价，鼓励策略不要长期输出极端电机命令。
    cost_effort_raw = np.linalg.norm(action)
    cost_effort = rew_coeff["effort"] * cost_effort_raw

    # `dynamics.rot[2, 2]` 近似描述机体 z 轴与世界 z 轴的对齐程度。
    # 正常飞行时希望机体“站正”，因此这里把姿态偏离转成惩罚。
    # 若已在地面上，则直接把姿态代价置成 1，避免地面姿态异常时再依赖旋转矩阵细节。
    if on_floor:
        cost_orient_raw = 1.0
    else:
        cost_orient_raw = -dynamics.rot[2, 2]

    cost_orient = rew_coeff["orient"] * cost_orient_raw

    # 角速度范数衡量自旋是否过大。
    # 这个项抑制无人机在目标点附近疯狂打转，帮助策略学到更稳定的姿态控制。
    cost_spin_raw = (dynamics.omega[0] ** 2 + dynamics.omega[1] ** 2 + dynamics.omega[2] ** 2) ** 0.5
    cost_spin = rew_coeff["spin"] * cost_spin_raw

    # 是否着地在这里被视为一种 crash 信号。
    # 这个定义主要服务于“能否维持飞行”而非复杂碰撞分析，后者在多机环境里继续细分。
    cost_crash_raw = float(on_floor)
    cost_crash = rew_coeff["crash"] * cost_crash_raw

    reward = -dt * np.sum([
        cost_pos,
        cost_effort,
        cost_crash,
        cost_orient,
        cost_spin,
    ])

    # `rew_*` 是已经乘过权重的训练用分项；
    # `rewraw_*` 保留未加权的原始物理量，方便上层 wrapper 在日志里分清“是系数调大了，还是物理状态真的变差了”。
    rew_info = {
        "rew_main": -cost_pos,
        'rew_pos': -cost_pos,
        'rew_action': -cost_effort,
        'rew_crash': -cost_crash,
        "rew_orient": -cost_orient,
        "rew_spin": -cost_spin,

        "rewraw_main": -cost_pos_raw,
        'rewraw_pos': -cost_pos_raw,
        'rewraw_action': -cost_effort_raw,
        'rewraw_crash': -cost_crash_raw,
        "rewraw_orient": -cost_orient_raw,
        "rewraw_spin": -cost_spin_raw,
    }

    # 所有奖励项统一乘以 `dt`，把连续时间代价离散化成“每一个控制步贡献多少”。
    for k, v in rew_info.items():
        rew_info[k] = dt * v

    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


class QuadrotorSingle:
    # 这个类维护的是一架无人机在单个 episode 里的全部底层状态：
    # 动力学、控制器、房间边界、目标点、初始采样、观测空间和基础奖励计算都在这里定义。
    # 它不关心其他无人机在哪里，也不直接处理机间碰撞；这些都留给 `QuadrotorEnvMulti` 统一处理。
    def __init__(self, dynamics_params="DefaultQuad", dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr="xyz_vxyz_R_omega", ep_time=7, room_dims=(10.0, 10.0, 10.0),
                 init_random_state=False, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False, use_numba=False,
                 neighbor_obs_type='none', num_agents=1, num_use_neighbor_obs=0, use_obstacles=False):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            
            dynamics_randomize_every: [int] how often (trajectories) perform randomization dynamics_sampler_1: [dict] 
            the first sampler to be applied. Dict must contain type (see quadrotor_randomization) and whatever params 
            requires 
            dynamics_sampler_2: [dict] the second sampler to be applied. Convenient if you need to 
                fix some params after sampling.
            
            raw_control: [bool] use raw control or the Mellinger controller as a default
            raw_control_zero_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. 
            Options: 1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            ep_time: [float] episode time in simulated seconds. 
                This parameter is used to compute env max time length in steps.
            room_size: [int] env room size. Not the same as the initialization box to allow shorter episodes
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. If "default" then the default params 
                are loaded. Otherwise one can provide specific params.
            excite: [bool] change the set point at the fixed frequency to perturb the quad
        """
        self.use_numba = use_numba

        # 房间尺寸决定可飞行边界，也决定后续观测空间里位置/高度分量的上下界。
        self.room_length = room_dims[0]
        self.room_width = room_dims[1]
        self.room_height = room_dims[2]
        self.room_box = np.array([[-self.room_length / 2., -self.room_width / 2, 0.],
                                  [self.room_length / 2., self.room_width / 2., self.room_height]])

        self.init_random_state = init_random_state

        self.obs_repr = obs_repr
        self.rew_coeff = None

        # `ep_len` 是把用户给出的“秒级 episode 时长”换成离散控制步数后的结果。
        # 后面的 `tick > ep_len` 终止判断依赖这组时间尺度换算。
        self.ep_time = ep_time
        self.sim_steps = sim_steps
        self.dt = 1.0 / sim_freq
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        self.control_freq = sim_freq / sim_steps
        self.traj_count = 0

        # 这些字段控制单机动力学和控制器的具体形式。
        self.dim_mode = dim_mode
        self.raw_control_zero_middle = raw_control_zero_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.verbose = verbose
        self.raw_control = raw_control
        self.gravity = gravity
        self.update_sense_noise(sense_noise=sense_noise)
        self.t2w_std = t2w_std
        self.t2w_min = 1.5
        self.t2w_max = 10.0

        self.t2t_std = t2t_std
        self.t2t_min = 0.005
        self.t2t_max = 1.0
        self.excite = excite
        self.dynamics_simplification = dynamics_simplification
        self.max_init_vel = 1.
        self.max_init_omega = 2 * np.pi

        # 这里先根据 `dynamics_params` 选出一个基础动力学参数采样器，再叠加人工修正和额外随机化采样器。
        # 最终得到的 `self.dynamics_params` 会继续流向 `QuadrotorDynamics`，决定质量、推力、惯量等底层物理属性。
        self.dyn_base_sampler = getattr(quad_rand, dynamics_params)()
        self.dynamics_change = copy.deepcopy(dynamics_change)
        self.dynamics_params = self.dyn_base_sampler.sample()
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        self.dyn_sampler_1 = dyn_sampler_1
        if dyn_sampler_1 is not None:
            sampler_type = dyn_sampler_1["class"]
            self.dyn_sampler_1_params = copy.deepcopy(dyn_sampler_1)
            del self.dyn_sampler_1_params["class"]
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_1_params)

        self.dyn_sampler_2 = dyn_sampler_2
        if dyn_sampler_2 is not None:
            sampler_type = dyn_sampler_2["class"]
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            del self.dyn_sampler_2_params["class"]
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_2_params)

        # `resample_dynamics()` 不只是在重采样参数，它还会顺带重建 `QuadrotorDynamics` 和控制器。
        self.action_space = None
        self.resample_dynamics()

        # `state_vector` 是一个函数指针，按 `obs_repr` 决定如何把内部物理状态编码成观测向量。
        # 这让同一个单机环境可以在不同实验里输出不同观测布局。
        self.state_vector = self.state_vector = getattr(get_state, "state_" + self.obs_repr)
        if use_obstacles:
            # 障碍环境里初始采样盒更小，是为了让无人机和障碍物交互更频繁，避免出生位置过稀。
            self.box = 0.1
        else:
            self.box = 2.0
        self.box_scale = 1.0
        self.goal = None
        self.spawn_point = None

        # 这些字段本身不在单机 step 中使用邻居信息，但它们会影响观测空间声明，
        # 使单机对象知道自己未来可能被多机环境要求预留多少邻居观测维度。
        self.num_agents = num_agents
        self.neighbor_obs_type = neighbor_obs_type
        self.num_use_neighbor_obs = num_use_neighbor_obs

        self.use_obstacles = use_obstacles

        self.observation_space = self.make_observation_space()

        self._seed()

    def update_sense_noise(self, sense_noise):
        # 传感噪声在这里被统一转成 `SensorNoise` 对象。
        # 后续真正取状态时，`state_*` 函数会通过这个对象决定是否给位置、姿态等观测加噪声。
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False, use_numba=self.use_numba)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def update_dynamics(self, dynamics_params):
        # 这里把采样好的参数真正实例化成底层动力学系统。
        # 从这一步开始，质量、推力、速度上限、角速度上限等物理属性都进入可执行状态。
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params,
                                          dynamics_steps_num=self.sim_steps, room_box=self.room_box,
                                          dim_mode=self.dim_mode, gravity=self.gravity,
                                          dynamics_simplification=self.dynamics_simplification,
                                          use_numba=self.use_numba, dt=self.dt)

        # 控制器类型决定策略输出如何解释。
        # `raw_control=True` 时，策略更直接地驱动飞行器；
        # 否则策略输出会先进入非线性位置控制器，再由控制器生成底层控制量。
        if self.raw_control:
            if self.dim_mode == '1D':
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '2D':
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=self.tf_control)

        self.action_space = self.controller.action_space(self.dynamics)
        self.state_vector = getattr(get_state, "state_" + self.obs_repr)

    def make_observation_space(self):
        # 这里不是在生成当前观测值，而是在声明“当前环境可能输出什么范围的观测”。
        # 这份边界会被策略网络输入层、neighbor 裁剪和环境校验共同使用。
        room_range = self.room_box[1] - self.room_box[0]
        self.obs_space_low_high = {
            "xyz": [-room_range, room_range],
            "xyzr": [-room_range, room_range],
            "vxyz": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "vxyzr": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "acc": [-self.dynamics.acc_max * np.ones(3), self.dynamics.acc_max * np.ones(3)],
            "R": [-np.ones(9), np.ones(9)],
            "omega": [-self.dynamics.omega_max * np.ones(3), self.dynamics.omega_max * np.ones(3)],
            "t2w": [0. * np.ones(1), 5. * np.ones(1)],
            "t2t": [0. * np.ones(1), 1. * np.ones(1)],
            "h": [0. * np.ones(1), self.room_box[1][2] * np.ones(1)],
            "act": [np.zeros(4), np.ones(4)],
            "quat": [-np.ones(4), np.ones(4)],
            "euler": [-np.pi * np.ones(3), np.pi * np.ones(3)],
            "rxyz": [-room_range, room_range],
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)],
            "roxyz": [-room_range, room_range],
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)],
            "osize": [np.zeros(3), 20.0 * np.ones(3)],
            "otype": [np.zeros(1), 20.0 * np.ones(1)],
            "goal": [-room_range, room_range],
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
            "floor": [np.zeros(1), self.room_box[1][2] * np.ones(1)],
            "octmap": [-10 * np.ones(9), 10 * np.ones(9)],
        }
        self.obs_comp_names = list(self.obs_space_low_high.keys())
        self.obs_comp_sizes = [self.obs_space_low_high[name][1].size for name in self.obs_comp_names]

        # 先按 `obs_repr` 决定自观测内容，再按实验配置追加邻居和障碍部分。
        # 这样单机环境本身就知道多机版本最终会输出多长的观测向量。
        obs_comps = self.obs_repr.split("_")
        if self.neighbor_obs_type == 'pos_vel' and self.num_use_neighbor_obs > 0:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs

        if self.use_obstacles:
            obs_comps = obs_comps + ["octmap"]

        print("Observation components:", obs_comps)
        obs_low, obs_high = [], []
        for comp in obs_comps:
            obs_low.append(self.obs_space_low_high[comp][0])
            obs_high.append(self.obs_space_low_high[comp][1])
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)

        # 这几张索引表记录每个观测分量在完整向量中的位置，后续做切片或调试时会用到。
        self.obs_comp_sizes_dict, self.obs_space_comp_indx, self.obs_comp_end = {}, {}, []
        end_indx = 0
        for obs_i, obs_name in enumerate(self.obs_comp_names):
            end_indx += self.obs_comp_sizes[obs_i]
            self.obs_comp_sizes_dict[obs_name] = self.obs_comp_sizes[obs_i]
            self.obs_space_comp_indx[obs_name] = obs_i
            self.obs_comp_end.append(end_indx)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return self.observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # 保留前一时刻动作，便于后续如果需要把 action change 也纳入奖励或日志。
        self.actions[1] = copy.deepcopy(self.actions[0])
        self.actions[0] = copy.deepcopy(action)

        # 这里才是真正把策略输出送入控制器和动力学系统的地方。
        self.controller.step_func(dynamics=self.dynamics, action=action, goal=self.goal, dt=self.dt, observation=None)

        self.time_remain = self.ep_len - self.tick
        reward, rew_info = compute_reward_weighted(
            dynamics=self.dynamics, goal=self.goal, action=action, dt=self.dt, time_remain=self.time_remain,
            rew_coeff=self.rew_coeff, action_prev=self.actions[1], on_floor=self.dynamics.on_floor)

        self.tick += 1
        done = self.tick > self.ep_len

        # `state_vector(self)` 会读取当前动力学状态、目标、传感噪声配置等信息，
        # 按 `obs_repr` 指定的布局把它们拼成策略真正看到的观测。
        sv = self.state_vector(self)
        self.traj_count += int(done)

        return sv, reward, done, {'rewards': rew_info}

    def resample_dynamics(self):
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        # 这里按“基础采样 -> 人工修正 -> sampler1 -> sampler2”的顺序生成本次 episode 的动力学参数。
        self.dynamics_params = self.dyn_base_sampler.sample()

        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        if self.dyn_sampler_1 is not None:
            self.dynamics_params = self.dyn_sampler_1.sample(self.dynamics_params)

        if self.dyn_sampler_2 is not None:
            self.dynamics_params = self.dyn_sampler_2.sample(self.dynamics_params)

        # 在真正构建动力学对象前先校验物理参数是否仍然合理，避免无意义的随机化组合直接把仿真搞坏。
        quad_rand.check_quad_param_limits(self.dynamics_params)

        self.update_dynamics(dynamics_params=self.dynamics_params)

    def _reset(self):
        # 如果配置要求“每隔若干条轨迹就随机一次动力学”，这里会在 episode 开头重采样。
        if self.dynamics_randomize_every is not None and (self.traj_count + 1) % self.dynamics_randomize_every == 0:
            self.resample_dynamics()

        # 出生盒会逐步扩张到预设大小，让训练早期的起始状态更保守，后期再覆盖更大空间。
        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.spawn_point

        # 降维模式会锁住部分坐标，使任务退化成更简单的竖直或平面控制问题。
        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]

        # 为避免出生即判定落地碰撞，初始高度至少抬到 0.75m。
        if z < 0.75:
            z = 0.75
        pos = npa(x, y, z)

        # 这里决定初始速度、姿态和角速度是随机采样还是以更规整的“朝向目标、速度为零”方式启动。
        if self.init_random_state:
            if self.dim_mode == '1D':
                omega, rotation = np.zeros(3, dtype=np.float64), np.eye(3)
                vel = np.array([0, 0, self.max_init_vel * np.random.rand()])
            elif self.dim_mode == '2D':
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                vel = self.max_init_vel * np.random.rand(3)
                vel[1] = 0.
                theta = np.pi * np.random.rand()
                c, s = np.cos(theta), np.sin(theta)
                rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
            else:
                _, vel, rotation, omega = self.dynamics.random_state(
                    box=(self.room_length, self.room_width, self.room_height), vel_max=self.max_init_vel,
                    omega_max=self.max_init_omega
                )
        else:
            vel, omega = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                # 三维模式下初始朝向会尽量对着目标，以减轻高层控制在训练早期承担的纯姿态恢复负担。
                rotation = randyaw()
                while np.dot(rotation[:, 0], to_xyhat(-pos)) < 0.5:
                    rotation = randyaw()

        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.dynamics.reset()
        self.dynamics.on_floor = False
        self.dynamics.crashed_floor = self.dynamics.crashed_wall = self.dynamics.crashed_ceiling = False

        # episode 计数器和最近动作缓存也在这里清零，确保上一条轨迹的控制历史不会泄漏到下一条轨迹。
        self.tick = 0
        self.actions = [np.zeros([4, ]), np.zeros([4, ])]

        state = self.state_vector(self)
        return state

    def reset(self):
        return self._reset()

    def render(self, **kwargs):
        """This class is only meant to be used as a component of QuadMultiEnv."""
        raise NotImplementedError()

    def step(self, action):
        return self._step(action)
