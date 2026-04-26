# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_single.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

#!/usr/bin/env python
# 下面开始文档字符串说明。
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
# 导入当前模块依赖。
import copy

# 导入当前模块依赖。
from gymnasium.utils import seeding

# 导入当前模块依赖。
import gym_art.quadrotor_multi.get_state as get_state
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics
from gym_art.quadrotor_multi.sensor_noise import SensorNoise

# 保存或更新 `GRAV` 的值。
GRAV = 9.81  # default gravitational constant


# reasonable reward function for hovering at a goal and not flying too high
# 定义函数 `compute_reward_weighted`。
def compute_reward_weighted(dynamics, goal, action, dt, time_remain, rew_coeff, action_prev, on_floor=False):
    # Distance to the goal
    # 保存或更新 `dist` 的值。
    dist = np.linalg.norm(goal - dynamics.pos)
    # 保存或更新 `cost_pos_raw` 的值。
    cost_pos_raw = dist
    # 保存或更新 `cost_pos` 的值。
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    # Penalize amount of control effort
    # 保存或更新 `cost_effort_raw` 的值。
    cost_effort_raw = np.linalg.norm(action)
    # 保存或更新 `cost_effort` 的值。
    cost_effort = rew_coeff["effort"] * cost_effort_raw

    # Loss orientation
    # 根据条件决定是否进入当前分支。
    if on_floor:
        # 保存或更新 `cost_orient_raw` 的值。
        cost_orient_raw = 1.0
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `cost_orient_raw` 的值。
        cost_orient_raw = -dynamics.rot[2, 2]

    # 保存或更新 `cost_orient` 的值。
    cost_orient = rew_coeff["orient"] * cost_orient_raw

    # Loss for constant uncontrolled rotation around vertical axis
    # 保存或更新 `cost_spin_raw` 的值。
    cost_spin_raw = (dynamics.omega[0] ** 2 + dynamics.omega[1] ** 2 + dynamics.omega[2] ** 2) ** 0.5
    # 保存或更新 `cost_spin` 的值。
    cost_spin = rew_coeff["spin"] * cost_spin_raw

    # Loss crash for staying on the floor
    # 保存或更新 `cost_crash_raw` 的值。
    cost_crash_raw = float(on_floor)
    # 保存或更新 `cost_crash` 的值。
    cost_crash = rew_coeff["crash"] * cost_crash_raw

    # 保存或更新 `reward` 的值。
    reward = -dt * np.sum([
        cost_pos,
        cost_effort,
        cost_crash,
        cost_orient,
        cost_spin,
    ])

    # 保存或更新 `rew_info` 的值。
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

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for k, v in rew_info.items():
        # 保存或更新 `rew_info[k]` 的值。
        rew_info[k] = dt * v

    # 根据条件决定是否进入当前分支。
    if np.isnan(reward) or not np.isfinite(reward):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for key, value in locals().items():
            # 调用 `print` 执行当前处理。
            print('%s: %s \n' % (key, str(value)))
        # 主动抛出异常以中止或提示错误。
        raise ValueError('QuadEnv: reward is Nan')

    # 返回当前函数的结果。
    return reward, rew_info


# ENV Gym environment for quadrotor seeking the origin with no obstacles and full state observations. NOTES: - room
# size of the env and init state distribution are not the same ! It is done for the reason of having static (and
# preferably short) episode length, since for some distance it would be impossible to reach the goal
# 定义类 `QuadrotorSingle`。
class QuadrotorSingle:
    # 定义函数 `__init__`。
    def __init__(self, dynamics_params="DefaultQuad", dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr="xyz_vxyz_R_omega", ep_time=7, room_dims=(10.0, 10.0, 10.0),
                 init_random_state=False, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False, use_numba=False,
                 # 保存或更新 `neighbor_obs_type` 的值。
                 neighbor_obs_type='none', num_agents=1, num_use_neighbor_obs=0, use_obstacles=False):
        # 保存或更新 `np.seterr(under` 的值。
        np.seterr(under='ignore')
        # 下面开始文档字符串说明。
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
        # Numba Speed Up
        # 保存或更新 `use_numba` 的值。
        self.use_numba = use_numba

        # Room
        # 保存或更新 `room_length` 的值。
        self.room_length = room_dims[0]
        # 保存或更新 `room_width` 的值。
        self.room_width = room_dims[1]
        # 保存或更新 `room_height` 的值。
        self.room_height = room_dims[2]
        # 保存或更新 `room_box` 的值。
        self.room_box = np.array([[-self.room_length / 2., -self.room_width / 2, 0.],
                                  [self.room_length / 2., self.room_width / 2., self.room_height]])

        # 保存或更新 `init_random_state` 的值。
        self.init_random_state = init_random_state

        # Preset parameters
        # 保存或更新 `obs_repr` 的值。
        self.obs_repr = obs_repr
        # 保存或更新 `rew_coeff` 的值。
        self.rew_coeff = None
        # EPISODE PARAMS
        # 保存或更新 `ep_time` 的值。
        self.ep_time = ep_time  # In seconds
        # 保存或更新 `sim_steps` 的值。
        self.sim_steps = sim_steps
        # 保存或更新 `dt` 的值。
        self.dt = 1.0 / sim_freq
        # 保存或更新 `ep_len` 的值。
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        # 保存或更新 `tick` 的值。
        self.tick = 0
        # 保存或更新 `control_freq` 的值。
        self.control_freq = sim_freq / sim_steps
        # 保存或更新 `traj_count` 的值。
        self.traj_count = 0

        # Self dynamics
        # 保存或更新 `dim_mode` 的值。
        self.dim_mode = dim_mode
        # 保存或更新 `raw_control_zero_middle` 的值。
        self.raw_control_zero_middle = raw_control_zero_middle
        # 保存或更新 `tf_control` 的值。
        self.tf_control = tf_control
        # 保存或更新 `dynamics_randomize_every` 的值。
        self.dynamics_randomize_every = dynamics_randomize_every
        # 保存或更新 `verbose` 的值。
        self.verbose = verbose
        # 保存或更新 `raw_control` 的值。
        self.raw_control = raw_control
        # 保存或更新 `gravity` 的值。
        self.gravity = gravity
        # 保存或更新 `update_sense_noise(sense_noise` 的值。
        self.update_sense_noise(sense_noise=sense_noise)
        # 保存或更新 `t2w_std` 的值。
        self.t2w_std = t2w_std
        # 保存或更新 `t2w_min` 的值。
        self.t2w_min = 1.5
        # 保存或更新 `t2w_max` 的值。
        self.t2w_max = 10.0

        # 保存或更新 `t2t_std` 的值。
        self.t2t_std = t2t_std
        # 保存或更新 `t2t_min` 的值。
        self.t2t_min = 0.005
        # 保存或更新 `t2t_max` 的值。
        self.t2t_max = 1.0
        # 保存或更新 `excite` 的值。
        self.excite = excite
        # 保存或更新 `dynamics_simplification` 的值。
        self.dynamics_simplification = dynamics_simplification
        # 保存或更新 `max_init_vel` 的值。
        self.max_init_vel = 1.  # m/s
        # 保存或更新 `max_init_omega` 的值。
        self.max_init_omega = 2 * np.pi  # rad/s

        # DYNAMICS (and randomization)
        # Could be dynamics of a specific quad or a random dynamics (i.e. randomquad)
        # 保存或更新 `dyn_base_sampler` 的值。
        self.dyn_base_sampler = getattr(quad_rand, dynamics_params)()
        # 保存或更新 `dynamics_change` 的值。
        self.dynamics_change = copy.deepcopy(dynamics_change)
        # 保存或更新 `dynamics_params` 的值。
        self.dynamics_params = self.dyn_base_sampler.sample()
        # Now, updating if we are providing modifications
        # 根据条件决定是否进入当前分支。
        if self.dynamics_change is not None:
            # 调用 `dict_update_existing` 执行当前处理。
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        # 保存或更新 `dyn_sampler_1` 的值。
        self.dyn_sampler_1 = dyn_sampler_1
        # 根据条件决定是否进入当前分支。
        if dyn_sampler_1 is not None:
            # 保存或更新 `sampler_type` 的值。
            sampler_type = dyn_sampler_1["class"]
            # 保存或更新 `dyn_sampler_1_params` 的值。
            self.dyn_sampler_1_params = copy.deepcopy(dyn_sampler_1)
            # 执行这一行逻辑。
            del self.dyn_sampler_1_params["class"]
            # 保存或更新 `dyn_sampler_1` 的值。
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_1_params)

        # 保存或更新 `dyn_sampler_2` 的值。
        self.dyn_sampler_2 = dyn_sampler_2
        # 根据条件决定是否进入当前分支。
        if dyn_sampler_2 is not None:
            # 保存或更新 `sampler_type` 的值。
            sampler_type = dyn_sampler_2["class"]
            # 保存或更新 `dyn_sampler_2_params` 的值。
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            # 执行这一行逻辑。
            del self.dyn_sampler_2_params["class"]
            # 保存或更新 `dyn_sampler_2` 的值。
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_2_params)

        # Updating dynamics
        # 保存或更新 `action_space` 的值。
        self.action_space = None
        # 调用 `resample_dynamics` 执行当前处理。
        self.resample_dynamics()

        # Self info
        # 保存或更新 `state_vector` 的值。
        self.state_vector = self.state_vector = getattr(get_state, "state_" + self.obs_repr)
        # 根据条件决定是否进入当前分支。
        if use_obstacles:
            # 保存或更新 `box` 的值。
            self.box = 0.1
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `box` 的值。
            self.box = 2.0
        # 保存或更新 `box_scale` 的值。
        self.box_scale = 1.0
        # 保存或更新 `goal` 的值。
        self.goal = None
        # 保存或更新 `spawn_point` 的值。
        self.spawn_point = None

        # Neighbor info
        # 保存或更新 `num_agents` 的值。
        self.num_agents = num_agents
        # 保存或更新 `neighbor_obs_type` 的值。
        self.neighbor_obs_type = neighbor_obs_type
        # 保存或更新 `num_use_neighbor_obs` 的值。
        self.num_use_neighbor_obs = num_use_neighbor_obs

        # Obstacles info
        # 保存或更新 `use_obstacles` 的值。
        self.use_obstacles = use_obstacles

        # Make observation space
        # 保存或更新 `observation_space` 的值。
        self.observation_space = self.make_observation_space()

        # 调用 `_seed` 执行当前处理。
        self._seed()

    # 定义函数 `update_sense_noise`。
    def update_sense_noise(self, sense_noise):
        # 根据条件决定是否进入当前分支。
        if isinstance(sense_noise, dict):
            # 保存或更新 `sense_noise` 的值。
            self.sense_noise = SensorNoise(**sense_noise)
        # 当上一分支不满足时，继续判断新的条件。
        elif isinstance(sense_noise, str):
            # 根据条件决定是否进入当前分支。
            if sense_noise == "default":
                # 保存或更新 `sense_noise` 的值。
                self.sense_noise = SensorNoise(bypass=False, use_numba=self.use_numba)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 调用 `ValueError` 执行当前处理。
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        # 当上一分支不满足时，继续判断新的条件。
        elif sense_noise is None:
            # 保存或更新 `sense_noise` 的值。
            self.sense_noise = SensorNoise(bypass=True)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    # 定义函数 `update_dynamics`。
    def update_dynamics(self, dynamics_params):
        # DYNAMICS
        # Then loading the dynamics
        # 保存或更新 `dynamics_params` 的值。
        self.dynamics_params = dynamics_params
        # 保存或更新 `dynamics` 的值。
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params,
                                          dynamics_steps_num=self.sim_steps, room_box=self.room_box,
                                          dim_mode=self.dim_mode, gravity=self.gravity,
                                          dynamics_simplification=self.dynamics_simplification,
                                          use_numba=self.use_numba, dt=self.dt)

        # CONTROL
        # 根据条件决定是否进入当前分支。
        if self.raw_control:
            # 根据条件决定是否进入当前分支。
            if self.dim_mode == '1D':  # Z axis only
                # 保存或更新 `controller` 的值。
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            # 当上一分支不满足时，继续判断新的条件。
            elif self.dim_mode == '2D':  # X and Z axes only
                # 保存或更新 `controller` 的值。
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            # 当上一分支不满足时，继续判断新的条件。
            elif self.dim_mode == '3D':
                # 保存或更新 `controller` 的值。
                self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 主动抛出异常以中止或提示错误。
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `controller` 的值。
            self.controller = NonlinearPositionController(self.dynamics, tf_control=self.tf_control)

        # ACTIONS
        # 保存或更新 `action_space` 的值。
        self.action_space = self.controller.action_space(self.dynamics)

        # STATE VECTOR FUNCTION
        # 保存或更新 `state_vector` 的值。
        self.state_vector = getattr(get_state, "state_" + self.obs_repr)

    # 定义函数 `make_observation_space`。
    def make_observation_space(self):
        # 保存或更新 `room_range` 的值。
        room_range = self.room_box[1] - self.room_box[0]
        # 保存或更新 `obs_space_low_high` 的值。
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
            "rxyz": [-room_range, room_range],  # rxyz stands for relative pos between quadrotors
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)],
            # rvxyz stands for relative velocity between quadrotors
            "roxyz": [-room_range, room_range],  # roxyz stands for relative pos between quadrotor and obstacle
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)],
            # rovxyz stands for relative velocity between quadrotor and obstacle
            "osize": [np.zeros(3), 20.0 * np.ones(3)],  # obstacle size, [[0., 0., 0.], [20., 20., 20.]]
            "otype": [np.zeros(1), 20.0 * np.ones(1)],
            # obstacle type, [[0.], [20.]], which means we can support 21 types of obstacles
            "goal": [-room_range, room_range],
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
            "floor": [np.zeros(1), self.room_box[1][2] * np.ones(1)],
            "octmap": [-10 * np.ones(9), 10 * np.ones(9)],
        }
        # 保存或更新 `obs_comp_names` 的值。
        self.obs_comp_names = list(self.obs_space_low_high.keys())
        # 保存或更新 `obs_comp_sizes` 的值。
        self.obs_comp_sizes = [self.obs_space_low_high[name][1].size for name in self.obs_comp_names]

        # 保存或更新 `obs_comps` 的值。
        obs_comps = self.obs_repr.split("_")
        # 根据条件决定是否进入当前分支。
        if self.neighbor_obs_type == 'pos_vel' and self.num_use_neighbor_obs > 0:
            # 保存或更新 `obs_comps` 的值。
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs

        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `obs_comps` 的值。
            obs_comps = obs_comps + ["octmap"]

        # 调用 `print` 执行当前处理。
        print("Observation components:", obs_comps)
        # 同时更新 `obs_low`, `obs_high` 等变量。
        obs_low, obs_high = [], []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for comp in obs_comps:
            # 调用 `append` 执行当前处理。
            obs_low.append(self.obs_space_low_high[comp][0])
            # 调用 `append` 执行当前处理。
            obs_high.append(self.obs_space_low_high[comp][1])
        # 保存或更新 `obs_low` 的值。
        obs_low = np.concatenate(obs_low)
        # 保存或更新 `obs_high` 的值。
        obs_high = np.concatenate(obs_high)

        # 同时更新 `obs_comp_sizes_dict`, `obs_space_comp_indx`, `obs_comp_end` 等变量。
        self.obs_comp_sizes_dict, self.obs_space_comp_indx, self.obs_comp_end = {}, {}, []
        # 保存或更新 `end_indx` 的值。
        end_indx = 0
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for obs_i, obs_name in enumerate(self.obs_comp_names):
            # 保存或更新 `end_indx` 的值。
            end_indx += self.obs_comp_sizes[obs_i]
            # 保存或更新 `obs_comp_sizes_dict[obs_name]` 的值。
            self.obs_comp_sizes_dict[obs_name] = self.obs_comp_sizes[obs_i]
            # 保存或更新 `obs_space_comp_indx[obs_name]` 的值。
            self.obs_space_comp_indx[obs_name] = obs_i
            # 调用 `append` 执行当前处理。
            self.obs_comp_end.append(end_indx)

        # 保存或更新 `observation_space` 的值。
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        # 返回当前函数的结果。
        return self.observation_space

    # 定义函数 `_seed`。
    def _seed(self, seed=None):
        # 同时更新 `np_random`, `seed` 等变量。
        self.np_random, seed = seeding.np_random(seed)
        # 返回当前函数的结果。
        return [seed]

    # 定义函数 `_step`。
    def _step(self, action):
        # 保存或更新 `actions[1]` 的值。
        self.actions[1] = copy.deepcopy(self.actions[0])
        # 保存或更新 `actions[0]` 的值。
        self.actions[0] = copy.deepcopy(action)

        # 保存或更新 `controller.step_func(dynamics` 的值。
        self.controller.step_func(dynamics=self.dynamics, action=action, goal=self.goal, dt=self.dt, observation=None)

        # 保存或更新 `time_remain` 的值。
        self.time_remain = self.ep_len - self.tick
        # 同时更新 `reward`, `rew_info` 等变量。
        reward, rew_info = compute_reward_weighted(
            dynamics=self.dynamics, goal=self.goal, action=action, dt=self.dt, time_remain=self.time_remain,
            rew_coeff=self.rew_coeff, action_prev=self.actions[1], on_floor=self.dynamics.on_floor)

        # 保存或更新 `tick` 的值。
        self.tick += 1
        # 保存或更新 `done` 的值。
        done = self.tick > self.ep_len
        # 保存或更新 `sv` 的值。
        sv = self.state_vector(self)
        # 保存或更新 `traj_count` 的值。
        self.traj_count += int(done)

        # 返回当前函数的结果。
        return sv, reward, done, {'rewards': rew_info}

    # 定义函数 `resample_dynamics`。
    def resample_dynamics(self):
        # 下面开始文档字符串说明。
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        # Getting base parameters (could also be random parameters)
        # 保存或更新 `dynamics_params` 的值。
        self.dynamics_params = self.dyn_base_sampler.sample()

        # Now, updating if we are providing modifications
        # 根据条件决定是否进入当前分支。
        if self.dynamics_change is not None:
            # 调用 `dict_update_existing` 执行当前处理。
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        # Applying sampler 1
        # 根据条件决定是否进入当前分支。
        if self.dyn_sampler_1 is not None:
            # 保存或更新 `dynamics_params` 的值。
            self.dynamics_params = self.dyn_sampler_1.sample(self.dynamics_params)

        # Applying sampler 2
        # 根据条件决定是否进入当前分支。
        if self.dyn_sampler_2 is not None:
            # 保存或更新 `dynamics_params` 的值。
            self.dynamics_params = self.dyn_sampler_2.sample(self.dynamics_params)

        # Checking that quad params make sense
        # 调用 `check_quad_param_limits` 执行当前处理。
        quad_rand.check_quad_param_limits(self.dynamics_params)

        # Updating params
        # 保存或更新 `update_dynamics(dynamics_params` 的值。
        self.update_dynamics(dynamics_params=self.dynamics_params)

    # 定义函数 `_reset`。
    def _reset(self):
        # DYNAMICS RANDOMIZATION AND UPDATE
        # 根据条件决定是否进入当前分支。
        if self.dynamics_randomize_every is not None and (self.traj_count + 1) % self.dynamics_randomize_every == 0:
            # 调用 `resample_dynamics` 执行当前处理。
            self.resample_dynamics()

        # 根据条件决定是否进入当前分支。
        if self.box < 10:
            # 保存或更新 `box` 的值。
            self.box = self.box * self.box_scale
        # 同时更新 `x`, `y`, `z` 等变量。
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.spawn_point

        # 根据条件决定是否进入当前分支。
        if self.dim_mode == '1D':
            # 同时更新 `x`, `y` 等变量。
            x, y = self.goal[0], self.goal[1]
        # 当上一分支不满足时，继续判断新的条件。
        elif self.dim_mode == '2D':
            # 保存或更新 `y` 的值。
            y = self.goal[1]
        # Since being near the groud means crash we have to start above
        # 根据条件决定是否进入当前分支。
        if z < 0.75:
            # 保存或更新 `z` 的值。
            z = 0.75
        # 保存或更新 `pos` 的值。
        pos = npa(x, y, z)

        # INIT STATE
        # Initializing rotation and velocities
        # 根据条件决定是否进入当前分支。
        if self.init_random_state:
            # 根据条件决定是否进入当前分支。
            if self.dim_mode == '1D':
                # 同时更新 `omega`, `rotation` 等变量。
                omega, rotation = np.zeros(3, dtype=np.float64), np.eye(3)
                # 保存或更新 `vel` 的值。
                vel = np.array([0, 0, self.max_init_vel * np.random.rand()])
            # 当上一分支不满足时，继续判断新的条件。
            elif self.dim_mode == '2D':
                # 保存或更新 `omega` 的值。
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                # 保存或更新 `vel` 的值。
                vel = self.max_init_vel * np.random.rand(3)
                # 保存或更新 `vel[1]` 的值。
                vel[1] = 0.
                # 保存或更新 `theta` 的值。
                theta = np.pi * np.random.rand()
                # 同时更新 `c`, `s` 等变量。
                c, s = np.cos(theta), np.sin(theta)
                # 保存或更新 `rotation` 的值。
                rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # It already sets the state internally
                # 同时更新 `_`, `vel`, `rotation`, `omega` 等变量。
                _, vel, rotation, omega = self.dynamics.random_state(
                    box=(self.room_length, self.room_width, self.room_height), vel_max=self.max_init_vel,
                    omega_max=self.max_init_omega
                )
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # INIT HORIZONTALLY WITH 0 VEL and OMEGA
            # 同时更新 `vel`, `omega` 等变量。
            vel, omega = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

            # 根据条件决定是否进入当前分支。
            if self.dim_mode == '1D' or self.dim_mode == '2D':
                # 保存或更新 `rotation` 的值。
                rotation = np.eye(3)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # make sure we're sort of pointing towards goal (for mellinger controller)
                # 保存或更新 `rotation` 的值。
                rotation = randyaw()
                # 在条件成立时持续执行下面的循环体。
                while np.dot(rotation[:, 0], to_xyhat(-pos)) < 0.5:
                    # 保存或更新 `rotation` 的值。
                    rotation = randyaw()

        # 保存或更新 `init_state` 的值。
        self.init_state = [pos, vel, rotation, omega]
        # 调用 `set_state` 执行当前处理。
        self.dynamics.set_state(pos, vel, rotation, omega)
        # 调用 `reset` 执行当前处理。
        self.dynamics.reset()
        # 保存或更新 `dynamics.on_floor` 的值。
        self.dynamics.on_floor = False
        # 保存或更新 `dynamics.crashed_floor` 的值。
        self.dynamics.crashed_floor = self.dynamics.crashed_wall = self.dynamics.crashed_ceiling = False

        # Reseting some internal state (counters, etc)
        # 保存或更新 `tick` 的值。
        self.tick = 0
        # 保存或更新 `actions` 的值。
        self.actions = [np.zeros([4, ]), np.zeros([4, ])]

        # 保存或更新 `state` 的值。
        state = self.state_vector(self)
        # 返回当前函数的结果。
        return state

    # 定义函数 `reset`。
    def reset(self):
        # 返回当前函数的结果。
        return self._reset()

    # 定义函数 `render`。
    def render(self, **kwargs):
        # 下面的文档字符串用于说明当前模块或代码块。
        """This class is only meant to be used as a component of QuadMultiEnv."""
        # 主动抛出异常以中止或提示错误。
        raise NotImplementedError()

    # 定义函数 `step`。
    def step(self, action):
        # 返回当前函数的结果。
        return self._step(action)
