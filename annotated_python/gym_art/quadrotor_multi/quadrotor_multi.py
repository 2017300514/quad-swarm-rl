# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_multi.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy
import time
from collections import deque
from copy import deepcopy

# 导入当前模块依赖。
import gymnasium as gym
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    # 执行这一行逻辑。
    calculate_drone_proximity_penalties, perform_collision_between_drones
# 导入当前模块依赖。
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

# 导入当前模块依赖。
from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario


# 定义类 `QuadrotorEnvMulti`。
class QuadrotorEnvMulti(gym.Env):
    # 定义函数 `__init__`。
    def __init__(self, num_agents, ep_time, rew_coeff, obs_repr,
                 # Neighbor
                 neighbor_visible_num, neighbor_obs_type, collision_hitbox_radius, collision_falloff_radius,

                 # Obstacle
                 use_obstacles, obst_density, obst_size, obst_spawn_area,

                 # Aerodynamics, Numba Speed Up, Scenarios, Room, Replay Buffer, Rendering
                 use_downwash, use_numba, quads_mode, room_dims, use_replay_buffer, quads_view_mode,
                 quads_render,

                 # Quadrotor Specific (Do Not Change)
                 dynamics_params, raw_control, raw_control_zero_middle,
                 dynamics_randomize_every, dynamics_change, dyn_sampler_1,
                 sense_noise, init_random_state,
                 # Rendering
                 render_mode='human'
                 # 这里开始一个新的代码块。
                 ):
        # 调用 `super` 执行当前处理。
        super().__init__()

        # Predefined Parameters
        # 保存或更新 `num_agents` 的值。
        self.num_agents = num_agents
        # 保存或更新 `obs_self_size` 的值。
        obs_self_size = QUADS_OBS_REPR[obs_repr]
        # 根据条件决定是否进入当前分支。
        if neighbor_visible_num == -1:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = self.num_agents - 1
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = neighbor_visible_num

        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        # 保存或更新 `is_multiagent` 的值。
        self.is_multiagent = True
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims
        # 保存或更新 `quads_view_mode` 的值。
        self.quads_view_mode = quads_view_mode

        # Generate All Quadrotors
        # 保存或更新 `envs` 的值。
        self.envs = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(self.num_agents):
            # 保存或更新 `e` 的值。
            e = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr, ep_time=ep_time, room_dims=room_dims,
                use_numba=use_numba,
                # Neighbor
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=use_obstacles,
            )
            # 调用 `append` 执行当前处理。
            self.envs.append(e)

        # Set Obs & Act
        # 保存或更新 `observation_space` 的值。
        self.observation_space = self.envs[0].observation_space
        # 保存或更新 `action_space` 的值。
        self.action_space = self.envs[0].action_space

        # Aux variables
        # 保存或更新 `quad_arm` 的值。
        self.quad_arm = self.envs[0].dynamics.arm
        # 保存或更新 `control_freq` 的值。
        self.control_freq = self.envs[0].control_freq
        # 保存或更新 `control_dt` 的值。
        self.control_dt = 1.0 / self.control_freq
        # 保存或更新 `pos` 的值。
        self.pos = np.zeros([self.num_agents, 3])
        # 保存或更新 `vel` 的值。
        self.vel = np.zeros([self.num_agents, 3])
        # 保存或更新 `omega` 的值。
        self.omega = np.zeros([self.num_agents, 3])
        # 保存或更新 `rel_pos` 的值。
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))
        # 保存或更新 `rel_vel` 的值。
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))

        # Reward
        # 保存或更新 `rew_coeff` 的值。
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5.
        )
        # 保存或更新 `rew_coeff_orig` 的值。
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        # 根据条件决定是否进入当前分支。
        if rew_coeff is not None:
            # 断言当前条件成立，用于保护运行假设。
            assert isinstance(rew_coeff, dict)
            # 断言当前条件成立，用于保护运行假设。
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            # 调用 `update` 执行当前处理。
            self.rew_coeff.update(rew_coeff)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for key in self.rew_coeff.keys():
            # 保存或更新 `rew_coeff[key]` 的值。
            self.rew_coeff[key] = float(self.rew_coeff[key])

        # 保存或更新 `orig_keys` 的值。
        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        # 断言当前条件成立，用于保护运行假设。
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # Neighbors
        # 保存或更新 `neighbor_obs_size` 的值。
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]

        # 保存或更新 `clip_neighbor_space_length` 的值。
        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        # 保存或更新 `clip_neighbor_space_min_box` 的值。
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        # 保存或更新 `clip_neighbor_space_max_box` 的值。
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        # Obstacles
        # 保存或更新 `use_obstacles` 的值。
        self.use_obstacles = use_obstacles
        # 保存或更新 `obstacles` 的值。
        self.obstacles = None
        # 保存或更新 `num_obstacles` 的值。
        self.num_obstacles = 0
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `prev_obst_quad_collisions` 的值。
            self.prev_obst_quad_collisions = []
            # 保存或更新 `obst_quad_collisions_per_episode` 的值。
            self.obst_quad_collisions_per_episode = 0
            # 保存或更新 `obst_quad_collisions_after_settle` 的值。
            self.obst_quad_collisions_after_settle = 0
            # 保存或更新 `curr_quad_col` 的值。
            self.curr_quad_col = []
            # 保存或更新 `obst_density` 的值。
            self.obst_density = obst_density
            # 保存或更新 `obst_spawn_area` 的值。
            self.obst_spawn_area = obst_spawn_area
            # 保存或更新 `num_obstacles` 的值。
            self.num_obstacles = int(obst_density * obst_spawn_area[0] * obst_spawn_area[1])
            # 保存或更新 `obst_map` 的值。
            self.obst_map = None
            # 保存或更新 `obst_size` 的值。
            self.obst_size = obst_size

            # Log more info
            # 保存或更新 `distance_to_goal_3_5` 的值。
            self.distance_to_goal_3_5 = 0
            # 保存或更新 `distance_to_goal_5` 的值。
            self.distance_to_goal_5 = 0

        # Scenarios
        # 保存或更新 `quads_mode` 的值。
        self.quads_mode = quads_mode
        # 保存或更新 `scenario` 的值。
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # Collisions
        # # Collisions: Neighbors
        # 保存或更新 `collisions_per_episode` 的值。
        self.collisions_per_episode = 0
        # # # Ignore collisions because of spawn
        # 保存或更新 `collisions_after_settle` 的值。
        self.collisions_after_settle = 0
        # 保存或更新 `collisions_grace_period_steps` 的值。
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        # 保存或更新 `collisions_grace_period_seconds` 的值。
        self.collisions_grace_period_seconds = 1.5
        # 保存或更新 `prev_drone_collisions` 的值。
        self.prev_drone_collisions = []

        # 保存或更新 `collisions_final_grace_period_steps` 的值。
        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        # 保存或更新 `collisions_final_5s` 的值。
        self.collisions_final_5s = 0

        # # # Dense reward info
        # 保存或更新 `collision_threshold` 的值。
        self.collision_threshold = collision_hitbox_radius * self.quad_arm
        # 保存或更新 `collision_falloff_threshold` 的值。
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm

        # # Collisions: Room
        # 保存或更新 `collisions_room_per_episode` 的值。
        self.collisions_room_per_episode = 0
        # 保存或更新 `collisions_floor_per_episode` 的值。
        self.collisions_floor_per_episode = 0
        # 保存或更新 `collisions_wall_per_episode` 的值。
        self.collisions_wall_per_episode = 0
        # 保存或更新 `collisions_ceiling_per_episode` 的值。
        self.collisions_ceiling_per_episode = 0

        # 保存或更新 `prev_crashed_walls` 的值。
        self.prev_crashed_walls = []
        # 保存或更新 `prev_crashed_ceiling` 的值。
        self.prev_crashed_ceiling = []
        # 保存或更新 `prev_crashed_room` 的值。
        self.prev_crashed_room = []

        # Replay
        # 保存或更新 `use_replay_buffer` 的值。
        self.use_replay_buffer = use_replay_buffer
        # # only start using the buffer after the drones learn how to fly
        # 保存或更新 `activate_replay_buffer` 的值。
        self.activate_replay_buffer = False
        # # since the same collisions happen during replay, we don't want to keep resaving the same event
        # 保存或更新 `saved_in_replay_buffer` 的值。
        self.saved_in_replay_buffer = False
        # 保存或更新 `last_step_unique_collisions` 的值。
        self.last_step_unique_collisions = False
        # 保存或更新 `crashes_in_recent_episodes` 的值。
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        # 保存或更新 `crashes_last_episode` 的值。
        self.crashes_last_episode = 0

        # Numba
        # 保存或更新 `use_numba` 的值。
        self.use_numba = use_numba

        # Aerodynamics
        # 保存或更新 `use_downwash` 的值。
        self.use_downwash = use_downwash

        # Rendering
        # # set to true whenever we need to reset the OpenGL scene in render()
        # 保存或更新 `render_mode` 的值。
        self.render_mode =render_mode
        # 保存或更新 `quads_render` 的值。
        self.quads_render = quads_render
        # 保存或更新 `scenes` 的值。
        self.scenes = []
        # 根据条件决定是否进入当前分支。
        if self.quads_render:
            # 保存或更新 `reset_scene` 的值。
            self.reset_scene = False
            # 保存或更新 `simulation_start_time` 的值。
            self.simulation_start_time = 0
            # 保存或更新 `frames_since_last_render` 的值。
            self.frames_since_last_render = self.render_skip_frames = 0
            # 保存或更新 `render_every_nth_frame` 的值。
            self.render_every_nth_frame = 1
            # # Use this to control rendering speed
            # 保存或更新 `render_speed` 的值。
            self.render_speed = 1.0
            # 保存或更新 `quads_formation_size` 的值。
            self.quads_formation_size = 2.0
            # 保存或更新 `all_collisions` 的值。
            self.all_collisions = {}

        # Log
        # 保存或更新 `distance_to_goal` 的值。
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        # 保存或更新 `reached_goal` 的值。
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Log metric
        # 保存或更新 `agent_col_agent` 的值。
        self.agent_col_agent = np.ones(self.num_agents)
        # 保存或更新 `agent_col_obst` 的值。
        self.agent_col_obst = np.ones(self.num_agents)

        # Others
        # 保存或更新 `apply_collision_force` 的值。
        self.apply_collision_force = True

    # 定义函数 `all_dynamics`。
    def all_dynamics(self):
        # 返回当前函数的结果。
        return tuple(e.dynamics for e in self.envs)

    # 定义函数 `get_rel_pos_vel_item`。
    def get_rel_pos_vel_item(self, env_id, indices=None):
        # 保存或更新 `i` 的值。
        i = env_id

        # 根据条件决定是否进入当前分支。
        if indices is None:
            # if not specified explicitly, consider all neighbors
            # 执行这一行逻辑。
            indices = [j for j in range(self.num_agents) if j != i]

        # 保存或更新 `cur_pos` 的值。
        cur_pos = self.pos[i]
        # 保存或更新 `cur_vel` 的值。
        cur_vel = self.vel[i]
        # 保存或更新 `pos_neighbor` 的值。
        pos_neighbor = np.stack([self.pos[j] for j in indices])
        # 保存或更新 `vel_neighbor` 的值。
        vel_neighbor = np.stack([self.vel[j] for j in indices])
        # 保存或更新 `pos_rel` 的值。
        pos_rel = pos_neighbor - cur_pos
        # 保存或更新 `vel_rel` 的值。
        vel_rel = vel_neighbor - cur_vel
        # 返回当前函数的结果。
        return pos_rel, vel_rel

    # 定义函数 `get_obs_neighbor_rel`。
    def get_obs_neighbor_rel(self, env_id, closest_drones):
        # 保存或更新 `i` 的值。
        i = env_id
        # 同时更新 `pos_neighbors_rel`, `vel_neighbors_rel` 等变量。
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        # 保存或更新 `obs_neighbor_rel` 的值。
        obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        # 返回当前函数的结果。
        return obs_neighbor_rel

    # 定义函数 `extend_obs_space`。
    def extend_obs_space(self, obs, closest_drones):
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(len(self.envs)):
            # 保存或更新 `obs_neighbor_rel` 的值。
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            # 调用 `append` 执行当前处理。
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        # 保存或更新 `obs_ext` 的值。
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        # 返回当前函数的结果。
        return obs_ext

    # 定义函数 `neighborhood_indices`。
    def neighborhood_indices(self):
        # 下面的文档字符串用于说明当前模块或代码块。
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        # 执行这一行逻辑。
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        # 保存或更新 `indices` 的值。
        indices = np.array(indices)

        # 根据条件决定是否进入当前分支。
        if self.num_use_neighbor_obs == self.num_agents - 1:
            # 返回当前函数的结果。
            return indices
        # 当上一分支不满足时，继续判断新的条件。
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            # 保存或更新 `close_neighbor_indices` 的值。
            close_neighbor_indices = []

            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(self.num_agents):
                # 同时更新 `rel_pos`, `rel_vel` 等变量。
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])
                # 保存或更新 `rel_dist` 的值。
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                # 保存或更新 `rel_dist` 的值。
                rel_dist = np.maximum(rel_dist, 0.01)
                # 保存或更新 `rel_pos_unit` 的值。
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # the smaller the new_rel_dist, the closer the drones
                # 保存或更新 `new_rel_dist` 的值。
                new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)

                # 保存或更新 `rel_pos_index` 的值。
                rel_pos_index = new_rel_dist.argsort()
                # 保存或更新 `rel_pos_index` 的值。
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                # 调用 `append` 执行当前处理。
                close_neighbor_indices.append(indices[i][rel_pos_index])

            # 返回当前函数的结果。
            return close_neighbor_indices
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise RuntimeError("Incorrect number of neigbors")

    # 定义函数 `add_neighborhood_obs`。
    def add_neighborhood_obs(self, obs):
        # 保存或更新 `indices` 的值。
        indices = self.neighborhood_indices()
        # 保存或更新 `obs_ext` 的值。
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        # 返回当前函数的结果。
        return obs_ext

    # 定义函数 `can_drones_fly`。
    def can_drones_fly(self):
        # 下面开始文档字符串说明。
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        # 执行这一行逻辑。
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        # 返回当前函数的结果。
        return res

    # 定义函数 `calculate_room_collision`。
    def calculate_room_collision(self):
        # 保存或更新 `floor_collisions` 的值。
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        # 保存或更新 `wall_collisions` 的值。
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        # 保存或更新 `ceiling_collisions` 的值。
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        # 执行这一行逻辑。
        floor_crash_list = np.where(floor_collisions >= 1)[0]

        # 执行这一行逻辑。
        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        # 保存或更新 `wall_crash_list` 的值。
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        # 执行这一行逻辑。
        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        # 保存或更新 `ceiling_crash_list` 的值。
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        # 返回当前函数的结果。
        return floor_crash_list, wall_crash_list, ceiling_crash_list

    # 定义函数 `obst_generation_given_density`。
    def obst_generation_given_density(self, grid_size=1.0):
        # 同时更新 `obst_area_length`, `obst_area_width` 等变量。
        obst_area_length, obst_area_width = int(self.obst_spawn_area[0]), int(self.obst_spawn_area[1])
        # 保存或更新 `num_room_grids` 的值。
        num_room_grids = obst_area_length * obst_area_width

        # 保存或更新 `cell_centers` 的值。
        cell_centers = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width,
                                        grid_size=grid_size)

        # 保存或更新 `room_map` 的值。
        room_map = [i for i in range(0, num_room_grids)]

        # 保存或更新 `obst_index` 的值。
        obst_index = np.random.choice(a=room_map, size=int(num_room_grids * self.obst_density), replace=False)

        # 保存或更新 `obst_pos_arr` 的值。
        obst_pos_arr = []
        # 0: No Obst, 1: Obst
        # 保存或更新 `obst_map` 的值。
        obst_map = np.zeros([obst_area_length, obst_area_width])
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for obst_id in obst_index:
            # 同时更新 `rid`, `cid` 等变量。
            rid, cid = obst_id // obst_area_width, obst_id - (obst_id // obst_area_width) * obst_area_width
            # 保存或更新 `obst_map[rid, cid]` 的值。
            obst_map[rid, cid] = 1
            # 保存或更新 `obst_item` 的值。
            obst_item = list(cell_centers[rid + int(obst_area_length / grid_size) * cid])
            # 调用 `append` 执行当前处理。
            obst_item.append(self.room_dims[2] / 2.)
            # 调用 `append` 执行当前处理。
            obst_pos_arr.append(obst_item)

        # 返回当前函数的结果。
        return obst_map, obst_pos_arr, cell_centers

    # 定义函数 `init_scene_multi`。
    def init_scene_multi(self):
        # 保存或更新 `models` 的值。
        models = tuple(e.dynamics.model for e in self.envs)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(len(self.quads_view_mode)):
            # 调用 `append` 执行当前处理。
            self.scenes.append(Quadrotor3DSceneMulti(
                models=models,
                w=600, h=480, resizable=True, viewpoint=self.quads_view_mode[i],
                room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
                vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=25, viz_trace_nth_step=1,
                num_obstacles=self.num_obstacles, scene_index=i
            ))

    # 定义函数 `reset`。
    def reset(self, obst_density=None, obst_size=None):
        # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
        obs, rewards, dones, infos = [], [], [], []

        # 根据条件决定是否进入当前分支。
        if obst_density:
            # 保存或更新 `obst_density` 的值。
            self.obst_density = obst_density
        # 根据条件决定是否进入当前分支。
        if obst_size:
            # 保存或更新 `obst_size` 的值。
            self.obst_size = obst_size

        # Scenario reset
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `obstacles` 的值。
            self.obstacles = MultiObstacles(obstacle_size=self.obst_size, quad_radius=self.quad_arm)
            # 同时更新 `obst_map`, `obst_pos_arr`, `cell_centers` 等变量。
            self.obst_map, obst_pos_arr, cell_centers = self.obst_generation_given_density()
            # 保存或更新 `scenario.reset(obst_map` 的值。
            self.scenario.reset(obst_map=self.obst_map, cell_centers=cell_centers)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `reset` 执行当前处理。
            self.scenario.reset()

        # Replay buffer
        # 根据条件决定是否进入当前分支。
        if self.use_replay_buffer and not self.activate_replay_buffer:
            # 调用 `append` 执行当前处理。
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            # 保存或更新 `activate_replay_buffer` 的值。
            self.activate_replay_buffer = self.can_drones_fly()
            # 保存或更新 `crashes_last_episode` 的值。
            self.crashes_last_episode = 0

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, e in enumerate(self.envs):
            # 保存或更新 `e.goal` 的值。
            e.goal = self.scenario.goals[i]
            # 根据条件决定是否进入当前分支。
            if self.scenario.spawn_points is None:
                # 保存或更新 `e.spawn_point` 的值。
                e.spawn_point = self.scenario.goals[i]
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `e.spawn_point` 的值。
                e.spawn_point = self.scenario.spawn_points[i]
            # 保存或更新 `e.rew_coeff` 的值。
            e.rew_coeff = self.rew_coeff

            # 保存或更新 `observation` 的值。
            observation = e.reset()
            # 调用 `append` 执行当前处理。
            obs.append(observation)
            # 保存或更新 `pos[i, :]` 的值。
            self.pos[i, :] = e.dynamics.pos

        # Neighbors
        # 根据条件决定是否进入当前分支。
        if self.num_use_neighbor_obs > 0:
            # 保存或更新 `obs` 的值。
            obs = self.add_neighborhood_obs(obs)

        # Obstacles
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `quads_pos` 的值。
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            # 保存或更新 `obs` 的值。
            obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, pos_arr=obst_pos_arr)
            # 保存或更新 `obst_quad_collisions_per_episode` 的值。
            self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0
            # 保存或更新 `prev_obst_quad_collisions` 的值。
            self.prev_obst_quad_collisions = []
            # 保存或更新 `distance_to_goal_3_5` 的值。
            self.distance_to_goal_3_5 = 0
            # 保存或更新 `distance_to_goal_5` 的值。
            self.distance_to_goal_5 = 0

        # Collision
        # # Collision: Neighbor
        # 保存或更新 `collisions_per_episode` 的值。
        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        # 保存或更新 `prev_drone_collisions` 的值。
        self.prev_drone_collisions = []

        # # Collision: Room
        # 保存或更新 `collisions_room_per_episode` 的值。
        self.collisions_room_per_episode = 0
        # 保存或更新 `collisions_floor_per_episode` 的值。
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        # 保存或更新 `prev_crashed_walls` 的值。
        self.prev_crashed_walls = []
        # 保存或更新 `prev_crashed_ceiling` 的值。
        self.prev_crashed_ceiling = []
        # 保存或更新 `prev_crashed_room` 的值。
        self.prev_crashed_room = []

        # Log
        # # Final Distance (1s / 3s / 5s)
        # 保存或更新 `distance_to_goal` 的值。
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        # 保存或更新 `agent_col_agent` 的值。
        self.agent_col_agent = np.ones(self.num_agents)
        # 保存或更新 `agent_col_obst` 的值。
        self.agent_col_obst = np.ones(self.num_agents)
        # 保存或更新 `reached_goal` 的值。
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Rendering
        # 根据条件决定是否进入当前分支。
        if self.quads_render:
            # 保存或更新 `reset_scene` 的值。
            self.reset_scene = True
            # 保存或更新 `quads_formation_size` 的值。
            self.quads_formation_size = self.scenario.formation_size
            # 保存或更新 `all_collisions` 的值。
            self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        # 返回当前函数的结果。
        return obs

    # 定义函数 `step`。
    def step(self, actions):
        # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
        obs, rewards, dones, infos = [], [], [], []

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, a in enumerate(actions):
            # 保存或更新 `envs[i].rew_coeff` 的值。
            self.envs[i].rew_coeff = self.rew_coeff

            # 同时更新 `observation`, `reward`, `done`, `info` 等变量。
            observation, reward, done, info = self.envs[i].step(a)
            # 调用 `append` 执行当前处理。
            obs.append(observation)
            # 调用 `append` 执行当前处理。
            rewards.append(reward)
            # 调用 `append` 执行当前处理。
            dones.append(done)
            # 调用 `append` 执行当前处理。
            infos.append(info)

            # 保存或更新 `pos[i, :]` 的值。
            self.pos[i, :] = self.envs[i].dynamics.pos

        # 1. Calculate collisions: 1) between drones 2) with obstacles 3) with room
        # 1) Collisions between drones
        # 同时更新 `drone_col_matrix`, `curr_drone_collisions`, `distance_matrix` 等变量。
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            # 保存或更新 `calculate_collision_matrix(positions` 的值。
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)

        # # Filter curr_drone_collisions
        # 保存或更新 `curr_drone_collisions` 的值。
        curr_drone_collisions = curr_drone_collisions.astype(int)
        # 保存或更新 `curr_drone_collisions` 的值。
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        # 保存或更新 `old_quad_collision` 的值。
        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        # 保存或更新 `new_quad_collision` 的值。
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        # 保存或更新 `last_step_unique_collisions` 的值。
        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)

        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        # 执行这一行逻辑。
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        # 保存或更新 `distance_matrix` 的值。
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        # 保存或更新 `collisions_curr_tick` 的值。
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        # 保存或更新 `collisions_per_episode` 的值。
        self.collisions_per_episode += collisions_curr_tick

        # 根据条件决定是否进入当前分支。
        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            # 保存或更新 `collisions_after_settle` 的值。
            self.collisions_after_settle += collisions_curr_tick
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for agent_id in self.last_step_unique_collisions:
                # 保存或更新 `agent_col_agent[agent_id]` 的值。
                self.agent_col_agent[agent_id] = 0
        # 根据条件决定是否进入当前分支。
        if collisions_curr_tick > 0 and self.envs[0].time_remain <= self.collisions_final_grace_period_steps:
            # 保存或更新 `collisions_final_5s` 的值。
            self.collisions_final_5s += collisions_curr_tick

        # # Aux: Neighbor Collisions
        # 保存或更新 `prev_drone_collisions` 的值。
        self.prev_drone_collisions = curr_drone_collisions

        # 2) Collisions with obstacles
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `rew_obst_quad_collisions_raw` 的值。
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            # 同时更新 `obst_quad_col_matrix`, `quad_obst_pair` 等变量。
            obst_quad_col_matrix, quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            # We assume drone can only collide with one obstacle at the same time.
            # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
            # 保存或更新 `curr_quad_col` 的值。
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            # 保存或更新 `collisions_obst_curr_tick` 的值。
            collisions_obst_curr_tick = len(self.curr_quad_col)
            # 保存或更新 `obst_quad_collisions_per_episode` 的值。
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            # 根据条件决定是否进入当前分支。
            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                # 保存或更新 `obst_quad_collisions_after_settle` 的值。
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for qid in self.curr_quad_col:
                    # 保存或更新 `q_rel_dist` 的值。
                    q_rel_dist = np.linalg.norm(obs[qid][0:3])
                    # 根据条件决定是否进入当前分支。
                    if q_rel_dist > 3.5:
                        # 保存或更新 `distance_to_goal_3_5` 的值。
                        self.distance_to_goal_3_5 += 1
                    # 根据条件决定是否进入当前分支。
                    if q_rel_dist > 5.0:
                        # 保存或更新 `distance_to_goal_5` 的值。
                        self.distance_to_goal_5 += 1
                    # Used for log agent_success
                    # 保存或更新 `agent_col_obst[qid]` 的值。
                    self.agent_col_obst[qid] = 0

            # # Aux: Obstacle Collisions
            # 保存或更新 `prev_obst_quad_collisions` 的值。
            self.prev_obst_quad_collisions = obst_quad_col_matrix

            # 根据条件决定是否进入当前分支。
            if len(obst_quad_col_matrix) > 0:
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                # 保存或更新 `rew_obst_quad_collisions_raw[curr_quad_col]` 的值。
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0

        # 3) Collisions with room
        # 同时更新 `floor_crash_list`, `wall_crash_list`, `ceiling_crash_list` 等变量。
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        # 保存或更新 `room_crash_list` 的值。
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        # 保存或更新 `room_crash_list` 的值。
        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)
        # # Aux: Room Collisions
        # 保存或更新 `prev_crashed_walls` 的值。
        self.prev_crashed_walls = wall_crash_list
        # 保存或更新 `prev_crashed_ceiling` 的值。
        self.prev_crashed_ceiling = ceiling_crash_list
        # 保存或更新 `prev_crashed_room` 的值。
        self.prev_crashed_room = room_crash_list

        # 2. Calculate rewards and infos for collision
        # 1) Between drones
        # 保存或更新 `rew_collisions_raw` 的值。
        rew_collisions_raw = np.zeros(self.num_agents)
        # 根据条件决定是否进入当前分支。
        if self.last_step_unique_collisions.any():
            # 保存或更新 `rew_collisions_raw[last_step_unique_collisions]` 的值。
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        # 保存或更新 `rew_collisions` 的值。
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        # 根据条件决定是否进入当前分支。
        if len(distance_matrix) > 0:
            # 保存或更新 `rew_proximity` 的值。
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `rew_proximity` 的值。
            rew_proximity = np.zeros(self.num_agents)

        # 2) With obstacles
        # 保存或更新 `rew_collisions_obst_quad` 的值。
        rew_collisions_obst_quad = np.zeros(self.num_agents)
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `rew_collisions_obst_quad` 的值。
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

        # 3) With room
        # # TODO: reward penalty
        # 根据条件决定是否进入当前分支。
        if self.envs[0].tick >= self.collisions_grace_period_steps:
            # 保存或更新 `collisions_room_per_episode` 的值。
            self.collisions_room_per_episode += len(room_crash_list)
            # 保存或更新 `collisions_floor_per_episode` 的值。
            self.collisions_floor_per_episode += len(floor_crash_list)
            # 保存或更新 `collisions_wall_per_episode` 的值。
            self.collisions_wall_per_episode += len(wall_crash_list)
            # 保存或更新 `collisions_ceiling_per_episode` 的值。
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        # Reward & Info
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(self.num_agents):
            # 保存或更新 `rewards[i]` 的值。
            rewards[i] += rew_collisions[i]
            # 保存或更新 `rewards[i]` 的值。
            rewards[i] += rew_proximity[i]

            # 保存或更新 `infos[i][rewards][rew_quadcol]` 的值。
            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            # 保存或更新 `infos[i][rewards][rew_proximity]` 的值。
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            # 保存或更新 `infos[i][rewards][rewraw_quadcol]` 的值。
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            # 根据条件决定是否进入当前分支。
            if self.use_obstacles:
                # 保存或更新 `rewards[i]` 的值。
                rewards[i] += rew_collisions_obst_quad[i]
                # 保存或更新 `infos[i][rewards][rew_quadcol_obstacle]` 的值。
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                # 保存或更新 `infos[i][rewards][rewraw_quadcol_obstacle]` 的值。
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]

            # 执行这一行逻辑。
            self.distance_to_goal[i].append(-infos[i]["rewards"]["rewraw_pos"])
            # 根据条件决定是否进入当前分支。
            if len(self.distance_to_goal[i]) >= 5 and \
                    # 调用 `mean` 执行当前处理。
                    np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
                    # 这里开始一个新的代码块。
                    and not self.reached_goal[i]:
                # 保存或更新 `reached_goal[i]` 的值。
                self.reached_goal[i] = True

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        # 保存或更新 `self_state_update_flag` 的值。
        self_state_update_flag = False

        # # 1) aerodynamics
        # 根据条件决定是否进入当前分支。
        if self.use_downwash:
            # 保存或更新 `envs_dynamics` 的值。
            envs_dynamics = [env.dynamics for env in self.envs]
            # 保存或更新 `applied_downwash_list` 的值。
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            # 执行这一行逻辑。
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            # 根据条件决定是否进入当前分支。
            if len(downwash_agents_list) > 0:
                # 保存或更新 `self_state_update_flag` 的值。
                self_state_update_flag = True

        # # 2) Drones
        # 根据条件决定是否进入当前分支。
        if self.apply_collision_force:
            # 根据条件决定是否进入当前分支。
            if len(new_quad_collision) > 0:
                # 保存或更新 `self_state_update_flag` 的值。
                self_state_update_flag = True
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for val in new_quad_collision:
                    # 同时更新 `dyn1`, `dyn2` 等变量。
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    # 同时更新 `dyn1.vel`, `dyn1.omega`, `dyn2.vel`, `dyn2.omega` 等变量。
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

            # # 3) Obstacles
            # 根据条件决定是否进入当前分支。
            if self.use_obstacles:
                # 根据条件决定是否进入当前分支。
                if len(self.curr_quad_col) > 0:
                    # 保存或更新 `self_state_update_flag` 的值。
                    self_state_update_flag = True
                    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                    for val in self.curr_quad_col:
                        # 保存或更新 `obstacle_id` 的值。
                        obstacle_id = quad_obst_pair[int(val)]
                        # 保存或更新 `obstacle_pos` 的值。
                        obstacle_pos = self.obstacles.pos_arr[int(obstacle_id)]
                        # 保存或更新 `perform_collision_with_obstacle(drone_dyn` 的值。
                        perform_collision_with_obstacle(drone_dyn=self.envs[int(val)].dynamics,
                                                        obstacle_pos=obstacle_pos,
                                                        obstacle_size=self.obst_size)

            # # 4) Room
            # 根据条件决定是否进入当前分支。
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                # 保存或更新 `self_state_update_flag` 的值。
                self_state_update_flag = True

                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for val in wall_crash_list:
                    # 保存或更新 `perform_collision_with_wall(drone_dyn` 的值。
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for val in ceiling_crash_list:
                    # 保存或更新 `perform_collision_with_ceiling(drone_dyn` 的值。
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 4. Run the scenario passed to self.quads_mode
        # 调用 `step` 执行当前处理。
        self.scenario.step()

        # 5. Collect final observations
        # Collect positions after physical interaction
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(self.num_agents):
            # 保存或更新 `pos[i, :]` 的值。
            self.pos[i, :] = self.envs[i].dynamics.pos
            # 保存或更新 `vel[i, :]` 的值。
            self.vel[i, :] = self.envs[i].dynamics.vel

        # 根据条件决定是否进入当前分支。
        if self_state_update_flag:
            # 保存或更新 `obs` 的值。
            obs = [e.state_vector(e) for e in self.envs]

        # Concatenate observations of neighbor drones
        # 根据条件决定是否进入当前分支。
        if self.num_use_neighbor_obs > 0:
            # 保存或更新 `obs` 的值。
            obs = self.add_neighborhood_obs(obs)

        # Concatenate obstacle observations
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `obs` 的值。
            obs = self.obstacles.step(obs=obs, quads_pos=self.pos)

        # 6. Update info for replay buffer
        # Once agent learns how to take off, activate the replay buffer
        # 根据条件决定是否进入当前分支。
        if self.use_replay_buffer and not self.activate_replay_buffer:
            # 保存或更新 `crashes_last_episode` 的值。
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Rendering
        # 根据条件决定是否进入当前分支。
        if self.quads_render:
            # Collisions with room
            # 保存或更新 `ground_collisions` 的值。
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            # 根据条件决定是否进入当前分支。
            if self.use_obstacles:
                # 保存或更新 `obst_coll` 的值。
                obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `obst_coll` 的值。
                obst_coll = [0.0 for _ in range(self.num_agents)]
            # 保存或更新 `all_collisions` 的值。
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                                   'obstacle': obst_coll}

        # 7. DONES
        # 根据条件决定是否进入当前分支。
        if any(dones):
            # 保存或更新 `scenario_name` 的值。
            scenario_name = self.scenario.name()[9:]
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(infos)):
                # 根据条件决定是否进入当前分支。
                if self.saved_in_replay_buffer:
                    # 保存或更新 `infos[i][episode_extra_stats]` 的值。
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                        'num_collisions_obst_replay': self.obst_quad_collisions_per_episode,
                    }
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # 保存或更新 `distance_to_goal` 的值。
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    # 保存或更新 `reached_goal` 的值。
                    self.reached_goal = np.array(self.reached_goal)
                    # 保存或更新 `infos[i][episode_extra_stats]` 的值。
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_with_room': self.collisions_room_per_episode,
                        'num_collisions_with_floor': self.collisions_floor_per_episode,
                        'num_collisions_with_wall': self.collisions_wall_per_episode,
                        'num_collisions_with_ceiling': self.collisions_ceiling_per_episode,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'{scenario_name}/num_collisions': self.collisions_after_settle,

                        'num_collisions_final_5_s': self.collisions_final_5s,
                        f'{scenario_name}/num_collisions_final_5_s': self.collisions_final_5s,

                        'distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        'distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        'distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),

                        f'{scenario_name}/distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),
                    }

                    # 根据条件决定是否进入当前分支。
                    if self.use_obstacles:
                        # 保存或更新 `infos[i][episode_extra_stats][num_collisions_obst_quad]` 的值。
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            # 执行这一行逻辑。
                            self.obst_quad_collisions_per_episode
                        # 保存或更新 `infos[i][episode_extra_stats][num_collisions_obst_quad_after_settle]` 的值。
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = \
                            # 执行这一行逻辑。
                            self.obst_quad_collisions_after_settle
                        # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/num_collisions_obst]` 的值。
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = \
                            # 执行这一行逻辑。
                            self.obst_quad_collisions_per_episode

                        # 保存或更新 `infos[i][episode_extra_stats][num_collisions_obst_quad_3_5]` 的值。
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_3_5'] = \
                            # 执行这一行逻辑。
                            self.distance_to_goal_3_5
                        # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/num_collisions_obst_quad_3_5]` 的值。
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_3_5'] = \
                            # 执行这一行逻辑。
                            self.distance_to_goal_3_5

                        # 保存或更新 `infos[i][episode_extra_stats][num_collisions_obst_quad_5]` 的值。
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_5'] = \
                            # 执行这一行逻辑。
                            self.distance_to_goal_5
                        # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/num_collisions_obst_quad_5]` 的值。
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_5'] = \
                            # 执行这一行逻辑。
                            self.distance_to_goal_5

            # 根据条件决定是否进入当前分支。
            if not self.saved_in_replay_buffer:
                # agent_success_rate: base_success_rate, based on per agent
                # 0: collision; 1: no collision
                # 保存或更新 `agent_col_flag_list` 的值。
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                # 保存或更新 `agent_success_flag_list` 的值。
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                # 保存或更新 `agent_success_ratio` 的值。
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                # agent_deadlock_rate
                # Doesn't approach to the goal while no collisions with other objects
                # 保存或更新 `agent_deadlock_list` 的值。
                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                # 保存或更新 `agent_deadlock_ratio` 的值。
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                # agent_col_rate
                # Collide with other drones and obstacles
                # 保存或更新 `agent_col_ratio` 的值。
                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

                # agent_neighbor_col_rate
                # 保存或更新 `agent_neighbor_col_ratio` 的值。
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                # agent_obst_col_rate
                # 保存或更新 `agent_obst_col_ratio` 的值。
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for i in range(len(infos)):
                    # agent_success_rate
                    # 保存或更新 `infos[i][episode_extra_stats][metric/agent_success_rate]` 的值。
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/agent_success_rate]` 的值。
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    # agent_deadlock_rate
                    # 保存或更新 `infos[i][episode_extra_stats][metric/agent_deadlock_rate]` 的值。
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/agent_deadlock_rate]` 的值。
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    # 保存或更新 `infos[i][episode_extra_stats][metric/agent_col_rate]` 的值。
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/agent_col_rate]` 的值。
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    # agent_neighbor_col_rate
                    # 保存或更新 `infos[i][episode_extra_stats][metric/agent_neighbor_col_rate]` 的值。
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/agent_neighbor_col_rate]` 的值。
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    # 保存或更新 `infos[i][episode_extra_stats][metric/agent_obst_col_rate]` 的值。
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    # 保存或更新 `infos[i][episode_extra_stats][f{scenario_name}/agent_obst_col_rate]` 的值。
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio

            # 保存或更新 `obs` 的值。
            obs = self.reset()
            # terminate the episode for all "sub-envs"
            # 保存或更新 `dones` 的值。
            dones = [True] * len(dones)

        # 返回当前函数的结果。
        return obs, rewards, dones, infos

    # 定义函数 `render`。
    def render(self, verbose=False):
        # 保存或更新 `models` 的值。
        models = tuple(e.dynamics.model for e in self.envs)

        # 根据条件决定是否进入当前分支。
        if len(self.scenes) == 0:
            # 调用 `init_scene_multi` 执行当前处理。
            self.init_scene_multi()

        # 根据条件决定是否进入当前分支。
        if self.reset_scene:
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(self.scenes)):
                # 执行这一行逻辑。
                self.scenes[i].update_models(models)
                # 保存或更新 `scenes[i].formation_size` 的值。
                self.scenes[i].formation_size = self.quads_formation_size
                # 执行这一行逻辑。
                self.scenes[i].update_env(self.room_dims)

                # 执行这一行逻辑。
                self.scenes[i].reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles,
                                     self.all_collisions)

            # 保存或更新 `reset_scene` 的值。
            self.reset_scene = False

        # 根据条件决定是否进入当前分支。
        if self.quads_mode == "mix":
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(self.scenes)):
                # 保存或更新 `scenes[i].formation_size` 的值。
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(self.scenes)):
                # 保存或更新 `scenes[i].formation_size` 的值。
                self.scenes[i].formation_size = self.scenario.formation_size
        # 保存或更新 `frames_since_last_render` 的值。
        self.frames_since_last_render += 1

        # 根据条件决定是否进入当前分支。
        if self.render_skip_frames > 0:
            # 保存或更新 `render_skip_frames` 的值。
            self.render_skip_frames -= 1
            # 返回当前函数的结果。
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        # 根据条件决定是否进入当前分支。
        if self.simulation_start_time > 0:
            # 保存或更新 `simulation_time` 的值。
            simulation_time = time.time() - self.simulation_start_time
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `simulation_time` 的值。
            simulation_time = 0

        # 保存或更新 `realtime_control_period` 的值。
        realtime_control_period = 1 / self.control_freq

        # 保存或更新 `render_start` 的值。
        render_start = time.time()
        # 保存或更新 `goals` 的值。
        goals = tuple(e.goal for e in self.envs)
        # 保存或更新 `frames` 的值。
        frames = []
        # 保存或更新 `first_spawn` 的值。
        first_spawn = None
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(len(self.scenes)):
            # 同时更新 `frame`, `first_spawn` 等变量。
            frame, first_spawn = self.scenes[i].render_chase(all_dynamics=self.all_dynamics(), goals=goals,
                                                             collisions=self.all_collisions,
                                                             mode=self.render_mode, obstacles=self.obstacles,
                                                             first_spawn=first_spawn)
            # 调用 `append` 执行当前处理。
            frames.append(frame)
        # Update the formation size of the scenario
        # 根据条件决定是否进入当前分支。
        if self.quads_mode == "mix":
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(self.scenes)):
                # 调用 `update_formation_size` 执行当前处理。
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(self.scenes)):
                # 调用 `update_formation_size` 执行当前处理。
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        # 保存或更新 `render_time` 的值。
        render_time = time.time() - render_start

        # 保存或更新 `desired_time_between_frames` 的值。
        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        # 保存或更新 `time_to_sleep` 的值。
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        # 根据条件决定是否进入当前分支。
        if self.render_mode == "human" and time_to_sleep > 0:
            # 调用 `sleep` 执行当前处理。
            time.sleep(time_to_sleep)

        # 根据条件决定是否进入当前分支。
        if simulation_time + render_time > desired_time_between_frames:
            # 保存或更新 `render_every_nth_frame` 的值。
            self.render_every_nth_frame += 1
            # 根据条件决定是否进入当前分支。
            if verbose:
                # 调用 `print` 执行当前处理。
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                # 调用 `print` 执行当前处理。
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        # 当上一分支不满足时，继续判断新的条件。
        elif simulation_time + render_time < realtime_control_period * (
                # 这里开始一个新的代码块。
                self.frames_since_last_render - 1) / self.render_speed:
            # 保存或更新 `render_every_nth_frame` 的值。
            self.render_every_nth_frame -= 1
            # 根据条件决定是否进入当前分支。
            if verbose:
                # 调用 `print` 执行当前处理。
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        # 根据条件决定是否进入当前分支。
        if self.render_every_nth_frame > 5:
            # 保存或更新 `render_every_nth_frame` 的值。
            self.render_every_nth_frame = 5
            # 根据条件决定是否进入当前分支。
            if self.envs[0].tick % 20 == 0:
                # 调用 `print` 执行当前处理。
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        # 保存或更新 `render_skip_frames` 的值。
        self.render_skip_frames = self.render_every_nth_frame - 1
        # 保存或更新 `frames_since_last_render` 的值。
        self.frames_since_last_render = 0

        # 保存或更新 `simulation_start_time` 的值。
        self.simulation_start_time = time.time()

        # 根据条件决定是否进入当前分支。
        if self.render_mode == "rgb_array":
            # 返回当前函数的结果。
            return frame

    # 定义函数 `__deepcopy__`。
    def __deepcopy__(self, memo):
        # 下面的文档字符串用于说明当前模块或代码块。
        """OpenGL scene can't be copied naively."""

        # 保存或更新 `cls` 的值。
        cls = self.__class__
        # 保存或更新 `copied_env` 的值。
        copied_env = cls.__new__(cls)
        # 保存或更新 `memo[id(self)]` 的值。
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        # 保存或更新 `skip_copying` 的值。
        skip_copying = {"scene", "reward_shaping_interface"}

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for k, v in self.__dict__.items():
            # 根据条件决定是否进入当前分支。
            if k not in skip_copying:
                # 调用 `setattr` 执行当前处理。
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        # 保存或更新 `copied_env.scene` 的值。
        copied_env.scene = None

        # 返回当前函数的结果。
        return copied_env
