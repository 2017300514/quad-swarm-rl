# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_multi.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是多机四旋翼环境的主循环实现。
# 上游输入包括每架无人机的动作、场景生成器、障碍物配置、奖励系数和房间尺寸；
# 下游输出是策略网络消费的观测、逐 agent 奖励、done 标志，以及训练日志里使用的碰撞/到达目标统计。
# 它的职责是把多个 `QuadrotorSingle` 串成一个统一环境，并在每个 step 中依次完成：
# 单机推进、邻居关系更新、障碍物交互、碰撞判定、额外奖励结算、物理碰撞修正、场景推进和日志汇总。

import copy
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario


class QuadrotorEnvMulti(gym.Env):
    # 这个类维护多机训练所需的公共状态：
    # 所有单机环境实例、共享奖励系数、相对位置/速度缓存、场景对象、障碍物对象、
    # 机间/障碍/房间碰撞统计、replay 激活条件以及渲染场景。
    # 单机层只知道“我怎么飞”，这里则负责“多架无人机在同一场景里怎样相互影响”。
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
                 ):
        super().__init__()

        self.num_agents = num_agents

        # `obs_self_size` 是单机自观测部分的维度。
        # 多机环境需要先知道它，后面才能从完整 observation space 中切出“邻居观测应该落在哪一段”。
        obs_self_size = QUADS_OBS_REPR[obs_repr]

        # `neighbor_visible_num=-1` 表示策略对每架无人机都看到全部其他 agent；
        # 否则只保留有限个邻居槽位，由后面的 `neighborhood_indices()` 负责挑选具体是谁。
        if neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num

        # 这个标志告诉 Sample Factory：即便 `num_agents=1`，这个环境也按多智能体接口处理。
        self.is_multiagent = True
        self.room_dims = room_dims
        self.quads_view_mode = quads_view_mode

        # 多机环境的底层仍然是 `num_agents` 个 `QuadrotorSingle`。
        # 它们共享同一类动力学和任务时长，但各自维护自己的物理状态、目标和控制器。
        self.envs = []
        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr, ep_time=ep_time, room_dims=room_dims,
                use_numba=use_numba,
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                use_obstacles=use_obstacles,
            )
            self.envs.append(e)

        # 观测和动作空间直接复用单机定义。
        # 多机环境并不是重新定义新动作，而是把多个单机输入/输出按 agent 维度并排组织起来。
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # 这一组缓存每步都会从单机环境同步一次。
        # 它们是邻居筛选、碰撞矩阵和障碍物交互的主数据源，因此必须保存在连续数组里，而不是每次都临时从对象图上取。
        self.quad_arm = self.envs[0].dynamics.arm
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])
        self.vel = np.zeros([self.num_agents, 3])
        self.omega = np.zeros([self.num_agents, 3])
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))

        # 默认奖励表先给出多机任务的一组基线系数，再由上游配置覆盖。
        # 这里的 `quadcol_bin` / `quadcol_bin_smooth_max` / `quadcol_bin_obst`
        # 分别对应机间硬碰撞、连续接近惩罚和障碍碰撞惩罚。
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        # 这里显式检查键名合法性，防止命令行或配置里拼错奖励项时静默失效。
        orig_keys = list(rew_coeff_orig.keys())
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # 邻居观测的裁剪边界来自单机 observation space 中预留的那一段邻居槽位。
        # 后面 `extend_obs_space()` 会用这些边界把相对位置/速度观测压回网络期望范围。
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]

        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        self.use_obstacles = use_obstacles
        self.obstacles = None
        self.num_obstacles = 0
        if self.use_obstacles:
            # 这些状态服务于障碍物生成、障碍碰撞统计和论文里的障碍相关评估指标。
            self.prev_obst_quad_collisions = []
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_after_settle = 0
            self.curr_quad_col = []
            self.obst_density = obst_density
            self.obst_spawn_area = obst_spawn_area
            self.num_obstacles = int(obst_density * obst_spawn_area[0] * obst_spawn_area[1])
            self.obst_map = None
            self.obst_size = obst_size

            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0

        # 场景对象定义“目标怎么摆、出生点怎么摆、每步目标是否动态变化”。
        # 它是训练样本分布的总控开关。
        self.quads_mode = quads_mode
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # 这些字段记录整个 episode 的碰撞统计，而不是单步状态。
        self.collisions_per_episode = 0
        self.collisions_after_settle = 0
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        self.collisions_grace_period_seconds = 1.5
        self.prev_drone_collisions = []

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        self.collisions_final_5s = 0

        # 碰撞阈值按机臂长度缩放，是为了让“多近算碰撞/危险接近”跟机体真实尺寸挂钩。
        self.collision_threshold = collision_hitbox_radius * self.quad_arm
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm

        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_wall_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # replay 机制会先观察最近若干 episode 是否已经学会基本起飞，再决定是否允许从碰撞片段重开。
        self.use_replay_buffer = use_replay_buffer
        self.activate_replay_buffer = False
        self.saved_in_replay_buffer = False
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        self.use_numba = use_numba
        self.use_downwash = use_downwash

        # 渲染状态和训练逻辑本身分开保存，避免正常训练路径被可视化细节污染。
        self.render_mode = render_mode
        self.quads_render = quads_render
        self.scenes = []
        if self.quads_render:
            self.reset_scene = False
            self.simulation_start_time = 0
            self.frames_since_last_render = self.render_skip_frames = 0
            self.render_every_nth_frame = 1
            self.render_speed = 1.0
            self.quads_formation_size = 2.0
            self.all_collisions = {}

        # `distance_to_goal` 保存每个 agent 每步到目标的原始距离轨迹。
        # 这些轨迹不会参与即时奖励，但会在 episode 结束时转成“最后 1s/3s/5s 的平均距离”等指标。
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.reached_goal = [False for _ in range(len(self.envs))]

        # 这两个向量用于后面的成功率指标：
        # 1 表示当前 agent 还没发生对应类型碰撞，0 表示本局已发生。
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)

        self.apply_collision_force = True

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # 缺省情况下，把除自己外的所有无人机都视作邻居候选集合。
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.pos[i]
        cur_vel = self.vel[i]
        pos_neighbor = np.stack([self.pos[j] for j in indices])
        vel_neighbor = np.stack([self.vel[j] for j in indices])
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        # 单个 agent 的邻居观测由“相对位置 + 相对速度”拼接而成。
        # 这个编码直接决定策略如何判断邻居是在接近、远离还是处于碰撞风险中。
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
        # 这里把所有 agent 的邻居相对状态拼成与自观测并列的扩展观测。
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # 裁剪邻居观测的目的是保证极端距离和速度不会让输入尺度失控。
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        indices = np.array(indices)

        if self.num_use_neighbor_obs == self.num_agents - 1:
            return indices
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.num_agents):
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # 这里使用的不只是纯距离，而是“距离 + 径向相对速度”的混合指标。
                # 这样不仅靠得近的邻居更重要，正在快速朝自己逼近的邻居也会被优先暴露给策略。
                new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        return obs_ext

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        # replay buffer 只有在策略学会基本飞行后才启用，
        # 否则早期随机策略产生的失败片段没有教学价值，只会污染 buffer。
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def calculate_room_collision(self):
        # 房间碰撞单独从各单机动力学标志中读取，而不是走与机间碰撞相同的距离矩阵逻辑。
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    def obst_generation_given_density(self, grid_size=1.0):
        # 障碍物不是连续空间完全随机撒点，而是先把可用生成区域离散成网格，再按密度抽取哪些格子放障碍。
        # 这样既能稳定控制密度，也能给局部 SDF / octomap 观测提供更规整的几何结构。
        obst_area_length, obst_area_width = int(self.obst_spawn_area[0]), int(self.obst_spawn_area[1])
        num_room_grids = obst_area_length * obst_area_width

        cell_centers = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width,
                                        grid_size=grid_size)

        room_map = [i for i in range(0, num_room_grids)]

        obst_index = np.random.choice(a=room_map, size=int(num_room_grids * self.obst_density), replace=False)

        obst_pos_arr = []
        obst_map = np.zeros([obst_area_length, obst_area_width])
        for obst_id in obst_index:
            rid, cid = obst_id // obst_area_width, obst_id - (obst_id // obst_area_width) * obst_area_width
            obst_map[rid, cid] = 1
            obst_item = list(cell_centers[rid + int(obst_area_length / grid_size) * cid])
            obst_item.append(self.room_dims[2] / 2.)
            obst_pos_arr.append(obst_item)

        return obst_map, obst_pos_arr, cell_centers

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        for i in range(len(self.quads_view_mode)):
            self.scenes.append(Quadrotor3DSceneMulti(
                models=models,
                w=600, h=480, resizable=True, viewpoint=self.quads_view_mode[i],
                room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
                vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=25, viz_trace_nth_step=1,
                num_obstacles=self.num_obstacles, scene_index=i
            ))

    def reset(self, obst_density=None, obst_size=None):
        # reset 的工作不是简单“清零”。
        # 它要重新布置场景、重新生成障碍物、把目标和出生点写回每个单机环境，
        # 再重新拼出第一帧完整观测。
        obs, rewards, dones, infos = [], [], [], []

        if obst_density:
            self.obst_density = obst_density
        if obst_size:
            self.obst_size = obst_size

        if self.use_obstacles:
            self.obstacles = MultiObstacles(obstacle_size=self.obst_size, quad_radius=self.quad_arm)
            self.obst_map, obst_pos_arr, cell_centers = self.obst_generation_given_density()
            self.scenario.reset(obst_map=self.obst_map, cell_centers=cell_centers)
        else:
            self.scenario.reset()

        # replay 激活逻辑是在 episode 边界评估的：
        # 只有当最近一段时间里墙/地碰撞已经较少，才允许把失败片段回收进 buffer 继续训练。
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = self.can_drones_fly()
            self.crashes_last_episode = 0

        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            if self.scenario.spawn_points is None:
                e.spawn_point = self.scenario.goals[i]
            else:
                e.spawn_point = self.scenario.spawn_points[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
            self.pos[i, :] = e.dynamics.pos

        # 第一帧自观测构建完之后，再追加邻居观测和障碍观测。
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        if self.use_obstacles:
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, pos_arr=obst_pos_arr)
            self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0
            self.prev_obst_quad_collisions = []
            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0

        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []

        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)
        self.reached_goal = [False for _ in range(len(self.envs))]

        if self.quads_render:
            self.reset_scene = True
            self.quads_formation_size = self.scenario.formation_size
            self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        return obs

    def step(self, actions):
        # 每个环境步的主线：
        # 1. 先让所有单机环境各自推进一步
        # 2. 再在多机层统一计算邻居/障碍/房间碰撞和附加奖励
        # 3. 之后施加碰撞后的物理修正与场景更新
        # 4. 最后重组新的观测并在 episode 结束时汇总日志
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        # 1) 机间碰撞首先由位置矩阵推导。
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)

        # 去掉内部占位值，只保留真实碰撞对。
        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        # `last_step_unique_collisions` 只保留当前 step 新发生的碰撞，
        # 避免两架已经贴在一起的无人机在多个连续 step 中被重复计数。
        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)

        # 仅保留进入平滑惩罚半径的机体对，后面用来计算 proximity penalty。
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        # 起飞早期会有出生点重叠或姿态恢复导致的瞬时接触，因此用 grace period 把这些碰撞从正式指标里剔除。
        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0
        if collisions_curr_tick > 0 and self.envs[0].time_remain <= self.collisions_final_grace_period_steps:
            self.collisions_final_5s += collisions_curr_tick

        self.prev_drone_collisions = curr_drone_collisions

        # 2) 障碍碰撞从障碍物模块读取。
        if self.use_obstacles:
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            obst_quad_col_matrix, quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            collisions_obst_curr_tick = len(self.curr_quad_col)
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                for qid in self.curr_quad_col:
                    q_rel_dist = np.linalg.norm(obs[qid][0:3])
                    if q_rel_dist > 3.5:
                        self.distance_to_goal_3_5 += 1
                    if q_rel_dist > 5.0:
                        self.distance_to_goal_5 += 1
                    self.agent_col_obst[qid] = 0

            self.prev_obst_quad_collisions = obst_quad_col_matrix

            if len(obst_quad_col_matrix) > 0:
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0

        # 3) 房间碰撞单独统计。
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)
        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_list

        # 下面三组 reward 都不是单机 `_step()` 里算出来的，而是多机层额外叠加的安全项。
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.num_agents)

        rew_collisions_obst_quad = np.zeros(self.num_agents)
        if self.use_obstacles:
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

        if self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        # 把多机层额外奖励写回每个 agent 的奖励总和和 `info['rewards']`。
        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            rewards[i] += rew_proximity[i]

            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]

            # 用原始位置代价恢复“离目标还有多远”的轨迹，后面做最后若干秒的接近指标。
            self.distance_to_goal[i].append(-infos[i]["rewards"]["rewraw_pos"])
            if len(self.distance_to_goal[i]) >= 5 and \
                    np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
                    and not self.reached_goal[i]:
                self.reached_goal[i] = True

        # 只有发生真实物理相互作用时，才需要重新从动力学状态回写观测。
        self_state_update_flag = False

        # 1) 下洗气流是额外气动耦合，不属于传统刚体碰撞，但同样会改变速度状态。
        if self.use_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        if self.apply_collision_force:
            # 2) 机间碰撞会直接改写两架无人机的线速度和角速度。
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

            # 3) 障碍碰撞会把对应无人机的动力学状态按障碍法向方向做反应。
            if self.use_obstacles:
                if len(self.curr_quad_col) > 0:
                    self_state_update_flag = True
                    for val in self.curr_quad_col:
                        obstacle_id = quad_obst_pair[int(val)]
                        obstacle_pos = self.obstacles.pos_arr[int(obstacle_id)]
                        perform_collision_with_obstacle(drone_dyn=self.envs[int(val)].dynamics,
                                                        obstacle_pos=obstacle_pos,
                                                        obstacle_size=self.obst_size)

            # 4) 房间墙壁和天花板碰撞也通过修改底层动力学速度来体现弹回。
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                self_state_update_flag = True

                for val in wall_crash_list:
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                for val in ceiling_crash_list:
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 场景推进放在所有物理修正之后，这样下一时刻目标或队形变化总是基于已经更新后的机体状态。
        self.scenario.step()

        # 重新同步位置和速度缓存，为下一轮邻居观测、障碍观测和碰撞检测做准备。
        for i in range(self.num_agents):
            self.pos[i, :] = self.envs[i].dynamics.pos
            self.vel[i, :] = self.envs[i].dynamics.vel

        if self_state_update_flag:
            obs = [e.state_vector(e) for e in self.envs]

        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        if self.use_obstacles:
            obs = self.obstacles.step(obs=obs, quads_pos=self.pos)

        # replay 激活前先观察训练期的 crash 信号，作为“是否已经学会起飞”的代理指标。
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        if self.quads_render:
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            if self.use_obstacles:
                obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
            else:
                obst_coll = [0.0 for _ in range(self.num_agents)]
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                                   'obstacle': obst_coll}

        # 只要任意 agent 结束，本实现就把整局作为一个共同结束的 multi-agent episode。
        if any(dones):
            scenario_name = self.scenario.name()[9:]
            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                        'num_collisions_obst_replay': self.obst_quad_collisions_per_episode,
                    }
                else:
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    self.reached_goal = np.array(self.reached_goal)
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

                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = \
                            self.obst_quad_collisions_after_settle
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = \
                            self.obst_quad_collisions_per_episode

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5

            if not self.saved_in_replay_buffer:
                # 这些比例指标都不是即时 reward，而是 episode 结束后的评估口径。
                # 它们用于回答：本局是成功到达、无碰撞但卡住，还是与邻居/障碍发生了失败交互。
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                for i in range(len(infos)):
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio

            # 这里采用“自动重置并把 dones 全部拉高”的接口风格。
            # 也就是说，调用者拿到 done 时，`obs` 已经是下一局首帧观测。
            obs = self.reset()
            dones = [True] * len(dones)

        return obs, rewards, dones, infos

    def render(self, verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if len(self.scenes) == 0:
            self.init_scene_multi()

        if self.reset_scene:
            for i in range(len(self.scenes)):
                self.scenes[i].update_models(models)
                self.scenes[i].formation_size = self.quads_formation_size
                self.scenes[i].update_env(self.room_dims)

                self.scenes[i].reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles,
                                     self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        else:
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        frames = []
        first_spawn = None
        for i in range(len(self.scenes)):
            frame, first_spawn = self.scenes[i].render_chase(all_dynamics=self.all_dynamics(), goals=goals,
                                                             collisions=self.all_collisions,
                                                             mode=self.render_mode, obstacles=self.obstacles,
                                                             first_spawn=first_spawn)
            frames.append(frame)
        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        else:
            for i in range(len(self.scenes)):
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # 渲染线程会尝试跟真实控制频率对齐，如果渲染跟不上就自动降帧。
        if self.render_mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if self.render_mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # PBT 等场景可能需要深拷贝环境，但 OpenGL scene 不能被普通 deepcopy。
        # 这里故意跳过这些对象，保留环境逻辑状态，同时避免为每个副本都新开渲染窗口。
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        copied_env.scene = None

        return copied_env
