import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

# 这是障碍任务里最常见的随机起终点场景。
# reset 时不再围绕一个统一 formation_center 生成编队目标，而是直接在障碍地图自由格中为每架无人机采样
# 独立的出生点和终点；外层环境随后用 `spawn_points` 布置出生位置，用 `goals` 布置目标位置。


class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 比 `o_base` 默认阈值更严格，要求更靠近目标才算真正到达。
        self.approch_goal_metric = 0.5

    def update_formation_size(self, new_formation_size):
        # 该场景的任务结构由障碍图中的离散起终点主导，不依赖运行中编队伸缩。
        pass

    def step(self):
        tick = self.envs[0].tick

        if tick <= self.duration_step:
            return

        # 超过时间窗后，把逐 agent 的 `end_point` 再写回 `env.goal`。
        # 这里不调用 `generate_goals`，因为终点本身就是为每架机分别采好的。
        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]

        return

    def reset(self, obst_map, cell_centers):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        # 收集所有自由格，作为后续采样池。
        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # 这段逐个 append 的结果会被后面的批量采样覆盖；注释副本保留源码语义，不改实现。
        self.start_point = []
        self.end_point = []
        for i in range(self.num_agents):
            self.start_point.append(self.generate_pos_obst_map())
            self.end_point.append(self.generate_pos_obst_map())

        # 实际生效的是这里：一次性为全部 agent 采样互不重复的起点和终点。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)

        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        # `formation_center` 在这里主要只是保留统一接口；真正任务几何由逐 agent 起终点决定。
        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
