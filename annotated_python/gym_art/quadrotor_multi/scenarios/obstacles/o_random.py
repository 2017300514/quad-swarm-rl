# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/obstacles/o_random.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import copy

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


# 定义类 `Scenario_o_random`。
class Scenario_o_random(Scenario_o_base):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 0.5

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `step`。
    def step(self):
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick

        # 根据条件决定是否进入当前分支。
        if tick <= self.duration_step:
            # 返回当前函数的结果。
            return

        # 保存或更新 `duration_step` 的值。
        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, env in enumerate(self.envs):
            # 保存或更新 `env.goal` 的值。
            env.goal = self.end_point[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map, cell_centers):
        # 保存或更新 `obstacle_map` 的值。
        self.obstacle_map = obst_map
        # 保存或更新 `cell_centers` 的值。
        self.cell_centers = cell_centers
        # 根据条件决定是否进入当前分支。
        if obst_map is None:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError

        # 执行这一行逻辑。
        obst_map_locs = np.where(self.obstacle_map == 0)
        # 保存或更新 `free_space` 的值。
        self.free_space = list(zip(*obst_map_locs))

        # 保存或更新 `start_point` 的值。
        self.start_point = []
        # 保存或更新 `end_point` 的值。
        self.end_point = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(self.num_agents):
            # 调用 `append` 执行当前处理。
            self.start_point.append(self.generate_pos_obst_map())
            # 调用 `append` 执行当前处理。
            self.end_point.append(self.generate_pos_obst_map())

        # 保存或更新 `start_point` 的值。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # 保存或更新 `end_point` 的值。
        self.end_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # self.start_point = self.generate_pos_obst_map_2(self.num_agents)
        # self.end_point = self.generate_pos_obst_map_2(self.num_agents)

        # 保存或更新 `duration_step` 的值。
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # 保存或更新 `formation_center` 的值。
        self.formation_center = np.array((0., 0., 2.))
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = copy.deepcopy(self.start_point)
        # 保存或更新 `goals` 的值。
        self.goals = copy.deepcopy(self.end_point)
