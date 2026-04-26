# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import copy

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


# 定义类 `Scenario_o_swap_goals`。
class Scenario_o_swap_goals(Scenario_o_base):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        # 保存或更新 `duration_time` 的值。
        duration_time = 6.0
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    # 定义函数 `update_goals`。
    def update_goals(self):
        # 调用 `shuffle` 执行当前处理。
        np.random.shuffle(self.goals)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for env, goal in zip(self.envs, self.goals):
            # 保存或更新 `env.goal` 的值。
            env.goal = goal

    # 定义函数 `step`。
    def step(self):
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        # 根据条件决定是否进入当前分支。
        if tick % self.control_step_for_sec == 0 and tick > 0:
            # 调用 `update_goals` 执行当前处理。
            self.update_goals()

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map=None, cell_centers=None):
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

        # Update duration time
        # 保存或更新 `duration_time` 的值。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # 保存或更新 `start_point` 的值。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = copy.deepcopy(self.start_point)

        # 保存或更新 `formation_center` 的值。
        self.formation_center = self.max_square_area_center()

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        # 调用 `shuffle` 执行当前处理。
        np.random.shuffle(self.goals)
